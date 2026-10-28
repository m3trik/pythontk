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


class _SettingsStore:
    """uitk's SettingsManager surface: ``<objectName>/<signal>`` keys;
    ``remove(name)`` drops the row's whole group."""

    def __init__(self, data):
        self.data, self.synced = dict(data), False

    def keys(self):
        return list(self.data)

    def value(self, key, default=None):
        return self.data.get(key, default)

    def setValue(self, key, value):
        self.data[key] = value

    def remove(self, key):
        for k in [k for k in self.data if k == key or k.startswith(key + "/")]:
            del self.data[k]

    def sync(self):
        self.synced = True


class TestHooks(unittest.TestCase):
    def test_the_host_task_manager_is_built_with_the_exporter_logger(self):
        exporter = _Exporter()
        self.assertIsInstance(exporter.task_manager, _StubTasks)
        self.assertIs(exporter.task_manager.logger, exporter.logger)
        self.assertEqual(exporter.logger.name.rsplit(".", 1)[-1], "_Exporter")

    def test_a_panel_passes_its_logger_and_dialogs_in(self):
        """A panel composes the exporter: its logger is adopted BEFORE the task
        manager is built (so every task logs into the panel), its dialogs answer
        the two seams, and ``log_level=None`` leaves the panel's level alone.
        The class-shared logger and the default seams are untouched."""
        import logging

        panel_logger = logging.getLogger("test_scene_exporter_panel_logger")
        panel_logger.setLevel(logging.INFO)
        asked = []

        def confirm(question):
            asked.append(question)
            return True

        def decide(check, messages, remaining):
            return "override_all"

        exporter = _Exporter(
            log_level=None,
            logger=panel_logger,
            confirm=confirm,
            decide_check_failure=decide,
        )
        self.assertIs(exporter.logger, panel_logger)
        self.assertIs(exporter.task_manager.logger, panel_logger)
        self.assertEqual(panel_logger.level, logging.INFO)
        self.assertIs(exporter.confirm, confirm)
        self.assertIs(exporter.decide_check_failure, decide)
        self.assertEqual(
            exporter.decide_check_failure("check_a", [], []), "override_all"
        )
        # The headless default routes through the instance's confirm.
        self.assertEqual(
            _Exporter.decide_check_failure(exporter, "check_a", ["bad"], []),
            "override",
        )
        self.assertTrue(asked)
        plain = _Exporter()
        self.assertIsNot(plain.logger, panel_logger)
        self.assertNotIn("confirm", vars(plain))

    def test_export_cancelled_reads_the_last_run(self):
        exporter = _Exporter()
        self.assertFalse(exporter.export_cancelled)
        exporter._export_cancelled = True
        self.assertTrue(exporter.export_cancelled)

    def test_the_export_folder_falls_back_to_the_saved_scene(self):
        exporter = _Exporter()
        self.assertEqual(exporter.resolve_export_dir(None), "")
        exporter._saved_scene_path = lambda: os.path.join("proj", "scenes", "a.ma")
        self.assertEqual(
            exporter.resolve_export_dir(None), os.path.join("proj", "scenes")
        )
        self.assertEqual(exporter.resolve_export_dir("out"), os.path.abspath("out"))

    def test_the_log_file_records_what_the_run_logs(self):
        """Measured 2026-10-09: a production export's ``.log`` was 0 bytes --
        the handler sat on a bare logger named after the class, which neither
        the exporter nor its task manager logs through. Closing removes only
        that handler: a caller's own on the same logger stays.
        Added: 2026-10-09"""
        import logging

        from pythontk.file_utils.temp_artifacts import TempArtifacts

        exporter = _Exporter(log_level="INFO")
        caller = logging.NullHandler()
        exporter.logger.addHandler(caller)
        self.addCleanup(exporter.logger.removeHandler, caller)
        with TempArtifacts("test_scene_exporter_log", policy="scoped") as temp:
            path = os.path.join(temp.dir_path(), "run.log")
            exporter.setup_file_logging(path)
            exporter.task_manager.logger.info("Executing Task #1/1: smart_bake")
            exporter.close_file_handlers()
            with open(path, encoding="utf-8") as fh:
                self.assertIn("Executing Task #1/1: smart_bake", fh.read())
        self.assertIn(caller, exporter.logger.handlers)
        self.assertFalse(
            [h for h in exporter.logger.handlers if isinstance(h, logging.FileHandler)]
        )

    def test_the_log_file_keeps_the_run_banner_at_every_level(self):
        """Measured 2026-10-10 on real Maya exports: at the panel's ERROR level
        a clean run's ``.log`` was 0 bytes, and at every level the EXPORT
        SUCCESSFUL box was missing -- the file opened in cp1252, so the box's
        glyphs raised on write. The banner bypasses the level, so with it
        written the file always records how the run ended; its markup and the
        records' links are stripped to plain text.
        Added: 2026-10-10"""
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        exporter = _Exporter(log_level="ERROR")
        with TempArtifacts("test_scene_exporter_banner", policy="scoped") as temp:
            path = os.path.join(temp.dir_path(), "run.log")
            exporter.setup_file_logging(path)
            try:
                exporter.logger.log_box(
                    "EXPORT SUCCESSFUL",
                    ["✓ File written successfully", "Path: café.fbx"],
                    level="SUCCESS",
                )
                link = exporter.logger.log_link("pCube1", "select", node="|pCube1")
                exporter.logger.error(f"Missing → {link}")
            finally:
                exporter.close_file_handlers()
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
        self.assertIn("EXPORT SUCCESSFUL", content)
        self.assertIn("✓ File written successfully", content)
        self.assertIn("Missing → pCube1", content)
        self.assertNotIn("<span", content)
        self.assertNotIn("<a ", content)

    def test_the_kept_edits_advice_is_the_hosts_word(self):
        exporter = _Exporter()
        exporter.task_manager.kept_edits = ["tied keys"]
        exporter.KEPT_EDITS_ADVICE = "Undo is on you."
        with self.assertLogs(exporter.logger, "WARNING") as log:
            exporter._warn_stopped_before_write("Stopped.")
        self.assertIn("tied keys. Undo is on you.", log.output[0])


class TestRunLog(unittest.TestCase):
    """The run's log: ONE file per scene in the per-user cache
    (``ptk.RunLog``), never one per export beside the deliverable."""

    def setUp(self):
        from unittest import mock

        from pythontk.core_utils.user_config import UserConfig
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        temp = TempArtifacts("test_scene_exporter_run_log", policy="scoped")
        self.addCleanup(temp.cleanup)
        root = temp.dir_path()
        self.out = os.path.join(root, "out")
        os.makedirs(self.out)
        env = mock.patch.dict(
            os.environ, {UserConfig.CACHE_ROOT_ENV_VAR: os.path.join(root, "cache")}
        )
        env.start()
        self.addCleanup(env.stop)
        self.exporter = _Exporter(log_level="INFO")
        self.addCleanup(self.exporter.close_file_handlers)
        self.exporter.export_dir = self.out
        self.scene = os.path.join(root, "shot1", "hero.ma")
        self.exporter._saved_scene_path = lambda: self.scene

    def _export(self, name, succeeded=True):
        self.exporter.export_path = os.path.join(self.out, name + ".fbx")
        self.exporter._setup_file_logging()
        self.exporter.logger.info("exporting " + name)
        self.exporter._close_file_logging(succeeded)

    def _read_log(self):
        with open(self.exporter.log_file_path(), encoding="utf-8") as fh:
            return fh.read()

    def test_every_export_of_a_scene_appends_to_one_log(self):
        """Measured 2026-10-13: each export wrote ``<export name>.log`` beside
        the deliverable, so a timestamped Output Filename left a new log in the
        delivery folder on every run, and nothing ever removed one.
        Added: 2026-10-13"""
        self._export("hero_2026-10-13_10-00-00")
        self._export("hero_2026-10-13_10-05-00", succeeded=False)
        self.assertEqual([f for f in os.listdir(self.out) if f.endswith(".log")], [])
        log = self.exporter.log_file_path()
        self.assertEqual(os.listdir(os.path.dirname(log)), [os.path.basename(log)])
        content = self._read_log()
        self.assertIn("exporting hero_2026-10-13_10-00-00", content)
        self.assertIn("exporting hero_2026-10-13_10-05-00", content)
        self.assertIn("Export succeeded", content)
        self.assertIn("Export failed", content)
        self.assertIn(self.scene, content)

    def test_another_scene_of_the_same_name_keeps_its_own_log(self):
        first = self.exporter.log_file_path()
        self.scene = os.path.join(os.path.dirname(self.out), "shot2", "hero.ma")
        self.assertNotEqual(self.exporter.log_file_path(), first)
        self.assertTrue(os.path.basename(first).startswith("hero_"), first)

    def test_the_log_keeps_debug_detail_the_panel_never_shows(self):
        """The panel has no Log Level option any more: it shows the run at
        INFO and the run log records down to DEBUG. Added: 2026-10-14"""
        import io
        import logging

        stream = io.StringIO()
        panel = logging.StreamHandler(stream)
        self.exporter.logger.addHandler(panel)
        self.addCleanup(self.exporter.logger.removeHandler, panel)
        self.exporter.export_path = os.path.join(self.out, "hero.fbx")
        self.exporter._setup_file_logging()
        self.exporter.logger.debug("per-object detail")
        self.exporter._close_file_logging(True)
        self.assertIn("per-object detail", self._read_log())
        self.assertNotIn("per-object detail", stream.getvalue())
        self.assertEqual(self.exporter.logger.level, logging.INFO)

    def test_the_header_names_the_files_the_format_ships(self):
        """A GLB-only run's export path names its temp FBX; the header must
        name the GLB that actually ships."""
        from types import SimpleNamespace

        self.exporter.task_manager.run = SimpleNamespace(output_format="glb")
        self._export("hero")
        header = next(
            ln for ln in self._read_log().splitlines() if ln.startswith("===== ")
        )
        self.assertIn(os.path.join(self.out, "hero.glb"), header)
        self.assertNotIn("hero.fbx", header)

    def test_a_cancelled_run_says_so(self):
        self.exporter._export_cancelled = True
        self._export("hero")
        self.assertIn("Export cancelled", self._read_log())

    def test_the_panel_gets_a_link_to_the_log(self):
        import io
        import logging

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        self.exporter.logger.addHandler(handler)
        self.addCleanup(self.exporter.logger.removeHandler, handler)
        self._export("hero")
        self.assertIn(self.exporter.log_file_path(), stream.getvalue())

    def _panel(self):
        import io
        import logging

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        self.exporter.logger.addHandler(handler)
        self.addCleanup(self.exporter.logger.removeHandler, handler)
        return stream

    def test_a_shipped_glb_ends_the_run_with_a_webxr_link(self):
        """The run's last line opens the GLB it shipped in the WebXR preview,
        once: the next run names its own file or none. Added: 2026-10-14"""
        stream = self._panel()
        glb = os.path.join(self.out, "hero.glb")
        self.exporter.export_path = os.path.join(self.out, "hero.fbx")
        self.exporter._setup_file_logging()
        self.exporter._glb_created(glb)
        self.exporter._close_file_logging(True)
        lines = stream.getvalue().strip().splitlines()
        self.assertIn("action://preview", lines[-1])
        self.assertIn("hero.glb", lines[-1])
        self.assertIn("GLB created", stream.getvalue())
        self.assertNotIn("action://preview", self._read_log(), "a panel link")

        self._export("prop")  # no GLB this time
        self.assertEqual(stream.getvalue().count("action://preview"), 1)

    def test_the_webxr_link_survives_a_log_that_never_opened(self):
        stream = self._panel()
        self.exporter._glb_created(os.path.join(self.out, "hero.glb"))
        self.exporter._close_file_logging(True)
        self.assertIn("action://preview", stream.getvalue())

    def test_a_successful_run_ends_with_a_link_to_its_folder(self):
        """The run's last line opens the folder the deliverable landed in,
        beside the WebXR link when a GLB shipped; a run that shipped nothing
        links no folder. Added: 2026-10-19"""
        import html
        import re
        from urllib.parse import parse_qs, urlsplit

        stream = self._panel()
        self._export("hero")
        last = stream.getvalue().strip().splitlines()[-1]
        href = re.search(r'href="(action://open[^"]+)"', last).group(1)
        query = parse_qs(urlsplit(html.unescape(href)).query)
        self.assertEqual(query["path"], [self.out])

        self.exporter._glb_created(os.path.join(self.out, "hero.glb"))
        self._export("hero")
        last = stream.getvalue().strip().splitlines()[-1]
        self.assertIn("action://open", last)
        self.assertIn("action://preview", last)

        before = stream.getvalue().count("Open containing folder")
        self._export("prop", succeeded=False)
        self.assertEqual(stream.getvalue().count("Open containing folder"), before)

    def test_old_logs_beside_deliverables_are_removed(self):
        """The logs earlier versions left beside each deliverable go at the
        next export into that folder -- the exporter's own only: a log naming
        no deliverable there, or holding text the exporter never wrote, stays.
        Added: 2026-10-13"""
        stamp = "2026-10-01 10:00:00,123 - INFO - "
        files = {
            "hero.fbx": "",
            "hero.log": stamp + "Executing Task #1/1\n",
            "hero_v002.glb": "",
            ".hero_v002.log": "",  # empty: the 2026-10-09 wrong-logger bug
            "prop.usd": "",
            "prop.log": "╔══╗\n",  # a box first (ERROR level)
            "notes.log": stamp + "no deliverable of this name\n",
            "rig.fbx": "",
            "rig.log": "the artist's own notes\n",
        }
        for name, text in files.items():
            with open(os.path.join(self.out, name), "w", encoding="utf-8") as fh:
                fh.write(text)
        self._export("hero")
        left = sorted(f for f in os.listdir(self.out) if f.endswith(".log"))
        self.assertEqual(left, ["notes.log", "rig.log"])

    def test_a_log_that_cannot_open_never_stops_the_export(self):
        from unittest import mock

        from pythontk.core_utils.logging_mixin import RunLog

        self.exporter.export_path = os.path.join(self.out, "hero.fbx")
        for error in (
            PermissionError("denied"),  # an unwritable cache
            UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogate"),  # the header
        ):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(RunLog, "attach", side_effect=error):
                    with self.assertLogs(self.exporter.logger, "WARNING") as log:
                        self.exporter._setup_file_logging()
                self.assertIn("could not be opened", log.output[0])
                self.assertIsNone(getattr(self.exporter, "file_handler", None))
                self.exporter._close_file_logging(True)  # nothing open: a no-op

    def test_generate_log_file_path_is_retired(self):
        with self.assertWarns(DeprecationWarning):
            got = self.exporter.generate_log_file_path("ignored.fbx")
        self.assertEqual(got, self.exporter.log_file_path())


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

    def test_a_preset_picked_says_what_its_retired_rows_no_longer_do(self):
        """REGRESSION (2026-10-19): a released preset asking for a check whose
        row retired loaded with the check gone and nothing said. Both panels
        call this when a preset is picked."""
        exporter = _Exporter()
        exporter.task_manager.task_definitions = {
            "snap_keys_to_frame": {"widget_type": "QCheckBox", "setText": "Snap"}
        }
        stored = {"snap_keys_to_frame": False, "check_floating_point_keys": True}
        with self.assertLogs(exporter.logger, level="WARNING") as caught:
            exporter.note_retired_preset_rows("Unity", stored)
        self.assertIn("Preset 'Unity'", caught.output[0])
        self.assertIn("Floating Point Keys", caught.output[0])
        with self.assertNoLogs(exporter.logger, level="DEBUG"):
            exporter.note_retired_preset_rows("Unity", None)
            exporter.note_retired_preset_rows("", stored)

    def test_saved_panel_state_says_once_what_its_retired_rows_no_longer_do(self):
        """REGRESSION (2026-10-19): the panel's own saved state (not a preset)
        kept a retired check row's value where no widget restores it, so a
        user whose Snap Keys To Frame was saved off lost the check its row had
        run, with nothing said. Both panels call this on init; the retired
        keys go once said, so it is said once."""
        exporter = _Exporter()
        exporter.task_manager.task_definitions = {
            "snap_keys_to_frame": {
                "widget_type": "QCheckBox",
                "setText": "Snap Keys To Frame",
                "setChecked": True,
            },
            "conform_shape_names": {
                "widget_type": "QCheckBox",
                "setText": "Fix Mangled Names",
                "setChecked": True,
            },
        }
        store = _SettingsStore(
            {
                "snap_keys_to_frame/toggled": False,
                "check_floating_point_keys/toggled": True,
                # Its fix never saved a value: the row starts at its default (on).
                "check_mangled_names/toggled": False,
                "baked_reflections/currentIndexChanged": "Quarter",
                "cmb004/currentIndexChanged": 1,
            }
        )
        with self.assertLogs(exporter.logger, level="INFO") as caught:
            exporter.note_retired_panel_state(store)
        warned = [r.getMessage() for r in caught.records if r.levelname == "WARNING"]
        said = "\n".join(caught.output)
        self.assertTrue(
            any("Floating Point Keys (Snap Keys To Frame)" in m for m in warned), said
        )
        self.assertIn("Mangled Names (Fix Mangled Names)", said)
        self.assertIn("Baked Reflections to Quarter", said)
        self.assertEqual(
            sorted(store.data),
            ["cmb004/currentIndexChanged", "snap_keys_to_frame/toggled"],
            "the retired rows' keys go; a live row's stay",
        )
        self.assertTrue(store.synced, "the removal is flushed, so it holds")
        with self.assertNoLogs(exporter.logger, level="DEBUG"):
            exporter.note_retired_panel_state(store)

    def test_a_saved_regex_field_folds_into_the_saved_output_filename_once(self):
        """REGRESSION (2026-10-19): both panels read the retired RegEx field as
        ``settings.value("txt002")``, but the switchboard saves it under
        ``txt002/textChanged``, so a saved rule never migrated; and folded
        into the widget, the field's own restore would have overwritten it.
        The fold now lands where the Output Filename restores from."""
        exporter = _Exporter()
        store = _SettingsStore(
            {
                "txt001/textChanged": "WIP_*",
                "txt002/textChanged": "test_|prod_",
                "cmb004/currentIndexChanged": 1,
            }
        )
        saved = []

        def save(pattern):  # the panel's ui.state.save(txt001, pattern)
            saved.append(pattern)
            store.setValue("txt001/textChanged", pattern)

        with self.assertLogs(exporter.logger, level="INFO") as caught:
            folded = exporter.migrate_legacy_regex(store, "txt001", save)
        self.assertEqual(folded, "WIP_{scene:test_->prod_}")
        self.assertEqual(saved, [folded])
        self.assertIn(folded, caught.output[0])
        self.assertEqual(
            sorted(store.data),
            ["cmb004/currentIndexChanged", "txt001/textChanged"],
            "the retired key goes; a live row's stay",
        )
        self.assertTrue(store.synced, "the removal is flushed, so it holds")
        with self.assertNoLogs(exporter.logger, level="DEBUG"):
            self.assertIsNone(exporter.migrate_legacy_regex(store, "txt001", save))
        self.assertEqual(saved, [folded], "once")

    def test_a_regex_whose_fold_was_not_stored_is_kept_for_the_next_open(self):
        """The retired key goes only once the fold is stored: a save that wrote
        nothing (a suppressed store) must not cost the user the rule."""
        exporter = _Exporter()
        store = _SettingsStore({"txt002/textChanged": "_bar.*"})
        self.assertIsNone(
            exporter.migrate_legacy_regex(store, "txt001", lambda pattern: None)
        )
        self.assertEqual(store.data, {"txt002/textChanged": "_bar.*"})

    def test_a_blank_regex_field_just_goes(self):
        exporter = _Exporter()
        store = _SettingsStore({"txt002/textChanged": "  "})
        saved = []
        with self.assertNoLogs(exporter.logger, level="INFO"):
            self.assertIsNone(
                exporter.migrate_legacy_regex(store, "txt001", saved.append)
            )
        self.assertEqual((store.data, saved), ({}, []))

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
