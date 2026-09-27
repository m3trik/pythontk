#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk script_run (run_script_to_artifact / ScriptRunResult).

Uses ``sys.executable`` as the "app" so no DCC is required: the scripts under test
are plain Python that create (or fail to create) the expected artifact.

Run with:
    python -m pytest test_script_run.py -v
    python test_script_run.py
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from conftest import find_mayapy
from pythontk.core_utils.app_launcher import AppLauncher
from pythontk.core_utils.cancel_scope import CancelScope, OperationCancelled
from pythontk.core_utils.handoff.script_run import (
    REWRITTEN,
    ProgressRelay,
    ScriptRunner,
    ScriptRunResult,
)


class ScriptRunBase(unittest.TestCase):
    # Test-owned prefix so kept-on-failure scripts can be swept from the real
    # temp dir in tearDown without touching anyone else's script_run_* files.
    SCRIPT_PREFIX = "sr_test_script"

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sr_test_")
        self.artifact = os.path.join(self.dir, "out.bin")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        pattern = os.path.join(tempfile.gettempdir(), f"{self.SCRIPT_PREFIX}_*")
        for leftover in glob.glob(pattern):
            try:
                os.remove(leftover)
            except OSError:
                pass

    def run_script(self, script, **kwargs):
        kwargs.setdefault("artifact", self.artifact)
        kwargs.setdefault("script_prefix", self.SCRIPT_PREFIX)
        return ScriptRunner.run_script_to_artifact(sys.executable, script, **kwargs)


class TestSuccess(ScriptRunBase):
    def test_returns_result_with_artifact(self):
        script = (
            "import sys\n"
            "print('converting...')\n"
            f"open({self.artifact!r}, 'wb').write(b'payload')\n"
        )
        result = self.run_script(script)
        self.assertIsInstance(result, ScriptRunResult)
        self.assertEqual(result.artifact, self.artifact)
        self.assertTrue(os.path.isfile(result.artifact))
        self.assertEqual(result.returncode, 0)
        self.assertIn("converting...", result.output)
        self.assertGreaterEqual(result.duration, 0.0)

    def test_script_file_removed_on_success(self):
        script = f"open({self.artifact!r}, 'wb').write(b'x')\n"
        result = self.run_script(script)
        self.assertFalse(
            os.path.exists(result.script_path),
            "the rendered temp script must be cleaned up after a successful run",
        )

    def test_nonzero_exit_with_artifact_still_succeeds(self):
        # Success is judged by the artifact, not the exit code (DCC standalone
        # teardown is a known crasher — the artifact is the ground truth).
        script = (
            f"import os, sys\nopen({self.artifact!r}, 'wb').write(b'x')\nsys.exit(9)\n"
        )
        result = self.run_script(script)
        self.assertEqual(result.returncode, 9)
        self.assertTrue(os.path.isfile(result.artifact))

    def test_launch_args_shape_the_argv(self):
        # The default runs a Python interpreter off its env; a custom mapper must win.
        script = f"open({self.artifact!r}, 'wb').write(b'x')\n"
        seen = {}

        def mapper(script_path):
            seen["path"] = script_path
            return [script_path]

        self.run_script(script, launch_args=mapper)
        self.assertTrue(seen["path"].endswith(".py"))


class TestFailure(ScriptRunBase):
    def test_missing_artifact_raises_with_output_tail(self):
        script = "print('MARKER-42'); raise SystemExit(1)\n"
        with self.assertRaises(RuntimeError) as ctx:
            self.run_script(script)
        self.assertIn("MARKER-42", str(ctx.exception))

    def test_empty_artifact_is_a_failure(self):
        script = f"open({self.artifact!r}, 'wb').close()\n"
        with self.assertRaises(RuntimeError):
            self.run_script(script)

    def test_traceback_from_stderr_is_captured(self):
        script = "raise ValueError('kaboom-77')\n"
        with self.assertRaises(RuntimeError) as ctx:
            self.run_script(script)
        self.assertIn("kaboom-77", str(ctx.exception))

    def test_script_file_kept_on_failure(self):
        script = "raise SystemExit(1)\n"
        kept = None
        try:
            self.run_script(script)
        except RuntimeError as e:
            kept = getattr(e, "script_path", None)
        self.assertIsNotNone(kept, "the failure must carry the kept script path")
        self.assertTrue(os.path.exists(kept), "the script must survive for debugging")
        os.remove(kept)

    def test_stale_preexisting_artifact_does_not_fake_success(self):
        # A leftover artifact from a prior run must not mask a failed script:
        # the artifact check is the success criterion, so it must be judged
        # against THIS run's output only.
        with open(self.artifact, "wb") as f:
            f.write(b"stale bytes from a previous run")
        script = "raise SystemExit(1)\n"
        with self.assertRaises(RuntimeError) as ctx:
            self.run_script(script)
        os.remove(getattr(ctx.exception, "script_path"))

    def test_timeout_kills_and_raises(self):
        script = "import time; time.sleep(60)\n"
        with self.assertRaises(subprocess.TimeoutExpired) as ctx:
            self.run_script(script, timeout=3)
        # Same debuggability contract as the RuntimeError path: the timeout
        # carries the kept script's location.
        kept = getattr(ctx.exception, "script_path", None)
        self.assertIsNotNone(kept, "TimeoutExpired must carry the kept script path")
        self.assertTrue(os.path.exists(kept))


class TestRewrittenExpectation(ScriptRunBase):
    """``expect=REWRITTEN``: the artifact is also the app's INPUT.

    Round-trip hand-offs export a payload, hand it to the app to edit in place, and
    re-import it. Clearing the path first (the CREATED contract) would delete the
    input; and an app that exits cleanly without saving must not read as success, or
    the caller silently re-ingests its own untouched export.
    """

    def test_the_input_is_not_cleared_before_the_run(self):
        with open(self.artifact, "wb") as fh:
            fh.write(b"exported")
        script = (
            f"data = open({self.artifact!r}, 'rb').read()\n"
            "assert data == b'exported', 'the runner destroyed the input'\n"
            f"open({self.artifact!r}, 'wb').write(b'edited-by-the-app')\n"
        )
        result = self.run_script(script, expect=REWRITTEN)
        self.assertEqual(result.returncode, 0)
        with open(self.artifact, "rb") as fh:
            self.assertEqual(fh.read(), b"edited-by-the-app")

    def test_an_untouched_artifact_is_a_failure(self):
        with open(self.artifact, "wb") as fh:
            fh.write(b"exported")
        # Exits 0, never saves — the silent no-op this contract exists to catch.
        with self.assertRaises(RuntimeError) as ctx:
            self.run_script("print('did nothing')\n", expect=REWRITTEN)
        self.assertIn("unchanged on disk", str(ctx.exception))
        os.remove(getattr(ctx.exception, "script_path"))
        # The user's file is left exactly as it was.
        with open(self.artifact, "rb") as fh:
            self.assertEqual(fh.read(), b"exported")

    def test_a_size_change_alone_counts_as_written(self):
        """Same-second writes can share an mtime, so size is the second signal."""
        with open(self.artifact, "wb") as fh:
            fh.write(b"x")
        script = f"open({self.artifact!r}, 'wb').write(b'xxxxxxxxxxxx')\n"
        self.assertIsNotNone(self.run_script(script, expect=REWRITTEN))

    def test_an_emptied_artifact_is_still_a_failure(self):
        with open(self.artifact, "wb") as fh:
            fh.write(b"exported")
        script = f"open({self.artifact!r}, 'wb').close()\n"
        with self.assertRaises(RuntimeError) as ctx:
            self.run_script(script, expect=REWRITTEN)
        os.remove(getattr(ctx.exception, "script_path"))

    def test_created_remains_the_default(self):
        with open(self.artifact, "wb") as fh:
            fh.write(b"stale")
        script = (
            f"import os; assert not os.path.exists({self.artifact!r}), 'stale kept'\n"
            f"open({self.artifact!r}, 'wb').write(b'fresh')\n"
        )
        self.assertIsNotNone(self.run_script(script))

    def test_an_unknown_expectation_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_script("pass\n", expect="whatever")


class TestStreaming(ScriptRunBase):
    """``on_output``: a long headless run reports while it works and can be stopped.

    A blocking run used to be silent until the child exited, so a UI waiting on a
    minutes-long DCC conversion could show nothing, and the only way out was the
    timeout -- which killed a production conversion that was still working.
    """

    def _scripts_left(self):
        pattern = os.path.join(tempfile.gettempdir(), f"{self.SCRIPT_PREFIX}_*")
        return glob.glob(pattern)

    def test_lines_arrive_while_the_child_is_still_running(self):
        script = (
            "import time\n"
            "print('first', flush=True)\n"
            "time.sleep(0.6)\n"
            "print('second', flush=True)\n"
            f"open({self.artifact!r}, 'wb').write(b'x')\n"
        )
        seen = []
        result = self.run_script(script, on_output=seen.append)
        self.assertEqual(
            [line for line in seen if line is not None][:2], ["first", "second"]
        )
        # Quiet polls BETWEEN the two lines prove the first was delivered before
        # the child finished, not collected at exit.
        between = seen[seen.index("first") : seen.index("second")]
        self.assertIn(None, between)
        self.assertIn("second", result.output)

    def test_returning_false_kills_the_child_and_leaves_nothing(self):
        script = (
            "import time\n"
            "print('ready', flush=True)\n"
            "time.sleep(60)\n"
            f"open({self.artifact!r}, 'wb').write(b'x')\n"
        )
        started = time.monotonic()
        with self.assertRaises(OperationCancelled):
            self.run_script(script, on_output=lambda line: line != "ready")
        self.assertLess(time.monotonic() - started, 30, "the child was not killed")
        self.assertFalse(os.path.exists(self.artifact))
        self.assertEqual(self._scripts_left(), [])

    def test_a_cancelled_ambient_scope_stops_the_child(self):
        script = "import time\nprint('ready', flush=True)\ntime.sleep(60)\n"
        scope = CancelScope("test run")

        def on_output(line):
            if line == "ready":
                scope.cancel("test")

        with scope:
            with self.assertRaises(OperationCancelled):
                self.run_script(script, on_output=on_output)

    def test_the_timeout_still_applies_while_streaming(self):
        script = "import time\nwhile True:\n    print('tick', flush=True)\n    time.sleep(0.05)\n"
        with self.assertRaises(subprocess.TimeoutExpired) as ctx:
            self.run_script(script, timeout=2, on_output=lambda line: True)
        kept = getattr(ctx.exception, "script_path", None)
        self.assertTrue(kept and os.path.exists(kept))
        os.remove(kept)

    def test_a_failed_streamed_run_still_embeds_its_output(self):
        script = "print('MARKER-99', flush=True)\nraise SystemExit(1)\n"
        with self.assertRaises(RuntimeError) as ctx:
            self.run_script(script, on_output=lambda line: True)
        self.assertIn("MARKER-99", str(ctx.exception))
        os.remove(getattr(ctx.exception, "script_path"))

    def test_output_file_and_on_output_are_exclusive(self):
        with self.assertRaises(ValueError):
            AppLauncher.run(
                sys.executable,
                args=["-c", "pass"],
                output_file=os.path.join(self.dir, "log.txt"),
                on_output=lambda line: True,
            )

    def test_a_relay_turns_markers_into_bar_positions(self):
        script = (
            "print('::progress:: 1/2 half', flush=True)\n"
            "print('::progress:: 2/2 done', flush=True)\n"
            f"open({self.artifact!r}, 'wb').write(b'x')\n"
        )
        reports = []
        relay = ProgressRelay(
            lambda current, total, text: reports.append((current, text))
        )
        self.run_script(script, on_output=relay.reader(0, "Child"))
        self.assertIn((50, "Child: half"), reports)
        self.assertIn((100, "Child: done"), reports)


class TestNonAsciiScriptDir(unittest.TestCase):
    """The temp script lives under %TEMP%, which holds the user's name.

    mayapy.exe decodes its command line in the ANSI code page: measured on Maya
    2025, "José" arrived as "Jos\\udce9" and "Жук" as "???", so ``mayapy <script>``
    for a user so named failed with "can't open file" and the run produced nothing.
    The script's path must reach the interpreter some other way.
    """

    WORD = "José Жук"
    PROBE = (
        "import json, os, sys\n"
        "with open(OUT, 'w', encoding='utf-8') as fh:\n"
        "    json.dump({'argv': sys.argv, 'file': __file__, 'name': __name__,\n"
        "               'path0': sys.path[0],\n"
        "               'var': os.environ.get('PYTHONTK_ARGV')}, fh)\n"
    )

    def setUp(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.dir = os.path.join(here, "temp_tests", f"sr {self.WORD} {os.getpid()}")
        os.makedirs(self.dir)
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.artifact = os.path.join(self.dir, "out.json")
        # TempArtifacts' default root -- where the runner writes its script.
        patcher = mock.patch.object(tempfile, "tempdir", self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, exe):
        script = f"OUT = {self.artifact!r}\n" + self.PROBE
        result = ScriptRunner.run_script_to_artifact(
            exe, script, artifact=self.artifact, timeout=300
        )
        with open(self.artifact, encoding="utf-8") as fh:
            return result, json.load(fh)

    def test_no_path_reaches_the_command_line(self):
        seen = {}

        def fake_run(app, args=None, env=None, **kwargs):
            seen.update(args=list(args), env=env)
            with open(self.artifact, "w") as fh:
                fh.write("x")
            return subprocess.CompletedProcess(args, 0, "", "")

        with mock.patch.object(AppLauncher, "run", side_effect=fake_run):
            result = ScriptRunner.run_script_to_artifact(
                "mayapy", "pass", artifact=self.artifact
            )
        self.assertIn(self.WORD, result.script_path)
        self.assertTrue(all(arg.isascii() for arg in seen["args"]), seen["args"])
        carried = json.loads(seen["env"][AppLauncher.PYTHON_ARGV_VAR])
        self.assertEqual(carried, [result.script_path])

    def test_the_script_runs_as_python_script_py_would(self):
        result, seen = self._run(sys.executable)
        self.assertEqual(seen["argv"], [result.script_path])
        self.assertEqual(seen["file"], result.script_path)
        self.assertEqual(seen["name"], "__main__")
        self.assertEqual(seen["path0"], os.path.dirname(result.script_path))
        # Consumed before the script runs: nothing it spawns inherits it.
        self.assertIsNone(seen["var"])

    @unittest.skipUnless(find_mayapy(), "mayapy.exe not installed")
    def test_mayapy_runs_a_script_under_a_non_ascii_temp(self):
        """The measured failure itself, through a real mayapy."""
        result, seen = self._run(find_mayapy())
        self.assertEqual(seen["file"], result.script_path)
        self.assertEqual(seen["argv"], [result.script_path])

    @unittest.skipUnless(find_mayapy(), "mayapy.exe not installed")
    def test_maya_saves_a_scene_under_a_temp_its_code_page_cannot_hold(self):
        """Maya itself opens and saves through the ANSI code page. With TEMP under
        "José Жук" (cp1252 cannot hold "Жук") it read TEMP as "???", put its own
        temp files in the CURRENT directory, and ``cmds.file`` save said "An
        invalid path was specified". The child gets its TEMP, and the template the
        path it saves to, in 8.3 form."""
        import glob

        import pythontk
        from pythontk.core_utils.handoff.script_template import ScriptTemplate

        if (AppLauncher._short_name(self.dir) or "") == self.dir:
            self.skipTest("no 8.3 short names on this volume")
        artifact = os.path.join(self.dir, "cube.ma")
        report = os.path.join(self.dir, "report.json")
        root = os.path.dirname(os.path.dirname(os.path.abspath(pythontk.__file__)))
        script = (
            "import json, os, sys\n"
            "import maya.standalone\n"
            "maya.standalone.initialize(name='python')\n"
            "import maya.cmds as cmds\n"
            "cmds.polyCube(name='A5Cube')\n"
            f"cmds.file(rename=r'{ScriptTemplate.child_path(artifact)}')\n"
            "cmds.file(save=True, type='mayaAscii', force=True)\n"
            f"json.dump(cmds.internalVar(userTmpDir=True), open({report!r}, 'w'))\n"
            # Not os._exit: it runs Maya's faulting DLL detach, which files a
            # crash dump and an untitled[Recovered-...].ma into the child's TEMP.
            f"sys.path.insert(0, {root!r})\n"
            "from pythontk.core_utils.process_exit import ProcessExit\n"
            "ProcessExit.hard_exit(0)\n"
        )
        env = dict(os.environ, TEMP=self.dir, TMP=self.dir, MAYA_SKIP_USERSETUP_PY="1")
        ScriptRunner.run_script_to_artifact(
            find_mayapy(), script, artifact=artifact, env=env, timeout=600
        )
        with open(artifact, encoding="utf-8", errors="replace") as fh:
            self.assertIn("A5Cube", fh.read())
        with open(report, encoding="utf-8") as fh:
            maya_temp = json.load(fh)
        self.assertTrue(os.path.samefile(maya_temp, self.dir), ascii(maya_temp))
        # The child's TEMP holds no crash-save: no recovered scene, no crash log.
        crashed = glob.glob(os.path.join(self.dir, "*[[]Recovered-*"))
        crashed += glob.glob(os.path.join(self.dir, "MayaCrashLog*"))
        self.assertEqual(crashed, [])


class TestProgressRelay(unittest.TestCase):
    def test_marker_round_trip(self):
        line = ProgressRelay.line(3, 7, "Baking animation")
        self.assertEqual(line, "::progress:: 3/7 Baking animation")
        self.assertEqual(ProgressRelay.parse(line), (3, 7, "Baking animation"))

    def test_ordinary_output_is_not_a_marker(self):
        self.assertIsNone(ProgressRelay.parse("Warning: something"))
        self.assertIsNone(ProgressRelay.parse(None))
        self.assertIsNone(ProgressRelay.parse("::progress:: x/y text"))
        # A host that prefixes its console lines still delivers the marker.
        self.assertEqual(ProgressRelay.parse("# ::progress:: 1/2"), (1, 2, ""))

    def test_stages_share_one_bar_and_it_never_moves_back(self):
        calls = []
        relay = ProgressRelay(lambda c, t, m: calls.append((c, t, m)), stages=2)
        relay.reader(1, "Blender")("::progress:: 1/2 Importing")
        self.assertEqual(calls[-1], (75, 100, "Blender: Importing"))
        relay.report(0, 1, 1, "late")
        self.assertEqual(calls[-1][0], 75)
        self.assertEqual(relay.value, 75)

    def test_quiet_output_becomes_throttled_keep_alive_ticks(self):
        calls = []
        relay = ProgressRelay(lambda c, t, m: calls.append(c), throttle=60)
        reader = relay.reader(0)
        reader(None)
        reader("plain output")
        reader(None)
        self.assertEqual(calls, [None])

    def test_a_false_receiver_cancels(self):
        relay = ProgressRelay(lambda c, t, m: False)
        self.assertFalse(relay.report(0, 1, 2, "x"))
        self.assertFalse(relay.reader(0)("::progress:: 1/2 x"))

    def test_no_receiver_always_continues(self):
        relay = ProgressRelay()
        self.assertTrue(relay.report(0, 1, 2))
        self.assertTrue(relay.tick())


class TestRootExport(unittest.TestCase):
    def test_registered_on_package_root(self):
        import pythontk as ptk

        self.assertTrue(hasattr(ptk, "ScriptRunner"))
        self.assertTrue(callable(ptk.ScriptRunner.run_script_to_artifact))
        self.assertTrue(hasattr(ptk, "ScriptRunResult"))
        self.assertIs(ptk.ProgressRelay, ProgressRelay)


if __name__ == "__main__":
    unittest.main(verbosity=2)
