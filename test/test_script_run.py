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
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from pythontk.core_utils.script_run import (
    REWRITTEN,
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
        # Interpreter-style default is [script]; a custom mapper must win.
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


class TestRootExport(unittest.TestCase):
    def test_registered_on_package_root(self):
        import pythontk as ptk

        self.assertTrue(hasattr(ptk, "ScriptRunner"))
        self.assertTrue(callable(ptk.ScriptRunner.run_script_to_artifact))
        self.assertTrue(hasattr(ptk, "ScriptRunResult"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
