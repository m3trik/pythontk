#!/usr/bin/python
# coding=utf-8
"""Unit tests for the pythontk test runner's README-badge guard.

A partial run must not stamp the badge -- whether it was scoped by argument or
by environment (numpy/Pillow/Qt missing, so whole modules never import). See
m3trik/docs/TEST_BADGE_STANDARD.md.

Covers:
- discover_module_names: the expected module set is derived from disk, never
  recorded, so it cannot go stale.
- module_of / is_import_standin: unittest's synthetic stand-ins
  (ModuleImportFailure / ModuleSkipped) are attributed to the module that
  failed, and never counted as a run.
- StatusBadge.gate: refuses the stamp (with a printable reason) when a discovered
  module did not run, or when nothing ran at all.
- DetailedTestResult: end-to-end module coverage over a real discovery of a
  short, purpose-built module set.
"""
import importlib.util
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

from pythontk.core_utils.status_badge import StatusBadge
from pythontk.file_utils.temp_artifacts import TempArtifacts

TEST_DIR = Path(__file__).resolve().parent


class RunnerTestCase(unittest.TestCase):
    """Base: loads ``run_tests.py`` by path (it is a script, not a package)."""

    RUNNER_PATH = TEST_DIR / "run_tests.py"
    MODULE_NAME = "_pythontk_run_tests_under_test"

    @classmethod
    def load_runner(cls):
        """Import the runner module from its file path."""
        spec = importlib.util.spec_from_file_location(cls.MODULE_NAME, cls.RUNNER_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @classmethod
    def setUpClass(cls):
        cls.runner = cls.load_runner()


class TestModuleDiscovery(RunnerTestCase):
    """The expected module set comes off disk, matching unittest discovery."""

    def setUp(self):
        self.artifacts = TempArtifacts(
            "run_tests_probe", dir=str(TEST_DIR / "temp_tests"), policy="scoped"
        )
        self.probe_dir = Path(self.artifacts.dir_path())

    def tearDown(self):
        self.artifacts.cleanup()

    def test_matches_test_star_py_only(self):
        for name in ("test_alpha.py", "test_beta.py", "conftest.py", "helpers.py"):
            (self.probe_dir / name).write_text("", encoding="utf-8")

        self.assertEqual(
            StatusBadge.discover_module_names(self.probe_dir),
            {"test_alpha", "test_beta"},
        )

    def test_empty_dir_yields_empty_set(self):
        self.assertEqual(
            StatusBadge.discover_module_names(self.probe_dir), set()
        )

    def test_real_test_dir_is_non_trivial(self):
        """The live test tree must resolve to a real module set, not nothing."""
        found = StatusBadge.discover_module_names(TEST_DIR)
        self.assertIn(Path(__file__).stem, found)
        self.assertGreater(len(found), 1)


class TestBadgeGate(RunnerTestCase):
    """The gate refuses a stamp on any run that fell short of the module set."""

    def gate(self, expected, ran, passed=5, failed=0):
        return StatusBadge.gate(expected, ran, passed, failed)

    def test_allows_a_complete_run(self):
        allowed, reason = self.gate({"test_a", "test_b"}, {"test_a", "test_b"})
        self.assertTrue(allowed)
        self.assertEqual(reason, "")

    def test_allows_a_complete_run_with_failures(self):
        """A red run still stamps -- the badge exists to report the failures."""
        allowed, _ = self.gate({"test_a"}, {"test_a"}, passed=0, failed=3)
        self.assertTrue(allowed)

    def test_refuses_when_a_module_did_not_run(self):
        allowed, reason = self.gate({"test_a", "test_img"}, {"test_a"})
        self.assertFalse(allowed)
        self.assertIn("test_img", reason)
        self.assertIn("1 of 2", reason)

    def test_reason_truncates_a_long_missing_list(self):
        expected = {f"test_{i}" for i in range(12)}
        allowed, reason = self.gate(expected, set())
        self.assertFalse(allowed)
        self.assertIn("+6 more", reason)
        self.assertLess(len(reason), 200)

    def test_refuses_when_nothing_ran(self):
        allowed, reason = self.gate({"test_a"}, {"test_a"}, passed=0, failed=0)
        self.assertFalse(allowed)
        self.assertIn("no test cases ran", reason)

    def test_extra_ran_modules_do_not_block(self):
        allowed, _ = self.gate({"test_a"}, {"test_a", "test_b"})
        self.assertTrue(allowed)


class TestModuleCoverage(RunnerTestCase):
    """End-to-end: a short module set run through real unittest discovery."""

    MODULES = {
        # Imports fine, one passing case.
        "test_probe_ok.py": (
            "import unittest\n"
            "class TestOk(unittest.TestCase):\n"
            "    def test_passes(self):\n"
            "        self.assertTrue(True)\n"
        ),
        # The numpy/Pillow/Qt case: a hard module-level import that is missing.
        "test_probe_broken.py": (
            "import _probe_missing_dependency_xyz  # noqa: F401\n"
            "import unittest\n"
            "class TestBroken(unittest.TestCase):\n"
            "    def test_never_runs(self):\n"
            "        pass\n"
        ),
        # The other idiom: a guarded import that skips the module wholesale.
        "test_probe_module_skip.py": (
            "import unittest\n"
            "raise unittest.SkipTest('numpy unavailable')\n"
        ),
        # Imported and run, but every case is environment-gated.
        "test_probe_all_skipped.py": (
            "import unittest\n"
            "@unittest.skipUnless(False, 'toktx not installed')\n"
            "class TestGated(unittest.TestCase):\n"
            "    def test_gated(self):\n"
            "        pass\n"
        ),
    }

    def setUp(self):
        self.artifacts = TempArtifacts(
            "run_tests_probe", dir=str(TEST_DIR / "temp_tests"), policy="scoped"
        )
        self.probe_dir = Path(self.artifacts.dir_path())
        self._sys_path = list(sys.path)
        self._sys_modules = set(sys.modules)

    def tearDown(self):
        for name in set(sys.modules) - self._sys_modules:
            sys.modules.pop(name, None)
        sys.path[:] = self._sys_path
        self.artifacts.cleanup()

    def write_modules(self, *names):
        for name in names:
            (self.probe_dir / name).write_text(self.MODULES[name], encoding="utf-8")

    def run_probe(self):
        """Discover + run the probe dir, returning the runner's result object."""
        suite = unittest.TestLoader().discover(
            start_dir=str(self.probe_dir),
            pattern="test_*.py",
            top_level_dir=str(self.probe_dir),
        )
        return unittest.TextTestRunner(
            stream=StringIO(), verbosity=0, resultclass=self.runner.DetailedTestResult
        ).run(suite)

    def test_import_failure_and_module_skip_are_not_runs(self):
        self.write_modules(*self.MODULES)
        result = self.run_probe()

        self.assertEqual(
            result.modules_ran, {"test_probe_ok", "test_probe_all_skipped"}
        )
        self.assertEqual(result.modules_executed, {"test_probe_ok"})

    def test_gate_refuses_the_environment_scoped_run(self):
        self.write_modules(*self.MODULES)
        result = self.run_probe()
        expected = StatusBadge.discover_module_names(self.probe_dir)

        allowed, reason = StatusBadge.gate(
            expected, result.modules_ran, passed=1, failed=1
        )
        self.assertFalse(allowed)
        self.assertIn("test_probe_broken", reason)
        self.assertIn("test_probe_module_skip", reason)

    def test_gate_allows_a_run_where_every_module_ran(self):
        self.write_modules("test_probe_ok.py", "test_probe_all_skipped.py")
        result = self.run_probe()
        expected = StatusBadge.discover_module_names(self.probe_dir)

        allowed, reason = StatusBadge.gate(
            expected, result.modules_ran, passed=1, failed=0
        )
        self.assertTrue(allowed, reason)
        # A fully skipped module still counts as run (skips stay green), but the
        # runner names it -- it is the difference the warning reports.
        self.assertEqual(
            result.modules_ran - result.modules_executed, {"test_probe_all_skipped"}
        )


class TestRunLogCapture(RunnerTestCase):
    """A run log has to hold the OUTPUT the failure was explained by.

    unittest's runner writes through a ``ptk.TeeStream``, so its own report
    reaches the log; product code does not -- a bare print(), or the traceback.print_exc()
    of an exception the product swallowed, goes to the real stdout/stderr. Six
    instances of the full-suite file-IO flake were logged as unexplained
    assertion diffs because of exactly that (2026-08-20).
    """

    PROBE = (
        "import sys\n"
        "import unittest\n"
        "class TestNoisy(unittest.TestCase):\n"
        "    def test_prints(self):\n"
        "        print('PROBE-STDOUT-MARKER')\n"
        "        print('PROBE-STDERR-MARKER', file=sys.stderr)\n"
    )

    def setUp(self):
        self.artifacts = TempArtifacts(
            "run_tests_log_probe", dir=str(TEST_DIR / "temp_tests"), policy="scoped"
        )
        self.probe_dir = Path(self.artifacts.dir_path())
        (self.probe_dir / "test_probe_noisy.py").write_text(
            self.PROBE, encoding="utf-8"
        )
        self._sys_path = list(sys.path)
        self._sys_modules = set(sys.modules)

    def tearDown(self):
        for name in set(sys.modules) - self._sys_modules:
            sys.modules.pop(name, None)
        sys.path[:] = self._sys_path
        self.artifacts.cleanup()

    def run_probe(self):
        """Run the probe dir through the real TestRunner, returning its log.

        Both console streams are stood in for, so the probe's markers do not
        land in the parent run's own output and get mistaken for real failures.
        """
        runner = self.runner.TestRunner(self.probe_dir, verbosity=0)
        with mock.patch.object(sys, "stdout", StringIO()):
            with mock.patch.object(sys, "stderr", StringIO()):
                runner.run()
        return runner.log_buffer.getvalue()

    def test_product_stdout_reaches_the_log(self):
        self.assertIn("PROBE-STDOUT-MARKER", self.run_probe())

    def test_product_stderr_reaches_the_log(self):
        self.assertIn("PROBE-STDERR-MARKER", self.run_probe())

    def test_the_streams_are_restored_afterwards(self):
        runner = self.runner.TestRunner(self.probe_dir, verbosity=0)
        with mock.patch.object(sys, "stdout", StringIO()) as fake_out:
            with mock.patch.object(sys, "stderr", StringIO()) as fake_err:
                runner.run()
                # Read INSIDE the patch: mock.patch puts the attribute back on
                # exit whatever the runner left behind, so the same two asserts
                # after the context would pass with the restore deleted.
                self.assertIs(sys.stdout, fake_out)
                self.assertIs(sys.stderr, fake_err)


class TestBadgeWiring(RunnerTestCase):
    """The gate is wired into the badge write, not merely importable."""

    def make_result(self, modules_ran, tests_run=5):
        raw = unittest.TestResult()
        raw.testsRun = tests_run
        raw.modules_ran = set(modules_ran)
        raw.modules_executed = set(modules_ran)
        return self.runner.TestResult(raw, 0.0)

    def test_result_carries_module_coverage(self):
        result = self.make_result({"test_a"})
        self.assertEqual(result.modules_ran, {"test_a"})
        self.assertEqual(result.modules_executed, {"test_a"})

    def test_partial_run_never_writes_the_badge(self):
        result = self.make_result({"test_run_tests"})  # one module of many
        readme = TEST_DIR / "temp_tests" / "unwritten_README.md"

        with mock.patch.object(StatusBadge, "update_test_badge") as writer:
            stamped = self.runner.TestRunner.stamp_badge(TEST_DIR, readme, result)

        self.assertFalse(stamped)
        writer.assert_not_called()
        self.assertFalse(readme.exists())

    def test_complete_run_writes_the_badge(self):
        modules = StatusBadge.discover_module_names(TEST_DIR)
        result = self.make_result(modules)
        readme = TEST_DIR / "temp_tests" / "unwritten_README.md"

        with mock.patch.object(StatusBadge, "update_test_badge") as writer:
            stamped = self.runner.TestRunner.stamp_badge(TEST_DIR, readme, result)

        self.assertTrue(stamped)
        writer.assert_called_once()
        self.assertEqual(writer.call_args[0][1:3], (5, 0))  # passed, failed


if __name__ == "__main__":
    unittest.main()
