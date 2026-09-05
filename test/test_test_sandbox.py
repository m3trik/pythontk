# !/usr/bin/python
# coding=utf-8
"""Tests for :class:`pythontk.TestSandbox` -- the process-level test isolation.

The runner activates the sandbox before discovery, so these run INSIDE it and
assert what it does rather than construct a second one; a standalone run of
this module activates it in ``setUp``.
"""

import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
import webbrowser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk.core_utils.test_sandbox import TestSandbox
from pythontk.file_utils.temp_artifacts import TempArtifacts

URL = "http://127.0.0.1:1/"


class TestSandboxTestCase(unittest.TestCase):
    def setUp(self):
        TestSandbox.activate()
        self._launches_before = len(TestSandbox.launches)

    def tearDown(self):
        # A refusal this test provoked on purpose must not read as a leak in
        # the runner's end-of-run report.
        del TestSandbox.launches[self._launches_before :]

    def test_a_browser_launch_is_refused_loudly_and_recorded(self):
        for name in ("open", "open_new", "open_new_tab"):
            with self.subTest(launcher=name):
                with self.assertRaises(RuntimeError) as caught:
                    getattr(webbrowser, name)(URL)
                self.assertIn("open_browser=False", str(caught.exception))
        self.assertEqual(TestSandbox.launches[self._launches_before :], [URL] * 3)

    def test_a_test_that_patches_the_launcher_gets_its_mock_then_the_guard_back(self):
        with unittest.mock.patch("webbrowser.open", return_value=True) as opened:
            self.assertTrue(webbrowser.open(URL))
        opened.assert_called_once_with(URL)
        self.assertTrue(TestSandbox.is_active())

    def test_the_preview_server_cannot_open_a_tab_from_here(self):
        """The guard reaches the one production launcher the suites hit."""
        from pythontk.net_utils.preview.server import PreviewServer

        server = PreviewServer(port=0)
        try:
            with self.assertRaises(RuntimeError):
                server.open_in_browser()
        finally:
            server.stop()

    def test_the_temp_dir_is_one_throwaway_root_that_children_inherit(self):
        root = TestSandbox.temp()
        self.assertTrue(os.path.isdir(root))
        self.assertEqual(tempfile.gettempdir(), root)
        for name in ("TMPDIR", "TEMP", "TMP"):
            self.assertEqual(os.environ[name], root)
        # Every default-dir store lands inside it...
        store = TempArtifacts("sandbox_probe", policy="scoped")
        self.assertEqual(os.path.dirname(store.path()), root)
        store.cleanup()
        # ...and so does a child process's.
        child = subprocess.check_output(
            [sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"],
            text=True,
        ).strip()
        self.assertEqual(os.path.normcase(child), os.path.normcase(root))

    def test_activate_is_idempotent(self):
        self.assertEqual(TestSandbox.activate(), TestSandbox.activate())
        self.assertTrue(TestSandbox.is_active())


class PytestLeakGateTest(unittest.TestCase):
    """The browser-leak check has to gate the run CI actually performs.

    ``run_tests.py`` reads ``TestSandbox.launches`` and exits 1 on a leak, but
    ``.github/workflows/tests.yml`` runs bare ``pytest ... test/`` -- which
    loads ``conftest.py``, not the runner. The sandbox was armed there and
    nothing read the record, so a swallowed launch left CI green: five
    downstream suites arm the guard and no reader existed on that path.

    These run pytest in a subprocess against a synthetic test, so they measure
    the real exit status rather than a stand-in for it.
    """

    def _run_pytest(self, body: str):
        """Write a one-test file into a sandbox dir and run pytest on it with
        the repo's conftest in scope; return the CompletedProcess."""
        import textwrap

        from pythontk.file_utils.temp_artifacts import TempArtifacts

        # In test/ so the repo conftest -- the thing under test -- is in scope.
        # The prefix must keep pytest's test_*.py shape or nothing is collected.
        scratch = TempArtifacts(
            "test_leakgate", policy="scoped", dir=os.path.dirname(__file__)
        )
        self.addCleanup(scratch.cleanup)
        path = scratch.path(extension=".py")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(textwrap.dedent(body))
        return subprocess.run(
            [sys.executable, "-m", "pytest", path, "-q", "-p", "no:cacheprovider"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )

    def test_a_swallowed_browser_launch_fails_the_pytest_run(self):
        """The whole point: the test itself passes, and the run must not."""
        proc = self._run_pytest(
            """
            import webbrowser

            def test_swallows_a_refused_launch():
                try:
                    webbrowser.open("http://leaked.example/page")
                except Exception:
                    pass  # exactly the broad except this gate exists to catch
            """
        )
        self.assertIn("1 passed", proc.stdout, proc.stdout + proc.stderr)
        self.assertNotEqual(
            proc.returncode,
            0,
            "a swallowed browser launch must fail the pytest run:\n" + proc.stdout,
        )
        self.assertIn("leaked.example", proc.stdout + proc.stderr)

    def test_a_clean_run_still_exits_zero(self):
        proc = self._run_pytest(
            """
            def test_touches_no_browser():
                assert True
            """
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class CollectorDriftTest(unittest.TestCase):
    """The two collectors must see the same tests.

    This suite is run two ways -- ``run_tests.py`` discovers with ``unittest``,
    CI runs ``pytest test/`` -- and each is blind to things the other collects.
    A module-level ``def test_x()`` is invisible to unittest discovery; a
    ``TestCase`` in a file whose name does not match the pattern is invisible
    to both. Either way the tests silently never run, and nothing reports it,
    because a collector cannot miss what it never saw.

    Counts, not names: a name-level diff would have to model both collectors'
    id grammars, and drift shows up in the count first either way.
    """

    def test_pytest_and_unittest_collect_the_same_number_of_tests(self):
        recorded = os.environ.get("PYTHONTK_PYTEST_COLLECTED")
        if recorded is None:
            self.skipTest(
                "no whole-directory pytest count recorded (unittest runner, or "
                "a subset run -- run_tests.py owns its own count)"
            )
        collected = int(recorded)

        here = os.path.dirname(os.path.abspath(__file__))
        loader = unittest.defaultTestLoader
        suite = loader.discover(here, pattern="test_*.py", top_level_dir=here)

        def count(s):
            return sum(count(x) if isinstance(x, unittest.TestSuite) else 1 for x in s)

        discovered = count(suite)
        self.assertFalse(
            loader.errors,
            "unittest discovery hit import errors:\n"
            + "\n".join(str(e) for e in loader.errors),
        )
        self.assertEqual(
            discovered,
            collected,
            f"collector drift: unittest discovered {discovered}, pytest "
            f"collected {collected}. Tests visible to only one collector "
            "never run on the other path.",
        )


if __name__ == "__main__":
    unittest.main()
