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

    def test_a_browser_built_directly_is_refused_too(self):
        """Regression: patching the module functions left a second route open.

        Those functions only ever open the system DEFAULT browser, so anything
        that needs a specific one -- as the preview does, since the default is
        frequently a build with no WebXR backend -- constructs a browser class
        itself and calls `open` on it. That path reached a real launch.
        """
        for name in ("BackgroundBrowser", "GenericBrowser"):
            with self.subTest(browser=name):
                # A path that cannot launch: should this guard ever regress,
                # the test fails without also spawning a process on the way.
                browser = getattr(webbrowser, name)(r"C:\nonexistent\nothing.exe")
                with self.assertRaises(RuntimeError) as caught:
                    browser.open(URL)
                self.assertIn("open_browser=False", str(caught.exception))
        # Recorded under the URL, not the browser object standing in front of it.
        self.assertEqual(TestSandbox.launches[self._launches_before :], [URL] * 2)

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

    def test_a_child_sandbox_nests_inside_the_root_even_when_the_probe_fails(self):
        """A child's own root goes INSIDE the parent's, named, never re-derived.

        A child that activates the sandbox (mayatk's chunk driver, tentacle's
        in-Maya dispatcher) found the parent's root the way any fresh
        interpreter does: ``tempfile`` probes TMPDIR/TEMP/TMP with a throwaway
        write and, on any failure but ``FileExistsError``, falls past all three
        to the user's real temp dir without a word. Measured: a root that had
        gone, an ``OSError`` on three probes in a row, or a refused create.
        Here every probe write into the root is refused.
        """
        root = TestSandbox.temp()
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        child = "\n".join(
            [
                "import errno, os, sys",
                f"sys.path.insert(0, {repo!r})",
                "root = os.path.normcase(os.environ['TEMP'])",
                "real_open = os.open",
                "def refuse(path, *args, **kwargs):",
                "    if os.path.normcase(os.path.dirname(os.path.abspath(path))) == root:",
                "        raise OSError(errno.EIO, 'probe write refused', path)",
                "    return real_open(path, *args, **kwargs)",
                "os.open = refuse",
                "from pythontk.core_utils.test_sandbox import TestSandbox",
                "nested = TestSandbox.temp()",
                "os.open = real_open  # the exit cleanup's rmtree opens dirs on POSIX",
                "print(nested)",
            ]
        )
        nested = subprocess.check_output(
            [sys.executable, "-c", child], text=True
        ).splitlines()[-1]
        self.assertEqual(
            os.path.normcase(os.path.dirname(nested)), os.path.normcase(root)
        )
        self.assertFalse(os.path.exists(nested), "the child's root goes with it")
        self.assertEqual(os.environ.get(TestSandbox._ROOT_ENV), root)

    def test_the_trash_is_the_scratch_folders_not_the_machines(self):
        """A re-bake sets superseded maps aside through ``move_to_trash``: under
        the sandbox that is "no trash here", so they go beside themselves into
        ``_superseded`` inside the temp root -- never into the developer's
        Recycle Bin, which a mayatk run filled with 11 scratch maps."""
        from pythontk.file_utils._file_utils import FileUtils
        from pythontk.file_utils.file_dependencies import FileDependencies

        store = TempArtifacts("sandbox_trash", policy="scoped")
        self.addCleanup(store.cleanup)
        path = os.path.join(store.dir_path(), "old_Lightmap.exr")
        with open(path, "wb") as fh:
            fh.write(b"map")
        before = len(TestSandbox.trashed)
        where = FileDependencies.set_aside(path)
        self.assertEqual(TestSandbox.trashed[before:], [os.path.abspath(path)])
        del TestSandbox.trashed[before:]
        self.assertEqual(
            os.path.normcase(os.path.abspath(where)),
            os.path.normcase(
                os.path.join(os.path.dirname(path), "_superseded", "old_Lightmap.exr")
            ),
        )
        with self.assertRaises(FileNotFoundError):
            FileUtils.move_to_trash(path)  # the real refusals still stand
        self.assertFalse(
            FileUtils.can_trash(where), "a prompt is told there is no trash, too"
        )
        real = TestSandbox._state["trash"]
        with TestSandbox.real_trash():
            self.assertIs(FileUtils.__dict__["move_to_trash"], real)
            self.assertIs(
                FileUtils.__dict__["can_trash"], TestSandbox._state["can_trash"]
            )
        self.assertIsNot(FileUtils.__dict__["move_to_trash"], real, "guarded again")

    def test_activate_is_idempotent(self):
        self.assertEqual(TestSandbox.activate(), TestSandbox.activate())
        self.assertTrue(TestSandbox.is_active())

    def test_user_config_hides_the_developers_convention_then_restores_it(self):
        """A developer's Lightmap = "_LM" must not reach a test, and a test's
        write must not reach the developer's doc."""
        import json

        from pythontk.core_utils.naming_convention import (
            CONFIG_ENV_VAR,
            NamingConvention,
        )
        from pythontk.core_utils.user_config import CONFIG_ROOT_ENV_VAR

        store = TempArtifacts("sandbox_dev_config", policy="scoped")
        dev_root = store.dir_path()
        self.addCleanup(store.cleanup)
        studio = os.path.join(dev_root, "studio.json")
        with open(studio, "w") as f:
            json.dump({"mesh": "_MSH"}, f)
        saved = {v: os.environ.get(v) for v in (CONFIG_ROOT_ENV_VAR, CONFIG_ENV_VAR)}

        def restore():
            for var, value in saved.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
            NamingConvention.reload()

        self.addCleanup(restore)
        os.environ[CONFIG_ROOT_ENV_VAR] = dev_root
        os.environ[CONFIG_ENV_VAR] = studio
        NamingConvention.set("lightmap", "_LM")  # the developer's own edit

        with TestSandbox.user_config() as root:
            self.assertNotEqual(os.path.normcase(root), os.path.normcase(dev_root))
            self.assertEqual(NamingConvention.affix("lightmap"), "_Lightmap")
            self.assertEqual(NamingConvention.affix("mesh"), "_GEO")  # no studio doc
            NamingConvention.set("group", "_G")  # a test's write
        self.assertFalse(os.path.exists(root), "the throwaway root is removed")
        self.assertEqual(os.environ[CONFIG_ENV_VAR], studio)
        self.assertEqual(NamingConvention.affix("lightmap"), "_LM")
        self.assertEqual(NamingConvention.affix("mesh"), "_MSH")
        self.assertEqual(NamingConvention.affix("group"), "_GRP")


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
        # Named outside test_*.py: pytest still collects a path it is handed,
        # while neither collector's own discovery ever sees it -- a run killed
        # before cleanup would otherwise leave a test that fails every later run
        # (it leaks a browser launch by design) until the age sweep takes it.
        scratch = TempArtifacts(
            "_leakgate", policy="scoped", dir=os.path.dirname(__file__)
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
