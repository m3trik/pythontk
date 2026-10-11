# !/usr/bin/python
# coding=utf-8
"""Tests for :class:`pythontk.TestSandbox` -- the process-level test isolation.

The runner activates the sandbox before discovery, so these run INSIDE it and
assert what it does rather than construct a second one; a standalone run of
this module activates it in ``setUp``.
"""

import json
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


class ActivatedTest(unittest.TestCase):
    """:meth:`TestSandbox.activated` -- every guard for a block, then off again.

    For a host that outlives the run: mayatk's in-session harness runs a suite
    inside the user's own Maya, and :meth:`TestSandbox.activate` left that
    session refusing browser launches, keeping nothing from the trash and
    writing temp files into a root deleted when Maya exits.

    Each case runs in a fresh interpreter: this process is already sandboxed,
    and a block entered under standing guards leaves them standing.
    """

    #: Run in the child first: what every guard replaces, captured by identity.
    PRELUDE = "\n".join(
        [
            "import json, os, sys, tempfile, webbrowser",
            "sys.path.insert(0, {repo!r})",
            "from pythontk.core_utils.test_sandbox import TestSandbox",
            "from pythontk.file_utils._file_utils import FileUtils",
            "def snapshot():",
            "    return [",
            "        [id(getattr(webbrowser, n)) for n in TestSandbox._LAUNCHERS],",
            "        [id(getattr(webbrowser, n).__dict__.get('open'))",
            "         for n in TestSandbox._LAUNCHER_TYPES],",
            "        [id(FileUtils.__dict__[n]) for n in ('move_to_trash', 'can_trash')],",
            "        tempfile.tempdir,",
            "        {{n: os.environ.get(n) for n in ('TMPDIR', 'TEMP', 'TMP',",
            "                                       TestSandbox._ROOT_ENV)}},",
            "    ]",
            "def refused():",
            "    try:",
            "        webbrowser.open('http://127.0.0.1:1/')",
            "    except RuntimeError:",
            "        return True",
            "    return False",
            "before = snapshot()",
            "report = {{}}",
        ]
    )

    def _child(self, body):
        """Run *body* after :attr:`PRELUDE` in a fresh interpreter; return the
        ``report`` dict it prints as its last line."""
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        code = self.PRELUDE.format(repo=repo) + "\n" + body
        out = subprocess.check_output([sys.executable, "-c", code], text=True)
        return json.loads(out.splitlines()[-1])

    def test_every_guard_comes_off_on_exit(self):
        report = self._child(
            "\n".join(
                [
                    "with TestSandbox.activated() as root:",
                    "    report['active'] = TestSandbox.is_active()",
                    "    report['refused'] = refused()",
                    "    report['in_root'] = tempfile.gettempdir() == root",
                    "    report['trash_guarded'] = not FileUtils.can_trash(root)",
                    "report['restored'] = snapshot() == before",
                    "report['inactive'] = not TestSandbox.is_active()",
                    "report['root_gone'] = not os.path.exists(root)",
                    "print(json.dumps(report))",
                ]
            )
        )
        self.assertEqual(
            report,
            dict.fromkeys(
                ("active", "refused", "in_root", "trash_guarded", "restored")
                + ("inactive", "root_gone"),
                True,
            ),
        )

    def test_every_guard_comes_off_when_the_block_raises(self):
        report = self._child(
            "\n".join(
                [
                    "try:",
                    "    with TestSandbox.activated() as root:",
                    "        raise ValueError('a failing run')",
                    "except ValueError:",
                    "    report['raised'] = True",
                    "report['restored'] = snapshot() == before",
                    "report['inactive'] = not TestSandbox.is_active()",
                    "report['root_gone'] = not os.path.exists(root)",
                    "print(json.dumps(report))",
                ]
            )
        )
        self.assertEqual(
            report,
            dict.fromkeys(("raised", "restored", "inactive", "root_gone"), True),
        )

    def test_a_reload_inside_the_block_is_rearmed_and_still_restored(self):
        """mayatk's in-session harness reloads pythontk inside the block: the
        reloaded ``FileUtils`` trashed for real, and the reloaded sandbox read
        as off. ``rearm()`` carries the standing guards over; the way out
        still puts back what each class had."""
        report = self._child(
            "\n".join(
                [
                    "import importlib",
                    "import pythontk.core_utils.test_sandbox as ts_mod",
                    "import pythontk.file_utils._file_utils as fu_mod",
                    "guarded = lambda F: F.__dict__['move_to_trash'].__func__.__name__",
                    "with TestSandbox.activated():",
                    "    importlib.reload(fu_mod)",
                    "    importlib.reload(ts_mod)",
                    "    F2 = fu_mod.FileUtils",
                    "    reals = (F2.__dict__['move_to_trash'], F2.__dict__['can_trash'])",
                    "    report['lost_on_reload'] = guarded(F2) == 'move_to_trash'",
                    "    TestSandbox.rearm()",
                    "    report['rearmed'] = guarded(F2) == 'no_trash'",
                    "    report['reloaded_reads_active'] = ts_mod.TestSandbox.is_active()",
                    "report['restored'] = snapshot() == before",
                    "report['reloaded_class_restored'] = (",
                    "    F2.__dict__['move_to_trash'], F2.__dict__['can_trash']) == reals",
                    "print(json.dumps(report))",
                ]
            )
        )
        self.assertEqual(
            report,
            dict.fromkeys(
                ("lost_on_reload", "rearmed", "reloaded_reads_active", "restored")
                + ("reloaded_class_restored",),
                True,
            ),
        )

    def test_guards_standing_on_entry_stay_standing(self):
        """A block inside an :meth:`activate`'d process (a conftest import in a
        runner's child) neither re-redirects nor takes anything down."""
        report = self._child(
            "\n".join(
                [
                    "root = TestSandbox.activate()",
                    "standing = snapshot()",
                    "with TestSandbox.activated() as inner:",
                    "    report['same_root'] = inner == root",
                    "report['kept'] = snapshot() == standing",
                    "report['active'] = TestSandbox.is_active()",
                    "report['root_kept'] = os.path.isdir(root)",
                    "print(json.dumps(report))",
                ]
            )
        )
        self.assertEqual(
            report,
            dict.fromkeys(("same_root", "kept", "active", "root_kept"), True),
        )


class OrphanRootTest(unittest.TestCase):
    """A root its owner could not remove goes at the next activation.

    A root's owner removes it at exit as far as it will go, and a runner that
    exits while the GUI Maya it launched still holds files in the root (its
    licence log, its scratch) left the whole root -- for the seven-day age
    sweep: 67 roots in four days (measured 2026-10-05), every one free to
    delete by the next run. The root is a session store, which names its
    owner (``TempArtifacts``), so the next activation's sweep reclaims it.

    Hermetic: both children nest inside this test's own root.
    """

    def setUp(self):
        TestSandbox.activate()

    def _child(self, tail):
        """A child that activates the sandbox, prints its root, then *tail*."""
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        code = "\n".join(
            [
                "import os, sys",
                f"sys.path.insert(0, {repo!r})",
                "from pythontk.core_utils.test_sandbox import TestSandbox",
                "root = TestSandbox.temp()",
                "print(root, flush=True)",
                tail,
            ]
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        return proc.stdout.strip()

    def test_the_root_names_its_owner(self):
        root = TestSandbox.temp()
        self.assertTrue(os.path.basename(root).endswith(f"-{os.getpid()}"), root)

    def test_a_killed_owners_root_goes_at_the_next_activation(self):
        orphan = self._child("os.kill(os.getpid(), 9)  # no exit cleanup runs")
        self.assertTrue(orphan and os.path.isdir(orphan), "the killed root remains")

        self._child("")

        self.assertFalse(os.path.exists(orphan))


class DetachedLaunchTest(unittest.TestCase):
    """A detached app is launched with the host's environment, not the sandbox's.

    ``AppLauncher.launch(detached=True)`` is the launch that keeps running after
    the launcher ends, and the root goes when its owner does -- at exit, or at
    the next activation's sweep of a root whose owner has exited. A GUI Maya
    the mayatk runner launched kept the root as its TEMP after the run, the
    user went on working in it, and another suite's activation swept the root:
    the WebXR preview's FBX was deleted between its export and its conversion
    (measured 2026-10-05, "FBX source not found").

    A waited child is unchanged -- it ends inside the run, so it shares the
    root (mayapy chunks, FBX2glTF) -- and so is a value the caller chose.
    """

    REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    #: Set at the C level in the child, as a DCC host sets its own variables:
    #: invisible to ``os.environ``, inherited by every child all the same.
    NATIVE = "PYTHONTK_NATIVE_PROBE"
    #: The grandchild: writes the variables it was handed to ``argv[1]``.
    PROBE = (
        "import json, os, sys\n"
        f"names = ('TMPDIR', 'TEMP', 'TMP', 'PYTHONTK_TEST_TEMP_ROOT', {NATIVE!r})\n"
        "with open(sys.argv[1], 'w') as fh:\n"
        "    json.dump({n: os.environ.get(n) for n in names}, fh)\n"
    )

    def setUp(self):
        TestSandbox.activate()
        self.scratch = TempArtifacts("detached_probe", policy="scoped")
        self.addCleanup(self.scratch.cleanup, force=True)

    def test_a_detached_app_gets_the_host_env_and_a_waited_one_the_root(self):
        outs = {k: self.scratch.path(extension=".json") for k in ("d", "w", "c")}
        chosen = self.scratch.dir_path()
        code = "\n".join(
            [
                "import ctypes, json, os, sys",
                f"sys.path.insert(0, {self.REPO!r})",
                "from pythontk.core_utils.test_sandbox import TestSandbox",
                "from pythontk.core_utils.app_launcher import AppLauncher",
                "names = ('TMPDIR', 'TEMP', 'TMP', TestSandbox._ROOT_ENV)",
                "host = {n: os.environ.get(n) for n in names}",
                "if os.name == 'nt':",
                "    ctypes.windll.kernel32.SetEnvironmentVariableW(",
                f"        {self.NATIVE!r}, 'host-set')",
                "else:",
                f"    ctypes.CDLL(None).setenv({self.NATIVE.encode()!r}, b'host-set', 1)",
                "root = TestSandbox.activate()",
                f"probe = {self.PROBE!r}",
                "def run(out, **kwargs):",
                "    AppLauncher.launch(sys.executable, ['-c', probe, out], **kwargs).wait()",
                f"run({outs['d']!r}, detached=True)",
                f"run({outs['w']!r}, detached=False)",
                f"run({outs['c']!r}, detached=True, env=dict(os.environ, TEMP={chosen!r}))",
                "print(json.dumps({'host': host, 'root': root}))",
            ]
        )
        out = subprocess.check_output([sys.executable, "-c", code], text=True)
        seen = json.loads(out.splitlines()[-1])
        host, root = seen["host"], seen["root"]
        got = {}
        for key, path in outs.items():
            with open(path, encoding="utf-8") as fh:
                got[key] = json.load(fh)

        native = got["d"].pop(self.NATIVE)
        self.assertEqual(got["d"], host, "a detached app keeps the host's env")
        self.assertEqual(native, "host-set", "...read from the live block")
        for name in ("TMPDIR", "TEMP", "TMP", TestSandbox._ROOT_ENV):
            self.assertEqual(got["w"][name], root, f"a waited child shares {name}")
        self.assertEqual(got["c"]["TEMP"], chosen, "the caller's own value stands")
        self.assertEqual(got["c"]["TMP"], host["TMP"])

    def test_nothing_redirected_means_nothing_to_undo(self):
        code = "\n".join(
            [
                "import sys",
                f"sys.path.insert(0, {self.REPO!r})",
                "from pythontk.core_utils.test_sandbox import TestSandbox",
                "before = TestSandbox.host_environ()",
                "with TestSandbox.activated():",
                "    during = TestSandbox.host_environ() is not None",
                "print(before is None, during, TestSandbox.host_environ() is None)",
            ]
        )
        out = subprocess.check_output([sys.executable, "-c", code], text=True)
        self.assertEqual(out.split(), ["True", "True", "True"])

    def test_a_shallow_patch_of_the_state_gets_the_record_back(self):
        """mayatk's runner tests wrap a fresh ``temp()`` in
        ``mock.patch.dict(TestSandbox._state, ...)``, a shallow copy: a record
        edited in place would outlive the patch, and every later detached
        launch in that process would get the test's long-gone values."""
        before = TestSandbox._state["host_env"]
        with (
            unittest.mock.patch.dict(os.environ),
            unittest.mock.patch.dict(TestSandbox._state, {}),
        ):
            TestSandbox._redirect_env("PYTHONTK_SANDBOX_PROBE", "redirected")
            self.assertIn("PYTHONTK_SANDBOX_PROBE", TestSandbox._state["host_env"])
        self.assertEqual(TestSandbox._state["host_env"], before)
        self.assertNotIn("PYTHONTK_SANDBOX_PROBE", os.environ)


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
