# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.operation_catalog.session_operation_host:
SessionOperationHost, one instance kept for a session (a fresh Python
process stands in for the application), the catalog that keeps it, and the
MCP server's view of kept instances."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pythontk import (
    BatchOperationHost,
    CancelScope,
    Operation,
    OperationCatalog,
    OperationProvider,
    OperationServer,
    Outcome,
    PureSceneFiles,
    SessionOperationHost,
    TempArtifacts,
)

_PACKAGE = "ptk_session_fixture"
_ENTRY_POINTS = "pythontk.core_utils.operation_catalog._operation_catalog.PackageManager.entry_points"
_SCENE = '''\
import json
import os
import time

SCENE = {"path": None, "objects": [], "dirty": False}


class JsonSceneFiles:
    """A JSON file of object names plays the scene."""

    def __init__(self):
        if os.environ.get("PTK_SESSION_SLOW_START"):
            time.sleep(30)
        if os.environ.get("PTK_SESSION_FAIL_START"):
            print("checking the licence", flush=True)
            raise RuntimeError("no licence")

    def open(self, paths):
        for path in paths:
            with open(path, encoding="utf-8") as f:
                SCENE.update(path=path, objects=json.load(f)["objects"], dirty=False)

    def modified(self):
        return [SCENE["path"]] if SCENE["path"] and SCENE["dirty"] else []

    def unsaved(self):
        return SCENE["dirty"]

    def save(self, paths):
        for path in paths:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"objects": SCENE["objects"]}, f)
        SCENE["dirty"] = False

    def close(self):
        pass


class ScriptedSceneFiles(JsonSceneFiles):
    """As in Blender: its flag misses an edit made from code, so the run tells it."""

    def changed(self):
        SCENE["dirty"] = True


def whoami() -> int:
    """This process's id."""
    return os.getpid()


def count() -> int:
    """How many objects the scene holds."""
    return len(SCENE["objects"])


def rename(old: str, new: str, ctx=None) -> str:
    """Rename one object."""
    index = SCENE["objects"].index(old)
    if ctx.should_change(old, "rename to " + new):
        SCENE["objects"][index] = new
        SCENE["dirty"] = True
    return new


def wait_for_cancel(ctx=None) -> None:
    """Report half way, then wait at checkpoints until cancelled."""
    ctx.progress(0.5, "half")
    for _ in range(600):
        ctx.check()
        time.sleep(0.05)


def die() -> None:
    """End the process before any outcome is written."""
    os._exit(3)


def die_slowly() -> None:
    """End the process a second into the call."""
    time.sleep(1)
    os._exit(3)


def hang() -> None:
    """Hold the interpreter in a native call, as a host's command can."""
    import ctypes

    ctypes.PyDLL("kernel32").Sleep(60000)
'''


def _entries():
    def op(name, **kwargs):
        return Operation(
            f"py.{name}", f"{_PACKAGE}.scene:{name}", host="py", **kwargs
        ).describe()

    console = Operation(
        "py.python.exec", "pythontk:PythonConsole.run", effect="code", host="py"
    )
    return [
        op("whoami"),
        op("count"),
        op("rename", effect="state"),
        op("wait_for_cancel", effect="state"),
        op("die"),
        op("die_slowly"),
        op("hang"),
        console.describe(),
    ]


class _SessionFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / _PACKAGE).mkdir()
        (root / _PACKAGE / "__init__.py").write_text("", encoding="utf-8")
        self.module = root / _PACKAGE / "scene.py"
        self.module.write_text(_SCENE, encoding="utf-8")
        sys.path.insert(0, self._tmp.name)
        self.scene = root / "shot.json"
        self.scene.write_text(json.dumps({"objects": ["a", "b"]}), encoding="utf-8")
        self.session = f"test-{uuid.uuid4().hex[:10]}"
        self.hosts = []

    def tearDown(self):
        for host in self.hosts:
            host.close()
        sys.path.remove(self._tmp.name)
        for name in [m for m in sys.modules if m.split(".")[0] == _PACKAGE]:
            del sys.modules[name]
        self._tmp.cleanup()

    def batch(self, **kwargs):
        options = dict(
            scene_files=f"{_PACKAGE}.scene:JsonSceneFiles",
            packages=(_PACKAGE,),
            timeout=60,
            grace=30,
        )
        options.update(kwargs)
        return BatchOperationHost(sys.executable, **options)

    def host(self, keep=None, reload=(), **kwargs):
        host = SessionOperationHost(
            self.batch(**kwargs), self.session, "py", keep=keep, reload=reload
        )
        self.hosts.append(host)
        return host

    def catalog(self, host=None):
        return OperationCatalog(_entries(), hosts={"py": host or self.host()})

    def objects(self):
        return json.loads(self.scene.read_text(encoding="utf-8"))["objects"]


class TestSessionOperationHost(_SessionFixture):
    def test_calls_share_one_instance_that_keeps_what_they_leave(self):
        catalog = self.catalog()
        first, second = catalog.run("py.whoami"), catalog.run("py.whoami")
        self.assertEqual((first.status, second.status), ("done", "done"))
        self.assertEqual(first.result, second.result)
        self.assertNotEqual(first.result, os.getpid())
        catalog.run("py.python.exec", {"code": "kept = 7"}, apply=True)
        self.assertEqual(
            catalog.run("py.python.exec", {"code": "kept * 6"}, apply=True).result[
                "value"
            ],
            42,
        )

    def test_runners_given_one_session_share_its_instance_and_only_its_starter_ends_it(
        self,
    ):
        starter, other = self.host(), self.host()
        pid = self.catalog(starter).run("py.whoami").result
        self.assertEqual(self.catalog(other).run("py.whoami").result, pid)
        other.close()  # not its instance: it stays
        self.assertEqual(self.catalog(other).run("py.whoami").result, pid)
        starter.close()
        self.assertEqual(starter.queue.worker()["state"], "stopped")

    def test_scenes_open_first_a_live_plan_saves_them_and_a_call_without_works_on_what_it_holds(
        self,
    ):
        catalog = self.catalog()
        plan = catalog.run("py.rename", {"old": "a", "new": "c"}, scenes=[self.scene])
        self.assertTrue(plan.dry_run)
        self.assertEqual(self.objects(), ["a", "b"])
        done = catalog.run(
            "py.rename", {"old": "a", "new": "c"}, apply=True, scenes=[self.scene]
        )
        self.assertEqual(done.status, "done", done.error)
        self.assertEqual(self.objects(), ["c", "b"])
        self.assertEqual(catalog.run("py.count").result, 2)

    def test_a_change_to_code_it_loaded_starts_a_fresh_instance_and_the_outcome_says_so(
        self,
    ):
        catalog = self.catalog()
        pid = catalog.run("py.whoami").result
        catalog.run("py.python.exec", {"code": "kept = 1"}, apply=True)
        time.sleep(0.05)
        self.module.write_text(_SCENE + "\n", encoding="utf-8")
        again = catalog.run("py.whoami")
        self.assertNotEqual(again.result, pid)
        self.assertEqual([i.code for i in again.issues], ["session.restarted"])
        self.assertIn("scene.py", again.issues[0].message)
        gone = catalog.run(
            "py.python.exec", {"code": "'kept' in globals()"}, apply=True
        )
        self.assertEqual(gone.result["value"], False)

    def test_progress_comes_back_and_a_cancel_stops_the_call(self):
        scope, seen = CancelScope("test"), []

        def on_progress(fraction, message):
            seen.append((fraction, message))
            if message == "half":
                scope.cancel("enough")

        outcome = self.catalog().run(
            "py.wait_for_cancel", apply=True, scope=scope, on_progress=on_progress
        )
        self.assertEqual(outcome.status, "cancelled")
        self.assertIn((0.5, "half"), seen)

    def test_a_call_cancelled_while_it_waits_its_turn_is_withdrawn_and_stops_nothing(
        self,
    ):
        catalog = self.catalog()
        pid = catalog.run("py.whoami").result
        first, started, results = CancelScope("first"), threading.Event(), {}

        def run_first():
            results["first"] = catalog.run(
                "py.wait_for_cancel",
                apply=True,
                scope=first,
                on_progress=lambda f, m: started.set(),
            )

        worker = threading.Thread(target=run_first)
        worker.start()
        self.assertTrue(started.wait(30))
        second = CancelScope("second")
        second.cancel("never mind")
        waiting = catalog.run("py.whoami", scope=second)
        self.assertEqual(
            (waiting.status, waiting.error), ("cancelled", "cancelled before it ran")
        )
        first.cancel("done")
        worker.join(30)
        self.assertEqual(results["first"].status, "cancelled")
        self.assertEqual(
            catalog.run("py.whoami").result, pid
        )  # the instance was never stopped

    def test_an_instance_that_dies_is_lost_with_its_call_and_the_next_call_starts_another(
        self,
    ):
        catalog = self.catalog()
        pid = catalog.run("py.whoami").result
        lost = catalog.run("py.die")
        self.assertEqual(lost.status, "failed")
        self.assertIn("instance ended", lost.error)
        self.assertNotEqual(catalog.run("py.whoami").result, pid)

    def test_an_instance_that_cannot_start_refuses_the_call_with_why(self):
        host = self.host(env=dict(os.environ, PTK_SESSION_FAIL_START="1"))
        outcome = self.catalog(host).run("py.whoami")
        self.assertEqual(outcome.status, "failed")
        self.assertIn("cannot start this session's py instance", outcome.error)
        self.assertIn("no licence", outcome.error)

    def test_a_runner_that_serves_its_run_folders_itself_cannot_be_kept(self):
        with self.assertRaises(ValueError):
            SessionOperationHost(
                BatchOperationHost(sys.executable, command=lambda run: []), "s", "py"
            )


class TestSessionRecovery(_SessionFixture):
    """What a kept instance does when it fails, dies, hangs or holds unsaved work."""

    def test_a_failed_start_leaves_nothing_that_fails_the_next(self):
        failing = self.host(env=dict(os.environ, PTK_SESSION_FAIL_START="1"))
        refused = self.catalog(failing).run("py.whoami")
        self.assertIn("no licence", refused.error)
        self.assertIn(
            "checking the licence",
            (failing.folder / "instance.log").read_text(encoding="utf-8"),
        )
        self.assertEqual(self.catalog().run("py.whoami").status, "done")

    def test_a_call_waiting_when_its_instance_dies_runs_in_the_next_and_says_so(self):
        catalog = self.catalog()
        pid = catalog.run("py.whoami").result
        results = {}
        first = threading.Thread(
            target=lambda: results.update(first=catalog.run("py.die_slowly"))
        )
        first.start()
        time.sleep(0.3)
        second = catalog.run(
            "py.whoami"
        )  # queued behind the call that kills its instance
        first.join(30)
        self.assertEqual(results["first"].status, "failed")
        self.assertIn("instance ended", results["first"].error)
        self.assertEqual(second.status, "done", second.error)
        self.assertNotEqual(second.result, pid)
        self.assertIn("session.restarted", [i.code for i in second.issues])

    def test_the_log_drain_is_published_only_once_it_runs(self):
        """A call failing while the next call starts the replacement instance
        reads the log tail, joining ``_drain``: a drain published before it
        started raised "cannot join thread before it is started" in that call
        (the test above, Windows CI, 2026-11-11). Pinned without the race."""
        host = self.host()
        drains = []
        start = threading.Thread.start

        def recorded(thread):
            if thread.name.endswith("-session-log"):
                drains.append(host._drain is thread)  # visible before it runs?
            start(thread)

        with patch.object(threading.Thread, "start", recorded):
            self.assertEqual(self.catalog(host).run("py.whoami").status, "done")
        self.assertEqual(drains, [False])

    def test_an_instance_gone_between_calls_is_replaced_and_the_next_outcome_says_so(
        self,
    ):
        from pythontk import AppLauncher

        catalog = self.catalog()
        pid = catalog.run("py.whoami").result
        AppLauncher.close_process(pid, force=True)
        again = catalog.run("py.whoami")
        self.assertNotEqual(again.result, pid)
        self.assertEqual([i.code for i in again.issues], ["session.restarted"])

    @unittest.skipUnless(
        os.name == "nt", "the hang holds the interpreter through a Windows call"
    )
    def test_another_runner_stops_an_instance_hung_in_a_native_call(self):
        from pythontk.core_utils.execution_monitor._sidecar import process_alive

        starter, other = self.host(), self.host(timeout=3)
        pid = self.catalog(starter).run("py.whoami").result
        hung = self.catalog(other).run("py.hang")
        self.assertEqual(hung.status, "failed")
        self.assertIn("timed out after 3 s, so its instance was stopped", hung.error)
        self.assertFalse(process_alive(pid))
        self.assertNotEqual(self.catalog(other).run("py.whoami").result, pid)

    def test_a_module_written_before_its_first_import_is_current(self):
        late = Path(self._tmp.name) / _PACKAGE / "late.py"
        late.write_text(
            'def value() -> int:\n    """One."""\n    return 1\n', encoding="utf-8"
        )
        catalog = OperationCatalog(
            _entries()
            + [Operation("py.late", f"{_PACKAGE}.late:value", host="py").describe()],
            hosts={"py": self.host()},
        )
        pid = catalog.run("py.whoami").result  # started before late.py changes
        time.sleep(0.05)
        # Another size too: a .pyc this process wrote is trusted within its second.
        late.write_text(
            'def value() -> int:\n    """Twenty-two."""\n    return 22\n',
            encoding="utf-8",
        )
        self.assertEqual(catalog.run("py.late").result, 22)
        again = catalog.run("py.whoami")
        self.assertEqual((again.result, again.issues), (pid, []))

    def test_a_cancel_interrupts_running_code_and_keeps_the_instance_and_its_names(self):
        # As Ctrl+C in a console: the loop stops, nothing the console holds is lost.
        catalog = self.catalog()
        pid = catalog.run("py.whoami").result
        catalog.run("py.python.exec", {"code": "kept = 5"}, apply=True)
        scope = CancelScope("test")
        threading.Timer(0.5, lambda: scope.cancel("enough")).start()
        started = time.monotonic()
        stopped = catalog.run(
            "py.python.exec", {"code": "import time\nwhile True:\n    time.sleep(0.01)"},
            apply=True, scope=scope,
        )
        self.assertLess(time.monotonic() - started, 15)  # well inside the grace (30 s)
        self.assertEqual(stopped.status, "partial")  # code keeps no journal
        self.assertIn("interrupted", stopped.error)
        again = catalog.run("py.python.exec", {"code": "kept"}, apply=True)
        self.assertEqual((again.result["value"], again.issues), (5, []))
        self.assertEqual(catalog.run("py.whoami").result, pid)

    def test_the_codes_own_keyboard_interrupt_keeps_the_instance_and_its_names(self):
        # Only the cancel's interrupt was caught: one the code raised ended the
        # instance, and every name the console held with it.
        catalog = self.catalog()
        pid = catalog.run("py.whoami").result
        catalog.run("py.python.exec", {"code": "kept = 5"}, apply=True)
        raised = catalog.run("py.python.exec", {"code": "raise KeyboardInterrupt"}, apply=True)
        self.assertEqual(raised.status, "partial")
        self.assertIn("KeyboardInterrupt", raised.error)
        again = catalog.run("py.python.exec", {"code": "kept"}, apply=True)
        self.assertEqual((again.result["value"], again.issues), (5, []))
        self.assertEqual(catalog.run("py.whoami").result, pid)

    def test_code_saves_only_what_it_saves_itself_and_what_it_leaves_is_held(self):
        # Code has no plan, so no save rule: a read-only snippet naming a scene
        # never rewrote the file (a Blender .blend upgraded on every look).
        catalog = self.catalog(self.host(scene_files=f"{_PACKAGE}.scene:ScriptedSceneFiles"))
        edit = f"import {_PACKAGE}.scene as s\ns.SCENE['objects'].append('x')"
        done = catalog.run("py.python.exec", {"code": edit}, apply=True, scenes=[self.scene])
        self.assertEqual(done.status, "done", done.error)
        self.assertEqual(done.changes, [])
        self.assertEqual(self.objects(), ["a", "b"])  # the file as it was
        refused = catalog.run("py.count", scenes=[self.scene])
        self.assertIn("unsaved changes", refused.error)  # what it changed is held

    def test_code_imported_from_outside_its_roots_is_watched_too(self):
        # A package the console imported that is none of the runner's roots
        # (uitk, in a session whose roots are pythontk's) goes stale the same way.
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        package = Path(other.name) / "ptk_session_other"
        package.mkdir()
        (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        sys.path.insert(0, other.name)
        self.addCleanup(sys.path.remove, other.name)
        catalog = self.catalog()
        pid = catalog.run("py.whoami").result
        read = "import ptk_session_other\nptk_session_other.VALUE"
        self.assertEqual(
            catalog.run("py.python.exec", {"code": read}, apply=True).result["value"], 1
        )
        time.sleep(0.05)
        (package / "__init__.py").write_text("VALUE = 22\n", encoding="utf-8")
        again = catalog.run("py.whoami")
        self.assertNotEqual(again.result, pid)
        self.assertEqual([i.code for i in again.issues], ["session.restarted"])
        self.assertIn("ptk_session_other", again.issues[0].message)

    def test_opening_scenes_over_unsaved_changes_is_refused(self):
        catalog = self.catalog()
        catalog.run(
            "py.rename", {"old": "a", "new": "c"}, apply=True, scenes=[self.scene]
        )
        dirty = f"import {_PACKAGE}.scene as s\ns.SCENE['objects'].append('x')\ns.SCENE['dirty'] = True"
        catalog.run("py.python.exec", {"code": dirty}, apply=True)
        refused = catalog.run("py.count", scenes=[self.scene])
        self.assertEqual(refused.status, "failed")
        self.assertIn("unsaved changes", refused.error)
        self.assertEqual(
            catalog.run("py.count").result, 3
        )  # what it holds is untouched
        saved = f"import {_PACKAGE}.scene as s\ns.SCENE['dirty'] = False"
        catalog.run("py.python.exec", {"code": saved}, apply=True)
        self.assertEqual(catalog.run("py.count", scenes=[self.scene]).result, 2)

    def test_code_that_saves_its_change_leaves_nothing_held(self):
        # Blender's flag misses a data-API edit, so a code run marks the scene:
        # before it runs, where the code's own save clears the mark as the
        # application's would (marked after, a saved scene read unsaved forever).
        host = self.host(scene_files=f"{_PACKAGE}.scene:ScriptedSceneFiles")
        catalog = self.catalog(host)
        catalog.run("py.count", scenes=[self.scene])
        scene = f"import {_PACKAGE}.scene as s\n"
        save = "s.JsonSceneFiles().save([s.SCENE['path']])"
        edit = scene + "s.SCENE['objects'].append('x')\n" + save
        catalog.run("py.python.exec", {"code": edit}, apply=True)
        self.assertEqual(self.objects(), ["a", "b", "x"])
        self.assertEqual(catalog.run("py.count", scenes=[self.scene]).result, 3)
        catalog.run("py.python.exec", {"code": scene + "s.SCENE['objects'].append('y')"}, apply=True)
        refused = catalog.run("py.count", scenes=[self.scene])  # an unsaved edit is still held
        self.assertIn("unsaved changes", refused.error)

    def test_code_that_does_not_compile_fails_and_holds_nothing(self):
        # None of it ran: "partial" (what it changed is unknown) held the scene
        # as unsaved, so a typo blocked the next call that opens scenes.
        catalog = self.catalog(self.host(scene_files=f"{_PACKAGE}.scene:ScriptedSceneFiles"))
        catalog.run("py.count", scenes=[self.scene])
        typo = catalog.run("py.python.exec", {"code": "def (:"}, apply=True)
        self.assertEqual(typo.status, "failed", typo.error)
        self.assertIn("SyntaxError", typo.error)
        self.assertEqual(catalog.run("py.count", scenes=[self.scene]).result, 2)

    def test_a_cancel_while_the_instance_starts_cancels_the_call(self):
        host = self.host(env=dict(os.environ, PTK_SESSION_SLOW_START="1"))
        scope = CancelScope("test")
        threading.Timer(0.5, lambda: scope.cancel("enough")).start()
        started = time.monotonic()
        outcome = self.catalog(host).run("py.whoami", scope=scope)
        self.assertEqual(outcome.status, "cancelled")
        self.assertLess(time.monotonic() - started, 10)


_STARTER = '''\
import sys
sys.path.insert(0, {tmp!r})
from pythontk import BatchOperationHost, Operation, OperationCatalog, SessionOperationHost
batch = BatchOperationHost(
    sys.executable, scene_files="{pkg}.scene:JsonSceneFiles", packages=("{pkg}",), timeout=60
)
host = SessionOperationHost(batch, {session!r}, "py", keep={keep!r})
entry = Operation("py.whoami", "{pkg}.scene:whoami", host="py").describe()
print(OperationCatalog([entry], hosts={{"py": host}}).run("py.whoami").result)
host.close()
'''


class TestKeptBeyondItsStarter(_SessionFixture):
    """What a test runner needs: an instance one run starts and the next reuses."""

    KEEP = 4.0

    def _ended(self, host, within):
        deadline = time.monotonic() + within
        while time.monotonic() < deadline:
            if not host._up():
                return True
            time.sleep(0.1)
        return False

    def test_it_outlives_the_process_that_started_it_then_ends_once_idle(self):
        script = Path(self._tmp.name) / "starter.py"
        script.write_text(
            _STARTER.format(
                tmp=self._tmp.name, pkg=_PACKAGE, session=self.session, keep=self.KEEP
            ),
            encoding="utf-8",
        )
        started = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(started.returncode, 0, started.stderr)
        pid = int(started.stdout.strip().splitlines()[-1])
        host = self.host(keep=self.KEEP)
        self.addCleanup(host._kill)
        again = self.catalog(host).run("py.whoami")
        self.assertEqual((again.result, again.issues), (pid, []))  # its starter is gone
        host.close()  # a kept instance is its lease's to end, not a runner's
        self.assertTrue(host._up())
        self.assertTrue(self._ended(host, self.KEEP + 20), "it never went idle")
        self.assertEqual(host.queue.worker()["state"], "stopped")

    def test_a_session_claimed_by_one_process_is_refused_to_another(self):
        # A test run resets what its instance holds, so it takes it alone.
        script = Path(self._tmp.name) / "claimer.py"
        script.write_text(
            "import os, sys, time\n"
            "from pythontk import BatchOperationHost, SessionOperationHost\n"
            f"host = SessionOperationHost(BatchOperationHost(sys.executable), {self.session!r}, 'py')\n"
            "print(host.claim(), os.getpid(), flush=True)\n"
            "time.sleep(30)\n",
            encoding="utf-8",
        )
        other = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
        self.addCleanup(other.kill)
        claimed, pid = other.stdout.readline().split()
        self.assertEqual(claimed, "True")
        mine = self.host()
        self.assertFalse(mine.claim())
        # The interpreter itself (a venv's python.exe only launches it).
        os.kill(int(pid), signal.SIGTERM)
        other.wait(10)
        deadline = time.monotonic() + 10
        while not mine.claim() and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertTrue(mine.claim())  # let go with its process
        self.assertTrue(mine.claim())  # and held, asking again
        mine.close()
        self.assertTrue(self.host().claim())  # close lets it go

    def test_a_call_longer_than_its_keep_does_not_end_it(self):
        # The lease is held while a call runs, not only renewed after it.
        host = self.host(keep=2.0)
        self.addCleanup(host._kill)
        catalog = self.catalog(host)
        pid = catalog.run("py.whoami").result
        slow = catalog.run("py.python.exec", {"code": "import time\ntime.sleep(5)"}, apply=True)
        self.assertEqual(slow.status, "done", slow.error)
        self.assertTrue(host._up(), "a call longer than keep ended the instance")
        again = catalog.run("py.whoami")
        self.assertEqual((again.result, again.issues), (pid, []))

    def test_a_folder_it_reloads_is_current_at_each_call_without_a_restart(self):
        late = Path(self._tmp.name) / _PACKAGE / "late.py"
        late.write_text('def value() -> int:\n    """One."""\n    return 1\n', encoding="utf-8")
        catalog = OperationCatalog(
            _entries()
            + [Operation("py.late", f"{_PACKAGE}.late:value", host="py").describe()],
            hosts={"py": self.host(reload=[str(late.parent)])},
        )
        pid = catalog.run("py.whoami").result
        self.assertEqual(catalog.run("py.late").result, 1)
        time.sleep(0.05)
        # Another size too: a .pyc this process wrote is trusted within its second.
        late.write_text(
            'def value() -> int:\n    """Twenty-two."""\n    return 22\n', encoding="utf-8"
        )
        self.assertEqual(catalog.run("py.late").result, 22)
        again = catalog.run("py.whoami")
        self.assertEqual((again.result, again.issues), (pid, []))


class TestCatalogSession(_SessionFixture):
    def test_a_catalog_with_a_session_keeps_each_registered_runner(self):
        batch = self.batch()
        ep = SimpleNamespace(
            name="py", value=f"{_PACKAGE}:host", load=lambda: lambda: batch
        )
        with patch(_ENTRY_POINTS, return_value=[ep]):
            kept = OperationCatalog(_entries(), session=self.session).host("py")
            fresh = OperationCatalog(_entries()).host("py")
        self.assertIsInstance(kept, SessionOperationHost)
        self.assertIs(kept.host, batch)
        self.assertIs(fresh, batch)
        self.hosts.append(kept)


class TestAnApplicationThatServesItself(_SessionFixture):
    """A runner whose application speaks the session protocol itself (Unity's
    editor, in C#), launched through ``serve``, one instance per project."""

    def _runner(self, default=None):
        roots = [self._tmp.name, *BatchOperationHost(sys.executable)._roots()]

        def serve(folder, owner, key):
            code = (
                f"import sys; sys.path[:0] = {roots!r}; "
                "from pythontk import SessionOperationHost; "
                f"sys.exit(SessionOperationHost.serve({str(folder)!r}, "
                f"'{_PACKAGE}.scene:JsonSceneFiles', owner={owner!r}))"
            )
            return ["-c", code]

        def instance(scenes):  # the project: the folder a scene file sits in
            return os.path.basename(os.path.dirname(scenes[0])) if scenes else default

        self.launched = []

        def app(key):  # the executable an instance needs (Unity: its project's version)
            self.launched.append(key)
            return sys.executable

        return BatchOperationHost(
            app,
            command=lambda run: ["-c", "pass"],
            serve=serve,
            instance=instance,
            timeout=60,
            grace=30,
        )

    def _scene_in(self, project):
        folder = Path(self._tmp.name, project)
        folder.mkdir(exist_ok=True)
        scene = folder / "shot.json"
        scene.write_text(json.dumps({"objects": ["a"]}), encoding="utf-8")
        return scene

    def test_it_is_kept_one_instance_per_project_the_last_one_staying(self):
        host = SessionOperationHost(self._runner(), self.session, "app")
        self.hosts.append(host)
        catalog = OperationCatalog(_entries_for("app"), hosts={"app": host})
        first = catalog.run("app.whoami", scenes=[self._scene_in("alpha")])
        self.assertEqual(first.status, "done", first.error)
        self.assertNotEqual(first.result, os.getpid())
        same = catalog.run("app.whoami")  # no scenes: the project it last worked in
        self.assertEqual(same.result, first.result)
        other = catalog.run("app.whoami", scenes=[self._scene_in("beta")])
        self.assertNotEqual(other.result, first.result)
        self.assertEqual(catalog.run("app.whoami").result, other.result)
        back = catalog.run("app.whoami", scenes=[self._scene_in("alpha")])
        self.assertEqual(back.result, first.result)  # still up: one per project
        self.assertEqual(self.launched, ["alpha", "beta"])  # each launched for its own

    def test_its_default_project_is_one_instance_with_or_without_its_scenes(self):
        # Unity's sandbox: a call without scenes starts there, a call naming a
        # scene inside it reaches that same editor (a project opens once), and a
        # call without scenes stays where the session last worked.
        host = SessionOperationHost(self._runner(default="sandbox"), self.session, "app")
        self.hosts.append(host)
        catalog = OperationCatalog(_entries_for("app"), hosts={"app": host})
        first = catalog.run("app.whoami")
        self.assertEqual(first.status, "done", first.error)
        named = catalog.run("app.whoami", scenes=[self._scene_in("sandbox")])
        self.assertEqual(named.result, first.result)
        other = catalog.run("app.whoami", scenes=[self._scene_in("alpha")])
        self.assertNotEqual(other.result, first.result)
        self.assertEqual(catalog.run("app.whoami").result, other.result)
        self.assertEqual(self.launched, ["sandbox", "alpha"])

    def test_every_process_serving_the_session_works_in_its_current_instance(self):
        # The plan and apply servers are two processes: the one that did not
        # name alpha last went to the default and planned against another project.
        runner = self._runner(default="sandbox")
        applies = SessionOperationHost(runner, self.session, "app")
        plans = SessionOperationHost(runner, self.session, "app")
        self.hosts += [applies, plans]
        entries = _entries_for("app")
        worked = OperationCatalog(entries, hosts={"app": applies}).run(
            "app.whoami", scenes=[self._scene_in("alpha")]
        )
        self.assertEqual(worked.status, "done", worked.error)
        planned = OperationCatalog(entries, hosts={"app": plans}).run("app.whoami")
        self.assertEqual(planned.result, worked.result)
        self.assertEqual(self.launched, ["alpha"])

    def test_a_catalog_keeps_it_and_one_that_cannot_serve_itself_is_refused(self):
        batch = self._runner()
        ep = SimpleNamespace(name="app", value=f"{_PACKAGE}:host", load=lambda: lambda: batch)
        with patch(_ENTRY_POINTS, return_value=[ep]):
            kept = OperationCatalog(_entries_for("app"), session=self.session).host("app")
        self.hosts.append(kept)
        self.assertIsInstance(kept, SessionOperationHost)
        with self.assertRaisesRegex(ValueError, "cannot be kept"):
            SessionOperationHost(
                BatchOperationHost(sys.executable, command=lambda run: []), self.session
            )


def _entries_for(host):
    return [
        Operation(f"{host}.whoami", f"{_PACKAGE}.scene:whoami", host=host).describe()
    ]


class TestPlainPythonHost(_SessionFixture):
    """The ``python`` host: code for a function no application owns."""

    def test_python_exec_runs_in_a_kept_plain_python_registered_as_python(self):
        ep = SimpleNamespace(
            name=PureSceneFiles.HOST,
            value="pythontk:PureSceneFiles.operation_host",
            load=lambda: PureSceneFiles.operation_host,
        )
        (entry,) = [
            e
            for e in OperationProvider.document(
                OperationProvider.declarations("pythontk.operations"), "pythontk"
            )["operations"]
            if e.get("host") == PureSceneFiles.HOST
        ]
        self.assertEqual(entry["id"], "python.exec")
        with patch(_ENTRY_POINTS, return_value=[ep]):
            catalog = OperationCatalog([entry], session=self.session)
            runner = catalog.host(PureSceneFiles.HOST)
        self.hosts.append(runner)
        self.assertEqual(runner.host.app, sys.executable)
        catalog.run("python.exec", {"code": "import os\nkept = os.getpid()"}, apply=True)
        again = catalog.run("python.exec", {"code": "kept"}, apply=True)
        self.assertEqual(again.status, "done", again.error)
        self.assertNotEqual(again.result["value"], os.getpid())
        refused = catalog.run("python.exec", {"code": "1"}, apply=True, scenes=[self.scene])
        self.assertIn("opens no scenes", refused.error)

    def test_code_is_served_where_changes_are_never_as_a_plan(self):
        exec_entry = Operation(
            "python.exec", "pythontk:PythonConsole.run", effect="code", host="python"
        ).describe()
        hosts = {"python": object()}
        plans = OperationServer(OperationCatalog([exec_entry], hosts=hosts, session="s"))
        changes = OperationServer(
            OperationCatalog([exec_entry], hosts=hosts, session="s"), apply=True
        )
        self.assertEqual([t["name"] for t in plans.tools()], [])
        self.assertEqual([t["name"] for t in changes.tools()], ["python_exec"])
        # However it is declared: a code entry that says it plans (its function
        # takes a ctx, which operations.check flags) still runs code.
        planned = dict(exec_entry, dry_run=True)
        plans = OperationServer(OperationCatalog([planned], hosts=hosts, session="s"))
        self.assertEqual([t["name"] for t in plans.tools()], [])

    def test_its_tool_takes_no_scenes(self):
        exec_entry = Operation(
            "python.exec", "pythontk:PythonConsole.run", effect="code", host="python"
        ).describe()
        (tool,) = OperationServer(
            OperationCatalog([exec_entry], hosts={"python": object()}, session="s"),
            apply=True,
        ).tools()
        self.assertNotIn("scenes", tool["inputSchema"]["properties"])
        server = OperationServer(
            OperationCatalog([exec_entry], hosts={"python": object()}, session="s"),
            apply=True,
        )
        self.assertNotIn("`scenes`", server.instructions())  # nor its instructions


class TestOperationServerSessions(_SessionFixture):
    class _Host:
        def run(self, entry, request, scope=None, on_progress=None):
            return Outcome(op=entry["id"], status="done")

    def test_the_server_says_whether_host_calls_run_in_a_kept_or_a_fresh_instance(self):
        fresh = OperationServer(
            OperationCatalog(_entries(), hosts={"py": self._Host()})
        )
        kept = OperationServer(
            OperationCatalog(_entries(), hosts={"py": self._Host()}, session="s")
        )
        self.assertIn("FRESH instance", fresh.instructions())
        self.assertIn("this session's own instance", kept.instructions())
        describe = {
            s: {t["name"]: t["description"] for t in s.tools()} for s in (fresh, kept)
        }
        self.assertIn("a fresh py instance per call", describe[fresh]["py_count"])
        self.assertIn("this session's py instance", describe[kept]["py_count"])
        self.assertIn(
            "replacing what it holds",
            kept.tools()[0]["inputSchema"]["properties"]["scenes"]["description"],
        )

    def test_the_session_is_the_clients_when_it_names_one(self):
        with patch.dict(
            os.environ, {"PYTHONTK_OPS_SESSION": "", "CLAUDE_CODE_SESSION_ID": "abc"}
        ):
            self.assertEqual(OperationServer.session_name(), "abc")
        with patch.dict(
            os.environ,
            {"PYTHONTK_OPS_SESSION": "mine", "CLAUDE_CODE_SESSION_ID": "abc"},
        ):
            self.assertEqual(OperationServer.session_name(), "mine")
        with patch.dict(
            os.environ, {"PYTHONTK_OPS_SESSION": "", "CLAUDE_CODE_SESSION_ID": ""}
        ):
            self.assertEqual(OperationServer.session_name(), f"pid{os.getpid()}")

    def test_an_image_the_run_made_comes_back_as_image_content(self):
        root, assets = Path(self._tmp.name, "root"), Path(self._tmp.name, "assets")
        root.mkdir()
        assets.mkdir()
        made, texture = (
            root / "view.png",
            assets / "texture.png",
        )  # a person's asset: never inlined
        for path in (made, texture):
            path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        def image(result, status="done"):
            outcome = Outcome(
                op="py.view",
                status=status,
                result=str(result),
                error=None if status == "done" else "x",
            )
            return OperationServer._image(outcome)

        with patch.object(TempArtifacts, "root", return_value=str(root)):
            self.assertEqual(
                (image(made)["type"], image(made)["mimeType"]), ("image", "image/png")
            )
            self.assertIsNone(image(made, status="failed"))
            self.assertIsNone(image(texture))
            self.assertIsNone(image("no image"))


if __name__ == "__main__":
    unittest.main()
