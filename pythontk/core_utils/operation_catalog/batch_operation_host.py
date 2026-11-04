# !/usr/bin/python
# coding=utf-8
"""BatchOperationHost -- an application's runner: every call in a fresh instance.

The runner a host library registers under ``pythontk.operation_hosts`` (one
per host name: ``maya``, ``blender``) is one of these, configured: which executable, how it is started on a run folder, and
what reads its failures. It follows the batch protocol
(:class:`~pythontk.OpRequest`): a fresh run folder per call, ``request.json``
(and the resolved ``entry.json``) in, ``outcome.json`` out, and a NEW instance
of the application each time, so an operation never reaches a session a
person has open, and never overwrites their unsaved work. A front end that
lives long (the MCP server) keeps one instance per session instead:
:class:`~pythontk.SessionOperationHost` wraps this runner and serves the same
run folders, its application, scene files and packages, from one instance.

How the instance serves the folder is a strategy. By default it runs Python:
the host writes a script into the folder that calls :meth:`serve`, which opens
the request's scenes through the application's :class:`SceneFiles`, hands the
call to :class:`~pythontk.PureOperationHost` (never back to a catalog, which
would route it to a runner again), saves the scenes a finished live run of
an operation that plans changed (code saves what it saves itself) and writes
the outcome. An application that serves a folder itself
(a C# server in a Unity project) passes ``command`` instead: the host still
owns the folder, the cancel file, the progress, the timeout and reading the
outcome, so every application's runner shares them.

A run that does not finish saves nothing, so its scene files are as they were,
and its outcome says so. A dry run lists the saves it would make with its plan.
A live run holds its scenes while it runs, so two calls in this process that
change one scene take turns; across processes (a person, a second server) it
saves nothing over a scene file that changed on disk after it opened it.
Cancelling is cooperative: a cancel of the caller's scope (the one passed, or
the ambient one) drops a cancel file in the run folder and the operation stops
at its next checkpoint; code, which has none, is interrupted as Ctrl+C does
(``INTERRUPT_AFTER`` seconds after the cancel). The launch runs outside the caller's ambient scope, so
nothing kills the instance early; one still running ``grace`` seconds after
the cancel is killed, but never while it saves its scenes (it marks the save
in the run folder), so a cancel never cuts a save short. The timeout is the
backstop for a hung instance and kills it whatever it is doing; a live run
killed is ``partial``, naming any scene it was saving. An outcome written
before an instance hung on its way out still stands. Progress comes back as
:class:`~pythontk.ProgressRelay` marker lines on the instance's output
(``::progress:: <step>/<steps> <text>``).
"""

from __future__ import annotations

import _thread
import contextlib
import contextvars
import inspect
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from pythontk.core_utils.cancel_scope import CancelScope, OperationCancelled
from pythontk.core_utils.operation_catalog.operation import Operation
from pythontk.core_utils.operation_catalog.operation_host import PureOperationHost
from pythontk.core_utils.operation_catalog.outcome import Change, OpRequest, Outcome


@runtime_checkable
class SceneFiles(Protocol):
    """An application's scene files, inside a batch run (its library implements it).

    Constructed in the fresh instance, before anything else runs there: a
    constructor is where an application's Python API starts (``maya.standalone``).
    An instance a session keeps also asks ``unsaved() -> bool``, when the
    class has one: whether what it holds has changes no file has (a scene
    never saved included); without it, what ``modified()`` names. A host
    whose own dirty flag misses an edit made from code (Blender's data API)
    has ``changed() -> None`` too: it marks what it holds as changed, so
    ``modified()`` and ``unsaved()`` see the edit. It is called before a live
    run that keeps no journal (code: a save the code makes clears the mark)
    and after one whose journal names a change.
    """

    def open(self, paths: Sequence[str]) -> None:
        """Open *paths*, the first as the main scene; raise when one cannot be."""

    def modified(self) -> List[str]:
        """The open scene files that have unsaved changes."""

    def save(self, paths: Sequence[str]) -> None:
        """Save *paths* where they are, in their own format."""

    def close(self) -> None:
        """Release the application before the process exits."""


class PureSceneFiles:
    """No scenes: a plain Python instance. A request that names one is refused.

    Its runner is the ``python`` host pythontk registers: code for a function
    no application owns runs there (``python.exec``).
    """

    #: The host name its runner is registered under (``pythontk.operation_hosts``).
    HOST = "python"

    @classmethod
    def operation_host(cls) -> "BatchOperationHost":
        """The ``python`` runner: a fresh instance of this interpreter per call
        (``pythontk.operation_hosts``)."""
        return BatchOperationHost(sys.executable)

    def open(self, paths: Sequence[str]) -> None:
        if paths:
            raise ValueError(
                f"this host opens no scenes (the request names {len(paths)})"
            )

    def modified(self) -> List[str]:
        return []

    def unsaved(self) -> bool:
        return False

    def save(self, paths: Sequence[str]) -> None:
        if paths:
            raise ValueError("this host saves no scenes")

    def close(self) -> None:
        pass


class _RunFolderInternal:
    """The run-folder protocol every runner of an application shares: the
    files a call's folder holds, and the request handling the instance runs
    (:class:`BatchOperationHost` per call, :class:`~pythontk.SessionOperationHost`
    for a session)."""

    #: The entry the runner resolved, beside the request: the instance runs
    #: exactly what the runner validated (a checkout's working declarations
    #: included) without discovering the catalog again.
    ENTRY_FILE = "entry.json"

    #: Dropped in the run folder to stop the run at its next checkpoint.
    CANCEL_FILE = "cancel"

    #: Held in the run folder while the instance saves its scenes (it names
    #: them): the grace period never kills a save.
    SAVING_FILE = "saving"

    @staticmethod
    def _write(path: Path, text: str) -> None:
        """Write *text* whole or not at all: a reader never sees half a document."""
        part = path.with_name(path.name + ".part")
        part.write_text(text, encoding="utf-8")
        os.replace(part, path)

    @staticmethod
    def _key(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    @classmethod
    def _stamps(cls, paths: Sequence[str]) -> Dict[str, Tuple[int, int]]:
        """Each existing file of *paths* as its modification time and size."""
        stamps = {}
        for path in paths:
            try:
                stat = os.stat(path)
            except OSError:
                continue
            stamps[cls._key(path)] = (stat.st_mtime_ns, stat.st_size)
        return stamps

    @staticmethod
    def _lost(
        entry: Mapping[str, Any],
        request: OpRequest,
        error: str,
        cancelled: bool = False,
    ) -> Outcome:
        """The outcome of a run whose instance never wrote one: ``partial`` when
        it was a live change (no checkpoint stopped it, so what it did is
        unknown), else ``cancelled`` when the caller cancelled it, else ``failed``."""
        changes = not request.dry_run and entry.get("effect") != Operation.READ
        if changes:
            error += "; what it changed before it stopped is unknown"
        status = "partial" if changes else "cancelled" if cancelled else "failed"
        return Outcome(
            op=request.op, status=status, dry_run=request.dry_run, error=error
        )

    @classmethod
    def _settle(
        cls,
        outcome: Outcome,
        request: OpRequest,
        entry: Mapping[str, Any],
        files: SceneFiles,
        run_dir: Path,
        stamps: Mapping[str, Tuple[int, int]],
    ) -> Outcome:
        """In the instance: save the scenes a finished live run changed (a dry
        run lists them), and say which a run that did not finish left alone.
        *stamps* are the scene files as they were opened: one changed on disk
        since is never saved over."""
        if not request.scenes or entry.get("effect") in (Operation.READ, Operation.CODE):
            return outcome  # code saves what it saves itself: it has no plan to save by
        changed = any(c.kind != "write" for c in outcome.changes)
        if request.dry_run:
            if changed:
                outcome.changes += [
                    Change(kind="write", target=p, action="save the scene")
                    for p in request.scenes
                ]
            return outcome
        if entry.get("dry_run") and not changed:
            return outcome  # it keeps a journal, and the journal changed no scene
        # Report each scene as the request spelled it, not as the application does.
        spelled = {cls._key(p): p for p in request.scenes}
        saving = run_dir / cls.SAVING_FILE
        try:
            modified = [spelled.get(cls._key(p), p) for p in files.modified()]
            if modified and outcome.status != "done":
                outcome.error = f"{outcome.error} (not saved, so as they were: {', '.join(modified)})"
                return outcome
            now = cls._stamps(modified)
            moved = [
                p
                for p in modified
                if cls._key(p) in stamps and now.get(cls._key(p)) != stamps[cls._key(p)]
            ]
            if moved:
                outcome.status = "failed"
                outcome.error = (
                    f"{', '.join(moved)} changed on disk since this run opened it, so it "
                    "saved nothing over it; run it again on the file as it is now"
                )
                return outcome
            cls._write(saving, "\n".join(modified))
            try:
                files.save(modified)
            finally:
                saving.unlink()
        except Exception as exc:
            outcome.status = "partial"
            outcome.error = f"saving the scenes failed: {type(exc).__name__}: {exc}"
            return outcome
        outcome.changes += [
            Change(kind="write", target=p, action="save the scene") for p in modified
        ]
        return outcome

    # -- in the instance ---------------------------------------------------------
    @staticmethod
    def _begin(scene_files: str) -> Tuple[Optional[SceneFiles], Optional[Exception]]:
        """In the instance: its :class:`SceneFiles`, which starts the
        application's Python API, or why it could not be built."""
        try:
            return Operation.resolve_target(scene_files)(), None
        except Exception as exc:
            return None, exc

    @staticmethod
    def _finish(files: Optional[SceneFiles]) -> None:
        """In the instance: release the application before the process exits."""
        if files is None:
            return
        try:
            files.close()
        except Exception as exc:
            print(f"closing the scenes failed: {exc}", file=sys.stderr, flush=True)

    @classmethod
    def _answer(
        cls,
        run: Path,
        files: Optional[SceneFiles],
        on_progress: Callable[[float, str], None],
        failure: Optional[Exception] = None,
    ) -> Outcome:
        """In the instance: run the request in *run* with *files*, write its
        outcome there and hand it back; *failure* is why *files* could not be
        built (the request is refused with it).

        Opening scenes over unsaved changes is refused, as an application
        asks before it discards them: an instance a session keeps may hold
        what a live run changed since it last opened scenes (a ``python.exec``),
        which no file has yet. A fresh instance holds none.
        """
        op_id, dry_run = "?", False
        try:
            data = json.loads(
                (run / OpRequest.REQUEST_FILE).read_text(encoding="utf-8")
            )
            op_id, dry_run = str(data.get("op") or "?"), bool(data.get("dry_run"))
            if failure is not None:
                raise failure
            if (
                not isinstance(data.get("format"), int)
                or data["format"] > OpRequest.FORMAT
            ):
                raise ValueError(
                    f"request format {data.get('format')!r} is not one this pythontk "
                    f"reads (up to {OpRequest.FORMAT})"
                )
            errors = OpRequest.validate(data).errors
            if errors:
                raise ValueError("; ".join(errors))
            request = OpRequest.from_dict(data)
            entry = json.loads((run / cls.ENTRY_FILE).read_text(encoding="utf-8"))
        except Exception as exc:
            outcome = Outcome.refused(
                op_id, f"the run could not start: {type(exc).__name__}: {exc}", dry_run
            )
        else:
            outcome = cls._answer_request(run, files, request, entry, on_progress)
        cls._write(run / OpRequest.OUTCOME_FILE, outcome.to_json())
        return outcome

    #: Whether a live run changed what this instance holds since it last
    #: opened scenes (set and read in the instance; a fresh one starts with
    #: none); the host's own flag then says whether a file has them.
    _held_work = False
    #: Seconds a cancelled call has to stop at a checkpoint before it is
    #: interrupted (as Ctrl+C does), well inside a caller's grace.
    INTERRUPT_AFTER = 2.0

    @classmethod
    @contextlib.contextmanager
    def _interrupt_on_cancel(cls, run: Path, fired: threading.Event):
        """While code runs on this process's main thread: once its call's
        cancel file appears and it has not ended within :attr:`INTERRUPT_AFTER`,
        raise ``KeyboardInterrupt`` in it at its next Python step (*fired* says
        so), as Ctrl+C does in a console, so code stops and the instance lives on.

        Code alone: an operation with checkpoints stops at one and keeps its
        journal. ``ExecutionMonitor`` avoids ``interrupt_main`` because it lands
        at an arbitrary step and cannot reach a native call; for code that is
        the point (it has no other place to stop), a native call (a save the
        code makes included) finishes before it lands, the runner's grace stops
        one that never returns, and :meth:`_settle`'s saves run after this
        watch ends. An application that gave Python no SIGINT handler
        (background Blender, measured in 5.1) gets Python's default, else the
        interrupt is dropped; one that ignores SIGINT on purpose is left alone."""
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        if signal.getsignal(signal.SIGINT) in (None, signal.SIG_DFL):
            signal.signal(signal.SIGINT, signal.default_int_handler)
        done = threading.Event()

        def watch() -> None:
            seen = None
            while not done.wait(0.05):
                if seen is None:
                    if (run / cls.CANCEL_FILE).exists():
                        seen = time.monotonic()
                elif time.monotonic() - seen > cls.INTERRUPT_AFTER:
                    fired.set()
                    _thread.interrupt_main()
                    return

        watcher = threading.Thread(target=watch, name="ops-interrupt", daemon=True)
        watcher.start()
        try:
            yield
        finally:
            done.set()
            watcher.join()

    @staticmethod
    def _mark(files: SceneFiles) -> None:
        """Tell a host whose dirty flag misses code edits that what it holds
        changed (``SceneFiles.changed``), when it has the hook."""
        mark = getattr(files, "changed", None)
        if not callable(mark):
            return
        try:
            mark()
        except Exception as exc:
            print(f"marking the scenes changed failed: {exc}", file=sys.stderr, flush=True)

    @classmethod
    def _unsaved(cls, files: SceneFiles) -> bool:
        """Whether opening scenes now would discard unsaved changes."""
        if not cls._held_work:
            return False
        unsaved = getattr(files, "unsaved", None)
        return bool(unsaved() if callable(unsaved) else files.modified())

    @classmethod
    def _answer_request(
        cls,
        run: Path,
        files: SceneFiles,
        request: OpRequest,
        entry: Mapping[str, Any],
        on_progress: Callable[[float, str], None],
    ) -> Outcome:
        """The body of :meth:`_answer`, the request read."""
        if request.scenes and cls._unsaved(files):
            return Outcome.refused(
                request.op,
                "this instance holds unsaved changes (a live run since it last opened "
                "scenes changed what it holds, and no file has them), which opening "
                "scenes would discard: save them from code, call without scenes to work "
                "on them, or discard them from code first",
                request.dry_run,
            )
        try:
            stamps = cls._stamps(request.scenes)
            files.open(list(request.scenes))
        except Exception as exc:
            return Outcome.refused(
                request.op,
                f"the run could not start: {type(exc).__name__}: {exc}",
                request.dry_run,
            )
        if request.scenes:
            _RunFolderInternal._held_work = False  # replaced by the files just opened
        # A live run that keeps no journal may change anything it reaches: what
        # it holds is marked changed BEFORE it runs, so a save the code makes
        # clears the mark as the application's own save would.
        journal = bool(entry.get("dry_run")) or entry.get("effect") == Operation.READ
        if not request.dry_run and not journal:
            cls._mark(files)
        scope = CancelScope().add_source((run / cls.CANCEL_FILE).exists)
        fired = threading.Event()
        watch = (
            cls._interrupt_on_cancel(run, fired)
            if entry.get("effect") == Operation.CODE
            else contextlib.nullcontext()
        )
        try:
            with watch:
                outcome = PureOperationHost().run(
                    entry, request, scope=scope, on_progress=on_progress
                )
            if fired.is_set():
                for _ in range(10000):  # an interrupt still on its way lands here
                    pass
        except KeyboardInterrupt:  # the cancel's interrupt, or the code's own
            live = not request.dry_run and entry.get("effect") != Operation.READ
            if fired.is_set():
                why = (
                    "interrupted after the cancel: it did not end within "
                    f"{cls.INTERRUPT_AFTER:g} s"
                )
            else:
                why = "KeyboardInterrupt"
            outcome = Outcome(
                op=request.op,
                status="partial" if live else "cancelled" if fired.is_set() else "failed",
                dry_run=request.dry_run,
                error=why + (" (what it changed before is unknown)" if live else ""),
            )
        changed = not request.dry_run and (
            any(c.kind != "write" for c in outcome.changes)
            or (not journal and outcome.status != "failed")
        )
        if changed and journal:  # its journal names changes the host's flag may miss
            cls._mark(files)
        outcome = cls._settle(outcome, request, entry, files, run, stamps)
        if changed:  # whether a file now has them, the host's own flag says
            _RunFolderInternal._held_work = True
        return outcome


class _BatchOperationHostInternal(_RunFolderInternal):
    """Internal helpers for :class:`BatchOperationHost`: its per-call launch."""

    def _instance_key(self, scenes: Sequence[str]) -> Optional[str]:
        """The instance *scenes* need (``instance``), ``None`` for the default."""
        if self.instance is None:
            return None
        try:
            return self.instance(list(scenes))
        except Exception:  # the run says what is wrong with its scenes
            return None

    def _executable(self, key: Optional[str] = None) -> Optional[str]:
        """The application to launch: ``app``, called with the instance key
        when it takes one (the editor a Unity project names)."""
        app = self.app
        if not callable(app):
            return app
        try:
            takes_key = bool(inspect.signature(app).parameters)
        except (TypeError, ValueError):
            takes_key = False
        return app(key) if takes_key else app()

    #: The Python script the default strategy runs, written in the run folder.
    SCRIPT_FILE = "serve.py"

    #: Progress markers count in thousandths.
    STEPS = 1000

    #: How much of the output's end a ``diagnose`` strategy is handed.
    TAIL_CHARS = 8000

    #: One lock per scene file a live run may save, shared by every runner in
    #: this process; how often a held one is checked for the caller's cancel.
    _scene_locks: Dict[str, threading.Lock] = {}

    _scene_locks_guard = threading.Lock()

    _WAIT_POLL = 0.25

    @classmethod
    def _hold(
        cls,
        request: OpRequest,
        scope: Optional[CancelScope],
        on_progress: Optional[Callable[[float, str], None]],
    ) -> Optional[List[threading.Lock]]:
        """Take the locks of the scenes a live *request* may save, in one order
        (so two runs never wait on each other); ``None`` when *scope* is
        cancelled while another run holds one. A dry run takes none."""
        scenes = {} if request.dry_run else {cls._key(p): p for p in request.scenes}
        with cls._scene_locks_guard:
            locks = [
                (cls._scene_locks.setdefault(k, threading.Lock()), scenes[k])
                for k in sorted(scenes)
            ]
        held: List[threading.Lock] = []
        for lock, path in locks:
            told = False
            while not lock.acquire(timeout=cls._WAIT_POLL):
                if scope is not None and scope.poll():
                    for taken in held:
                        taken.release()
                    return None
                if on_progress is not None and not told:
                    on_progress(0.0, f"waiting for another run to finish with {path}")
                    told = True
            held.append(lock)
        return held

    @staticmethod
    def _provider(entry: Mapping[str, Any]) -> str:
        """The top-level package *entry*'s function is in."""
        return str(entry.get("target", "")).partition(":")[0].split(".")[0]

    def _roots(self, *providers: str) -> List[str]:
        """The import roots an instance puts first on its ``sys.path``:
        pythontk's, this host's packages' and *providers*'."""
        from pythontk.core_utils.handoff.app_handoff import HandoffBridge

        return HandoffBridge.import_roots(
            *dict.fromkeys(p for p in ("pythontk", *self.packages, *providers) if p)
        )

    def _bootstrap(
        self, folder: Path, call: str, roots: List[str]
    ) -> Tuple[List[str], Any]:
        """The arguments and environment that start an instance on a script
        written in *folder*: *roots*, then this process's own import roots,
        lead its ``sys.path`` (``roots`` names the list there), and it exits
        with *call*'s code (``"BatchOperationHost.serve(...)"``)."""
        from pythontk.core_utils.app_launcher import AppLauncher
        from pythontk.core_utils.handoff.app_handoff import HandoffBridge

        path = list(roots) + [
            p for p in HandoffBridge.child_sys_path() if p not in roots
        ]
        owner = call.split(".", 1)[0]
        script = folder / self.SCRIPT_FILE
        self._write(
            script,
            "import sys\n"
            f"roots = {list(roots)!r}\n"
            f"sys.path[:0] = {path!r}\n"
            f"from pythontk import {owner}\n"
            f"sys.exit({call})\n",
        )
        if self.launch_args is not None:
            return list(self.launch_args(str(script))), self.env
        # A Python interpreter (mayapy): the script's path rides the environment,
        # never the command line, which mayapy decodes in the ANSI code page.
        return AppLauncher.python_args_via_env([str(script)], self.env)

    def _launch(self, run_dir: Path, entry: Mapping[str, Any]) -> Tuple[List[str], Any]:
        """The instance's arguments and environment for *run_dir*."""
        if self.command is not None:
            return list(self.command(run_dir)), self.env
        call = f"BatchOperationHost.serve({str(run_dir)!r}, {self.scene_files!r})"
        return self._bootstrap(run_dir, call, self._roots(self._provider(entry)))

    @classmethod
    def _report_progress(cls, fraction: float, message: str) -> None:
        """In the instance: one progress marker line the runner reads."""
        from pythontk.core_utils.handoff.script_run import ProgressRelay

        line = ProgressRelay.line(round(fraction * cls.STEPS), cls.STEPS, message)
        print(line, file=sys.stderr, flush=True)

    def _reader(
        self,
        run_dir: Path,
        scope: Optional[CancelScope],
        on_progress: Optional[Callable[[float, str], None]],
    ) -> Callable[[Optional[str]], bool]:
        """The launch's ``on_output``: progress markers to *on_progress*, and a
        cancel of *scope* to the cancel file. It stops the instance only when
        the run has not ended :attr:`grace` seconds after the cancel and is not
        saving its scenes."""
        from pythontk.core_utils.handoff.script_run import ProgressRelay

        cancel, saving = run_dir / self.CANCEL_FILE, run_dir / self.SAVING_FILE
        asked: List[float] = []

        def on_output(line: Optional[str]) -> bool:
            # poll(), not the flag: on this (the owner's) thread it also reads
            # the scope's pull sources, such as a held Esc key.
            if scope is not None and not asked and scope.poll():
                cancel.write_text(scope.reason or "cancelled", encoding="utf-8")
                asked.append(time.monotonic())
            marker = ProgressRelay.parse(line)
            if marker is not None and on_progress is not None:
                step, steps, text = marker
                on_progress(step / steps if steps else 0.0, text)
            late = (
                asked
                and self.grace is not None
                and time.monotonic() - asked[0] > self.grace
            )
            return not (late and not saving.exists())

        return on_output

    def _run_held(
        self,
        entry: Mapping[str, Any],
        request: OpRequest,
        caller: Optional[CancelScope],
        on_progress: Optional[Callable[[float, str], None]],
    ) -> Outcome:
        """The body of :meth:`BatchOperationHost.run`, its scenes held."""
        from pythontk.core_utils.app_launcher import AppLauncher
        from pythontk.core_utils.handoff.script_run import ScriptRunner
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        host = entry.get("host")
        try:
            exe = self._executable(self._instance_key(request.scenes))
            if not exe:
                raise FileNotFoundError("no executable found")
        except Exception as exc:
            return Outcome.refused(
                request.op, f"cannot launch {host!r}: {exc}", request.dry_run
            )

        store = TempArtifacts("ops_run", policy="scoped")
        run_dir = Path(store.dir_path())
        kept = f"run folder kept: {run_dir}"
        proc, stopped = None, None
        try:
            self._write(
                run_dir / OpRequest.REQUEST_FILE,
                json.dumps(request.to_dict(), indent=2),
            )
            self._write(run_dir / self.ENTRY_FILE, json.dumps(dict(entry), indent=2))
            args, env = self._launch(run_dir, entry)
            # The run folder is the instance's working directory, so what an
            # application drops there (a disk cache) goes with it.
            proc = contextvars.Context().run(
                AppLauncher.run,
                exe,
                args=args,
                cwd=str(run_dir),
                timeout=self.timeout,
                env=env,
                hide_window=True,
                on_output=self._reader(run_dir, caller, on_progress),
            )
        except OperationCancelled:
            stopped = f"it did not stop within {self.grace:g} s of the cancel, so it was killed"
        except subprocess.TimeoutExpired:
            stopped = f"it timed out after {self.timeout:g} s"
        except Exception as exc:
            return Outcome.refused(
                request.op,
                f"cannot launch {host!r}: {type(exc).__name__}: {exc}",
                request.dry_run,
            )
        outcome_path = run_dir / OpRequest.OUTCOME_FILE
        if not outcome_path.is_file():
            if stopped:
                saving = run_dir / self.SAVING_FILE
                if saving.is_file():
                    names = ", ".join(saving.read_text(encoding="utf-8").splitlines())
                    stopped += f" while it saved {names}, so those files may be damaged"
                cancelled = caller is not None and caller.cancelled
                return self._lost(
                    entry, request, f"{host}: {stopped} ({kept})", cancelled
                )
            output = proc.stdout or ""
            said = (
                self.diagnose(proc.returncode, output[-self.TAIL_CHARS :])
                if self.diagnose
                else None
            )
            said = said or ScriptRunner.failure_summary(output) or "it printed no error"
            return self._lost(
                entry,
                request,
                f"{host} exited without an outcome (exit code {proc.returncode}): {said} ({kept})",
            )
        try:
            data = json.loads(outcome_path.read_text(encoding="utf-8"))
            errors = Outcome.validate(data).errors
            if errors:
                raise ValueError("; ".join(errors))
            outcome = Outcome.from_dict(data)
        except Exception as exc:
            return self._lost(
                entry, request, f"{host} wrote an unreadable outcome: {exc} ({kept})"
            )
        if stopped:
            # It wrote its outcome, then hung on the way out: the outcome stands.
            note = f"(the instance did not exit: {stopped})"
            outcome.error = f"{outcome.error} {note}" if outcome.error else note
        store.cleanup()
        return outcome


class BatchOperationHost(_BatchOperationHostInternal):
    """Runs each call in a fresh instance of an application (see the module docstring).

    Parameters:
        app: The executable, or a callable that finds it (called per run, so
            an install made later counts; it returns ``None`` or raises when
            there is none). A callable that takes an argument is given the
            instance's key (``instance``): the version a Unity project names.
        scene_files: The :class:`SceneFiles` class the instance builds, as
            ``"module:qualified.name"`` (imported there, never here). Python
            strategy only.
        launch_args: Maps the written script's path to the application's
            arguments; the default runs it as a Python interpreter does
            (``mayapy``). Blender: ``["--background", "--factory-startup",
            "--python", path]``. Python strategy only.
        packages: Packages the instance must import, beyond pythontk and the
            operation's own (their roots lead its ``sys.path``). Python
            strategy only.
        timeout: Seconds a launch may take before it is stopped.
        env: Environment for the instance (default: this process's).
        command: For an application that serves a run folder itself: maps the
            folder to its arguments (Unity: ``-batchmode -projectPath P
            -executeMethod ... -runFolder <folder>``). It replaces the Python
            strategy; the folder, cancel, progress, timeout and outcome stay
            the host's.
        diagnose: Reads why an instance wrote no outcome: called with its exit
            code and the end of its output, it returns the error to report (a
            compile error, a held project lock, a licence wall), or ``None``
            for the generic one (the last error line it printed).
        grace: Seconds a cancelled run has to stop at a checkpoint before the
            instance is killed (``None``: only the timeout kills it).
        serve: For a *command* application that also serves a session's calls
            itself (Unity's editor, in C#): maps ``(session folder, owner pid or
            None, instance key or None)`` to the arguments that start an
            instance serving that folder, so a long-lived front end keeps one
            (:class:`~pythontk.SessionOperationHost`); without it such an
            application is launched per call.
        instance: Which instance a call needs, from its scenes: maps them to a
            key (Unity: their project folder), ``None`` for the runner's
            default; a session keeps one instance per key.

    Example:
        BatchOperationHost(
            "blender",
            scene_files="blendertk:BlenderSceneFiles",
            launch_args=lambda p: ["--background", "--factory-startup", "--python", p],
            packages=("blendertk",),
        )
    """

    def __init__(
        self,
        app: Union[str, Callable[[], Optional[str]]],
        scene_files: str = "pythontk:PureSceneFiles",
        launch_args: Optional[Callable[[str], Sequence[str]]] = None,
        packages: Sequence[str] = (),
        timeout: Optional[float] = 600.0,
        env: Optional[Mapping[str, str]] = None,
        command: Optional[Callable[[Path], Sequence[str]]] = None,
        diagnose: Optional[Callable[[int, str], Optional[str]]] = None,
        grace: Optional[float] = 60.0,
        serve: Optional[Callable[[Path, Optional[int], Optional[str]], Sequence[str]]] = None,
        instance: Optional[Callable[[Sequence[str]], Optional[str]]] = None,
    ):
        self.app = app
        self.scene_files = scene_files
        self.launch_args = launch_args
        self.packages = tuple(packages)
        self.timeout = timeout
        self.env = dict(env) if env is not None else None
        self.command = command
        self.diagnose = diagnose
        self.grace = grace
        self.serve_args = serve
        self.instance = instance

    def run(
        self,
        entry: Mapping[str, Any],
        request: OpRequest,
        scope: Optional[CancelScope] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> Outcome:
        """Run *request* in a fresh instance and hand back the outcome it wrote."""
        # The caller's cancel, explicit or ambient, reaches the run through the
        # cancel file; the launch itself runs in a fresh context, so no ambient
        # scope can kill the instance mid-save (AppLauncher stops a child when
        # the ambient scope is cancelled).
        caller = scope if scope is not None else CancelScope.current()
        held = self._hold(request, caller, on_progress)
        if held is None:
            return Outcome(
                op=request.op,
                status="cancelled",
                dry_run=request.dry_run,
                error="cancelled while another run held its scene",
            )
        try:
            return self._run_held(entry, request, caller, on_progress)
        finally:
            for lock in held:
                lock.release()

    @classmethod
    def serve(
        cls, run_dir: Union[str, Path], scene_files: str = "pythontk:PureSceneFiles"
    ) -> int:
        """In the fresh instance: run the request in *run_dir* and write its outcome.

        Parameters:
            run_dir: The run folder holding the request and the entry.
            scene_files: The :class:`SceneFiles` class, ``"module:qualified.name"``.

        Returns:
            The exit code the instance ends with: 0 when the run finished.
        """
        files, failure = cls._begin(scene_files)
        outcome = cls._answer(Path(run_dir), files, cls._report_progress, failure)
        cls._finish(files)
        return 0 if outcome.ok else 1
