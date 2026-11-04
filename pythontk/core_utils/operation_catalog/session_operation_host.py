# !/usr/bin/python
# coding=utf-8
"""SessionOperationHost -- an application's runner that keeps one instance for a session.

The runner a long-lived front end (the MCP server) wraps around each host's
:class:`~pythontk.BatchOperationHost`: the same application, scene files,
packages and run folders, but ONE instance per session and host, started on
the first call and kept. A call costs what the operation costs, not a launch,
and what one call leaves in the instance (an open scene, a console variable)
the next one sees. Every runner given one session name, in any process, shares
the instance, so a client's two servers (the plans and the changes) work on
one scene.

The instance serves a :class:`~pythontk.JobQueue` in a folder of pythontk's
temp root named for the host and the session: each call is a job naming its
run folder (the batch protocol's request, entry, cancel and saving files, and
a progress file), taken in order, so calls from every process take turns.

The instance holds an exclusive lock on a file in that folder for as long as
it lives, and the operating system releases it however the process ends: a
held lock is what says an instance is up (a heartbeat lags when a host
command holds the interpreter; a pid may name another process by now). Only
while the lock is held is the pid the instance recorded its own, so only then
is it stopped by that pid.

Session safety holds by construction: the instance is one this tooling started
(nothing else serves the queue), never a session a person opened, and it is
bound to the process that started it (``AppLauncher.spawn``; it also watches
that process, and ends when it is gone) -- unless the runner asks to *keep* it:
then it outlives its starter, for the next run to reuse (a test runner's), and
ends by itself once no call has come for that long (the queue's lease). It is reused only while its code is
current: before each call it compares each module it loaded from code it was
handed (its import roots, and every folder on its path outside its own Python's
install: a repo the console imported from) with the file as it was when
loaded, and when one changed it ends and a fresh instance takes the call. A
folder the runner names to *reload* is the exception: its modules are imported
afresh for each call instead (a test runner's package and tests, which its
driver purges between modules anyway), so a change there costs no restart. Whenever a call is served by another instance
than this runner's last call was (a code change, a crash, a stop), its
outcome says so: what the old one held is gone.

Scenes, as for a fresh instance: a call that names ``scenes`` opens them and a
finished live run of an operation that plans saves those it changed; a call
that names none works on what the instance holds. Opening scenes over changes
no file has yet (a live run changed what it holds since it last opened scenes) is refused, as an
application asks before it discards them. A cancel withdraws a call still
waiting its turn, and drops a running one's cancel file; an instance still
running it ``grace`` seconds later (and not saving) is stopped, as is one
running a call past the timeout (counted from when it took the call), and the
session's next call starts a fresh one. A call that waits its turn longer than
the timeout and grace together is withdrawn.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from pythontk.core_utils.cancel_scope import CancelScope, OperationCancelled
from pythontk.core_utils.operation_catalog.batch_operation_host import (
    BatchOperationHost,
    SceneFiles,
    _RunFolderInternal,
)
from pythontk.core_utils.operation_catalog.outcome import Issue, OpRequest, Outcome
from pythontk.file_utils.job_queue import JobQueue


class _SessionOperationHostInternal(_RunFolderInternal):
    """Internal helpers for :class:`SessionOperationHost`: the instance's lock
    and lifecycle, its serving loop and a caller's wait."""

    #: The run-folder file the instance writes a call's progress to.
    PROGRESS_FILE = "progress"
    #: The run-folder file naming the instance (its pid) that took the call.
    SERVED_FILE = "served_by"
    #: Held while one process starts the session's instance.
    LOCK_FILE = "launch.lock"
    #: Locked by the instance for its whole life: held means up.
    INSTANCE_LOCK = "instance.lock"
    #: Locked by a runner that took the session alone (:meth:`SessionOperationHost.claim`).
    CLAIM_FILE = "claim.lock"
    #: The key a call naming no scenes goes to (the last one named), in the
    #: session's folder: every process serving the session (the plan and apply
    #: servers) works in the same instance.
    CURRENT_FILE = "current"
    #: The instance's output: the first place to look when it fails to start.
    LOG_FILE = "instance.log"
    #: Seconds between the instance's looks for a call, and a caller's for its outcome.
    POLL = 0.02
    #: Seconds between the instance's heartbeats.
    BEAT = 0.5
    #: Seconds an instance may take to start serving.
    START_TIMEOUT = 300.0
    #: Seconds a launch lock may go untouched before another process takes it.
    LOCK_STALE = 10.0
    #: Seconds a starting instance waits for the one before it to let go of
    #: the instance lock (one ending to load changed code still releases its
    #: application first).
    LOCK_WAIT = 60.0
    #: Seconds an instance asked to stop has before it is killed.
    STOP_GRACE = 15.0
    #: Seconds between the instance's looks at the process that started it, and
    #: how long it keeps serving once that process is gone (its queue is stopped).
    OWNER_POLL = 2.0
    OWNER_GRACE = 30.0
    #: How much of the log's end a start failure is read from.
    TAIL_CHARS = 8000
    #: In the instance: its own Python's folders, whose modules are never
    #: watched (its install changes under no one's edit).
    _own: Sequence[str] = ()
    #: In the instance: the folders it reloads, each ``normcase``-d and
    #: ending in a separator; their modules are imported afresh per call.
    _reloaded: Sequence[str] = ()
    #: The folder in the session's that a kept instance's temp dir is: what it
    #: leaves there (a crash dump, a scratch file) goes with the session.
    TEMP_FOLDER = "temp"

    @staticmethod
    def _folder(name: str, session: str) -> Path:
        """The session's folder: one per host and session name, in the temp root."""
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{name}_{session}")
        if len(tag) > 80:  # cut, it keeps a hash of the whole: two keys never share one
            tag = tag[:70] + "-" + hashlib.sha1(tag.encode("utf-8")).hexdigest()[:9]
        return Path(TempArtifacts("ops_session", max_age_days=2).dir_path(name=tag))

    @staticmethod
    def _lock(path: Path) -> Optional[IO[bytes]]:
        """An exclusive lock on *path*, held until the handle closes or this
        process ends; ``None`` when another process holds it."""
        handle = open(path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return None
        return handle

    @classmethod
    def _held(cls, path: Path) -> bool:
        """Whether a process holds the lock on *path* (taken and let go at once to see)."""
        if not path.exists():
            return False
        handle = cls._lock(path)
        if handle is None:
            return True
        if os.name == "nt":  # at once: Windows may let a closed handle's lock go late
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()
        return False

    # -- in the instance ---------------------------------------------------------
    @classmethod
    def _progress(cls, run: Path) -> Callable[[float, str], None]:
        """A call's progress, written to its run folder for the caller to read."""
        path = run / cls.PROGRESS_FILE

        def report(fraction: float, message: str) -> None:
            try:
                cls._write(path, json.dumps([fraction, message]))
            except OSError:  # the caller held it open (Windows): the next one lands
                pass

        return report

    @classmethod
    def _loaded(cls, watched: Sequence[str]) -> Iterator[str]:
        """The file of each module this process loaded from under *watched*,
        but not from its own Python's install nor a folder it reloads."""
        roots = [root.rstrip("\\/") + os.sep for root in watched]
        for module in list(sys.modules.values()):
            try:
                path = vars(module).get("__file__")
            except TypeError:  # an object standing in for a module
                continue
            if not isinstance(path, str):
                continue
            key = cls._key(path)
            if any(key.startswith(root) for root in roots) and not any(
                key.startswith(skip) for skip in (*cls._own, *cls._reloaded)
            ):
                yield path

    @classmethod
    def _forget_reloaded(cls) -> None:
        """Drop every module loaded from a folder this instance reloads, so the
        call about to run imports each one as it is on disk now."""
        if not cls._reloaded:
            return
        for name, module in list(sys.modules.items()):
            try:
                path = vars(module).get("__file__")
            except TypeError:  # an object standing in for a module
                continue
            if isinstance(path, str) and any(
                cls._key(path).startswith(folder) for folder in cls._reloaded
            ):
                del sys.modules[name]
        importlib.invalidate_caches()

    @staticmethod
    def _mtime(path: str) -> Optional[float]:
        try:
            return os.stat(path).st_mtime
        except OSError:
            return None

    @classmethod
    def _stamp(cls, watched: Sequence[str], stamps: Dict[str, Optional[float]]) -> None:
        """Note each loaded module's file as it is now, the first time it is seen."""
        for path in cls._loaded(watched):
            if path not in stamps:
                stamps[path] = cls._mtime(path)

    @classmethod
    def _changed(
        cls, watched: Sequence[str], stamps: Dict[str, Optional[float]]
    ) -> Optional[str]:
        """The first loaded module whose file changed (or went) since it was
        noted, or ``None``; a module not noted yet is noted now."""
        for path in cls._loaded(watched):
            if path not in stamps:
                stamps[path] = cls._mtime(path)
            elif cls._mtime(path) != stamps[path]:
                return path
        return None

    @classmethod
    def _job(
        cls,
        job: Mapping[str, Any],
        files: SceneFiles,
        watched: List[str],
        stamps: Dict[str, Optional[float]],
        queue: JobQueue,
    ) -> Dict[str, Any]:
        """In the instance: answer one call, or end the instance when code it
        loaded changed (the caller then hands the call to a fresh one)."""
        for root in job.get("roots") or ():
            if root not in sys.path:
                sys.path.insert(0, root)
            if cls._key(root) not in watched:
                watched.append(cls._key(root))
        changed = cls._changed(watched, stamps)
        if changed:
            queue.stop()
            return {"stale": changed}
        cls._forget_reloaded()
        run = Path(job["run"])
        cls._write(run / cls.SERVED_FILE, str(os.getpid()))
        try:
            return cls._answer(run, files, cls._progress(run)).to_dict()
        finally:
            cls._stamp(watched, stamps)  # what this call imported, as it was loaded

    @classmethod
    def _watch_owner(cls, owner: int, queue: JobQueue) -> None:
        """End this instance once the process that started it is gone (should
        its lifetime binding not have taken): stop the queue, and leave if a
        call keeps it past the grace."""
        from pythontk.core_utils.execution_monitor._sidecar import process_alive

        def watch() -> None:
            while process_alive(owner):
                time.sleep(cls.OWNER_POLL)
            queue.stop()
            time.sleep(cls.OWNER_GRACE)
            os._exit(0)

        threading.Thread(target=watch, name="session-owner-watch", daemon=True).start()

    # -- the caller's side ------------------------------------------------------
    @contextlib.contextmanager
    def _launching(self) -> Iterator[Path]:
        """Hold the session's launch lock, so one process starts its instance.
        A lock its holder stopped touching (it died starting one) is taken."""
        self.folder.mkdir(parents=True, exist_ok=True)
        lock = self.folder / self.LOCK_FILE
        deadline = time.monotonic() + self.START_TIMEOUT + self.LOCK_STALE
        while True:
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                break
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > self.LOCK_STALE:
                        lock.unlink()
                        continue
                except OSError:
                    continue  # its holder released it meanwhile
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"another process held {lock} past the start timeout"
                )
            time.sleep(0.05)
        try:
            yield lock
        finally:
            try:
                lock.unlink()
            except OSError:
                pass

    def _worker(self) -> Dict[str, Any]:
        return self.queue.worker() or {}

    def _up(self) -> bool:
        """Whether an instance serves (or is starting to serve) the session:
        it holds the instance lock and has not stopped."""
        return (
            self._held(self.folder / self.INSTANCE_LOCK)
            and self._worker().get("state") != "stopped"
        )

    def _log_tail(self) -> str:
        """Why the instance failed, from the end of its output."""
        from pythontk.core_utils.handoff.script_run import ScriptRunner

        if self._drain is not None:
            self._drain.join(2.0)  # what it printed last is still on its way
        try:
            text = (self.folder / self.LOG_FILE).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return "it printed nothing"
        tail = text[-self.TAIL_CHARS :]
        said = self.host.diagnose(None, tail) if self.host.diagnose else None
        return said or ScriptRunner.failure_summary(tail) or "it printed no error"

    def _await_serving(
        self,
        on_progress: Optional[Callable[[float, str], None]],
        caller: Optional[CancelScope],
        proc: Optional[subprocess.Popen] = None,
    ) -> None:
        """Wait until the session's instance serves: *proc*, the one this
        process just started, or else another process's. Raise when it stops
        or never starts (its own failure named), or when *caller* cancels.

        Raises:
            OperationCancelled: *caller* was cancelled.
        """
        deadline, told = time.monotonic() + self.START_TIMEOUT, False
        lock = self.folder / self.LOCK_FILE
        while True:
            worker = self._worker()
            state = worker.get("state")
            if state == "serving" and self._up():
                return
            if state == "stopped" and worker.get("error"):
                raise RuntimeError(
                    f"it stopped while starting: {worker['error']} ({self._log_tail()})"
                )
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"it exited while starting (exit code {proc.returncode}): {self._log_tail()}"
                )
            if proc is None and not self._up():
                raise RuntimeError("the instance another process started is gone")
            if caller is not None and caller.poll():
                raise OperationCancelled(
                    caller.reason or "cancelled while the instance started"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"it did not start serving within {self.START_TIMEOUT:g} s"
                )
            if on_progress is not None and not told:
                on_progress(0.0, f"starting this session's {self.name} instance")
                told = True
            if proc is not None and lock.exists():
                os.utime(lock)  # still starting: the lock is held, not abandoned
                self._renew()  # a start longer than the lease must not outlive it
            time.sleep(0.05)

    def _spawn(
        self,
        on_progress: Optional[Callable[[float, str], None]],
        caller: Optional[CancelScope],
    ) -> None:
        """Start the session's instance and wait until it serves. The one
        before it is gone (its lock is free), so its record goes first: a
        record left behind never answers for the new instance."""

        exe = self.host._executable(self.key)
        if not exe:
            raise FileNotFoundError("no executable found")
        for name in (JobQueue.WORKER, JobQueue.STOP):
            try:
                (self.folder / name).unlink()
            except FileNotFoundError:
                pass
        work = self.folder / "work"  # what the application drops in its cwd stays here
        work.mkdir(parents=True, exist_ok=True)
        owner = None if self.keep is not None else os.getpid()
        if self.host.command is not None:  # it serves the session itself
            args, env = list(self.host.serve_args(self.folder, owner, self.key)), self.host.env
        else:
            roots = self.host._roots()
            call = (
                f"SessionOperationHost.serve({str(self.folder)!r}, {self.host.scene_files!r}, "
                f"roots, owner={owner}, reload={list(self.reload)!r})"
            )
            args, env = self.host._bootstrap(self.folder, call, roots)
        if self.keep is not None:
            self._renew()  # before it serves: a lease it finds lapsed is dropped
            proc = self._launch_kept(exe, args, work, env)
        else:
            proc = self._launch_bound(exe, args, work, env)
        self._proc = proc
        try:
            self._await_serving(on_progress, caller, proc)
        except BaseException:
            proc.kill()
            self._proc = None
            raise

    def _launch_bound(
        self, exe: str, args: Sequence[str], work: Path, env: Any
    ) -> subprocess.Popen:
        """The instance, bound to this process (it dies with it), its output
        drained into the session's log."""
        from pythontk.core_utils.app_launcher import AppLauncher

        log = open(self.folder / self.LOG_FILE, "wb")
        try:
            proc = AppLauncher.spawn(
                exe,
                args=args,
                cwd=str(work),
                env=env,
                hide_window=True,
                bind_lifetime=True,
            )
        except Exception:
            log.close()
            raise

        def drain() -> None:
            with log, proc.stdout:  # as it comes: a pipe's read waits for a full buffer
                for chunk in iter(lambda: proc.stdout.read1(4096), b""):
                    log.write(chunk)
                    log.flush()

        self._drain = threading.Thread(
            target=drain, name=f"{self.name}-session-log", daemon=True
        )
        self._drain.start()
        return proc

    def _launch_kept(
        self, exe: str, args: Sequence[str], work: Path, env: Any
    ) -> subprocess.Popen:
        """The instance, detached so it outlives this process (in a job that
        holds this one, it lasts as long as that job), its output straight into
        the session's log, and its temp dir one of the session's: a test
        sandbox's root goes with its run, and what a crash leaves stays out of
        the user's temp."""
        from pythontk.core_utils.app_launcher import AppLauncher

        temp = self.folder / self.TEMP_FOLDER
        temp.mkdir(parents=True, exist_ok=True)
        env = dict(env if env is not None else AppLauncher.process_environ())
        for name in ("TMPDIR", "TEMP", "TMP"):
            env[name] = str(temp)
        log = self.folder / self.LOG_FILE
        log.write_bytes(b"")  # this instance's output only
        proc = AppLauncher.launch(
            exe, args=list(args), cwd=str(work), detached=True, env=env, log=str(log)
        )
        if proc is None:
            raise RuntimeError(f"it could not be launched ({exe})")
        return proc

    def _renew(self) -> None:
        """Want a kept instance for :attr:`keep` seconds more (from now)."""
        if self.keep is not None:
            self.queue.lease(self.keep)

    def _ensure(
        self,
        on_progress: Optional[Callable[[float, str], None]] = None,
        caller: Optional[CancelScope] = None,
    ) -> None:
        """Make sure an instance serves the session: the one already up (this
        process's or another's), else a new one this process starts."""
        with self._guard:
            if self._up() and self._worker().get("state") == "serving":
                return
            with self._launching():
                if self._up():  # another process started one meanwhile
                    self._await_serving(on_progress, caller)
                    return
                self._spawn(on_progress, caller)

    def _kill(self) -> bool:
        """Stop the session's instance now, by the pid it recorded: only while
        it holds its lock is that pid its own. The process this one started
        (a launcher, perhaps) goes too."""
        from pythontk.core_utils.app_launcher import AppLauncher

        stopped = False
        worker = self._worker()
        pid = (
            worker.get("pid") if worker.get("state") != "stopped" else None
        )  # a stopped one's pid is gone
        if pid and self._held(self.folder / self.INSTANCE_LOCK):
            stopped = bool(AppLauncher.close_process(int(pid), force=True))
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            stopped = True
        deadline = time.monotonic() + 10.0
        while (
            stopped
            and self._held(self.folder / self.INSTANCE_LOCK)
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        return stopped and not self._held(self.folder / self.INSTANCE_LOCK)

    def _read_progress(
        self, run: Path, seen: List[Any], on_progress: Callable[[float, str], None]
    ) -> None:
        path = run / self.PROGRESS_FILE
        try:
            stamp = path.stat().st_mtime_ns
            if seen and seen[0] == stamp:
                return
            fraction, message = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        seen[:] = [stamp]
        on_progress(float(fraction), str(message))

    def _stopped(
        self,
        entry: Mapping[str, Any],
        request: OpRequest,
        why: str,
        cancelled: bool = False,
    ) -> Outcome:
        """The outcome of a call whose instance this runner stopped (or could not)."""
        if self._kill():
            return self._lost(
                entry,
                request,
                f"{self.name} session: {why}, so its instance was stopped "
                "(the next call starts a fresh one)",
                cancelled,
            )
        return self._lost(
            entry,
            request,
            f"{self.name} session: {why}, and its instance could not be stopped "
            "(it may still be running the call)",
            cancelled,
        )

    def _await(
        self,
        job: str,
        run: Path,
        entry: Mapping[str, Any],
        request: OpRequest,
        caller: Optional[CancelScope],
        on_progress: Optional[Callable[[float, str], None]],
    ) -> Union[Outcome, str]:
        """The call's outcome, or why the instance ended before it took the
        call (code it loaded changed): the caller then starts a fresh one."""
        submitted, running, asked, seen, missing = (
            time.monotonic(),
            None,
            None,
            [],
            None,
        )
        grace, timeout = self.host.grace, self.host.timeout
        renewed = time.monotonic()
        while True:
            if self.keep is not None and time.monotonic() - renewed > self.keep / 3:
                self._renew()  # held while the call runs: one longer than keep keeps it
                renewed = time.monotonic()
            state = self.queue.status(job)
            if (
                state is None
            ):  # between two of its records for a moment; gone for longer: lost
                missing = missing or time.monotonic()
                if time.monotonic() - missing > 1.0:
                    return self._lost(
                        entry,
                        request,
                        f"{self.name} session: the call's record vanished",
                    )
            else:
                missing = None
            if state in ("done", "failed"):
                try:
                    data = self.queue.result(job)
                except RuntimeError as exc:
                    return self._lost(entry, request, f"{self.name} session: {exc}")
                if "stale" in data:
                    return f"code it loaded changed: {data['stale']}"
                errors = Outcome.validate(data).errors
                if errors:
                    return self._lost(
                        entry,
                        request,
                        f"{self.name} session: an unreadable outcome: {'; '.join(errors)}",
                    )
                return Outcome.from_dict(data)
            if not self._up():
                if state != "queued":
                    cancelled = caller is not None and caller.cancelled
                    return self._lost(
                        entry,
                        request,
                        f"this session's {self.name} instance ended: {self._log_tail()}",
                        cancelled,
                    )
                try:  # the call waits for the session's next instance
                    self._ensure(on_progress, caller)
                except BaseException as exc:
                    self.queue.discard(job)
                    if isinstance(exc, OperationCancelled):
                        return Outcome(
                            op=request.op,
                            status="cancelled",
                            dry_run=request.dry_run,
                            error=str(exc),
                        )
                    if not isinstance(exc, Exception):
                        raise
                    return Outcome.refused(
                        request.op,
                        f"cannot start this session's {self.name} instance: {exc}",
                        request.dry_run,
                    )
                continue
            if on_progress is not None:
                self._read_progress(run, seen, on_progress)
            if caller is not None and asked is None and caller.poll():
                if (
                    state == "queued"
                ):  # not taken yet: withdraw it, never stop another's call
                    self.queue.discard(job)
                    if self.queue.status(job) is None:
                        return Outcome(
                            op=request.op,
                            status="cancelled",
                            dry_run=request.dry_run,
                            error="cancelled before it ran",
                        )
                (run / self.CANCEL_FILE).write_text(
                    caller.reason or "cancelled", encoding="utf-8"
                )
                asked = time.monotonic()
            if state == "running" and running is None:
                running = (
                    time.monotonic()
                )  # its timeout counts from here, not its wait in the queue
            if (
                state == "queued"
                and timeout is not None
                and time.monotonic() - submitted > timeout + (grace or 0)
            ):
                self.queue.discard(job)
                if self.queue.status(job) is None:
                    return Outcome.refused(
                        request.op,
                        f"{self.name} session: the instance stayed busy with another call for "
                        f"{time.monotonic() - submitted:.0f} s, so this one was withdrawn",
                        request.dry_run,
                    )
            late = (
                asked is not None
                and grace is not None
                and time.monotonic() - asked > grace
            )
            if late and state == "running" and not (run / self.SAVING_FILE).exists():
                return self._stopped(
                    entry,
                    request,
                    f"it did not stop within {grace:g} s of the cancel",
                    True,
                )
            if (
                timeout is not None
                and running is not None
                and time.monotonic() - running > timeout
            ):
                return self._stopped(
                    entry, request, f"it timed out after {timeout:g} s"
                )
            time.sleep(self.POLL)


class SessionOperationHost(_SessionOperationHostInternal):
    """Runs every call in one instance of an application kept for a session (see the module docstring).

    Parameters:
        host: The application's runner (its Python strategy): its executable,
            scene files, packages, environment, timeout and grace are this
            runner's too.
        session: The session's name: every runner given it for this host, in
            any process, shares one instance (an MCP client's session id).
        name: The host's name (``"maya"``): it names the session's folder and
            what the outcomes say.
        keep: Seconds the instance stays up after this runner's last call,
            outliving this process (a test runner's next run reuses it); then
            it ends by itself. ``None`` (an MCP server's): it lives as long as
            this process, and ends with it.
        reload: Folders whose modules the instance imports afresh for each
            call instead of restarting when they change (a test runner's
            package and tests).

    Raises:
        ValueError: *host* serves its run folders itself (``command``) and
            gives no way to serve a session (``serve``): it cannot be kept.
    """

    def __init__(
        self,
        host: BatchOperationHost,
        session: str,
        name: str = "host",
        keep: Optional[float] = None,
        reload: Sequence[Union[str, Path]] = (),
    ):
        if host.command is not None and host.serve_args is None:
            raise ValueError(
                f"the {name} runner serves its run folders itself (command) and has no "
                "way to serve a session (serve), so it is launched per call and cannot be kept"
            )
        self.host = host
        self.session = str(session)
        self.name = name
        self.keep = None if keep is None else float(keep)
        self.reload = tuple(os.path.abspath(str(folder)) for folder in reload)
        self.folder = self._folder(name, self.session)
        self.queue = JobQueue(str(self.folder))
        self._guard = threading.Lock()
        #: The instance this process started (``None`` when another's serves).
        self._proc: Optional[subprocess.Popen] = None
        self._drain: Optional[threading.Thread] = None
        #: The instance (its pid) that served this runner's last call.
        self._last: Optional[str] = None
        #: This runner's hold on the session (:meth:`claim`), or ``None``.
        self._claim: Optional[IO[bytes]] = None
        #: The instance this runner keeps, by its host's key (``instance``):
        #: ``None`` is the runner's default.
        self.key: Optional[str] = None
        #: The runners of this session's other instances, by key.
        self._instances: Dict[str, "SessionOperationHost"] = {}

    def run(
        self,
        entry: Mapping[str, Any],
        request: OpRequest,
        scope: Optional[CancelScope] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> Outcome:
        """Run *request* in the session's instance (started first when none
        serves) and hand back its outcome; never raises (the outcome says why).
        With a host that keys its instances (``instance``), the one its scenes
        name, else the last one named."""
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        target = self._instance_for(request)
        if target is not self:
            return target.run(entry, request, scope, on_progress)

        caller = scope if scope is not None else CancelScope.current()
        restarted = None
        for _ in range(2):  # again only in the fresh instance a code change started
            try:
                self._renew()
                self._ensure(on_progress, caller)
            except OperationCancelled as exc:
                return Outcome(
                    op=request.op,
                    status="cancelled",
                    dry_run=request.dry_run,
                    error=str(exc),
                )
            except Exception as exc:
                return Outcome.refused(
                    request.op,
                    f"cannot start this session's {self.name} instance: {exc}",
                    request.dry_run,
                )
            store = TempArtifacts("ops_run", policy="scoped")
            run = Path(store.dir_path())
            served = None
            try:
                self._write(
                    run / OpRequest.REQUEST_FILE,
                    json.dumps(request.to_dict(), indent=2),
                )
                self._write(run / self.ENTRY_FILE, json.dumps(dict(entry), indent=2))
                job = self.queue.submit(
                    {
                        "run": str(run),
                        "roots": self.host._roots(self.host._provider(entry)),
                    }
                )
                try:
                    outcome = self._await(job, run, entry, request, caller, on_progress)
                finally:
                    self.queue.discard(job)
                if (run / self.SERVED_FILE).is_file():
                    served = (run / self.SERVED_FILE).read_text(encoding="utf-8")
            except Exception as exc:
                outcome = self._lost(
                    entry, request, f"{self.name} session: {type(exc).__name__}: {exc}"
                )
            store.cleanup()
            self._renew()  # idle from the end of this call
            if isinstance(outcome, str):
                restarted = outcome
                continue
            if served is not None:
                if (
                    restarted is None
                    and self._last is not None
                    and served != self._last
                ):
                    restarted = "another instance than this session's last call's served this one"
                self._last = served
            if restarted is not None:
                outcome.issues.append(
                    Issue(
                        code="session.restarted",
                        message=(
                            f"this session's {self.name} instance restarted ({restarted}); what "
                            "it held before (an open scene, console names) is gone"
                        ),
                        severity="warning",
                    )
                )
            return outcome
        return Outcome.refused(
            request.op,
            f"this session's {self.name} instance kept restarting",
            request.dry_run,
        )

    def _instance_for(self, request: OpRequest) -> "SessionOperationHost":
        """The runner of the instance *request* needs, made once per key: the
        one its scenes name, else the one the session last worked in, else
        the host's default (``instance([])``; ``None``: this runner)."""
        if self.host.instance is None or self.key is not None:
            return self
        try:
            current = (self.folder / self.CURRENT_FILE).read_text(encoding="utf-8") or None
        except OSError:
            current = None
        if request.scenes or current is None:
            try:
                key = self.host.instance(list(request.scenes))
            except Exception:  # the call's own run says what is wrong with its scenes
                key = None
        else:
            key = current  # no scenes: what it holds
        if key is None:
            key = current
        if key is None:
            return self
        if key != current:
            self._write(self.folder / self.CURRENT_FILE, key)
        with self._guard:
            runner = self._instances.get(key)
            if runner is None:
                tag = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
                runner = SessionOperationHost(
                    self.host, f"{self.session}~{tag}", self.name, self.keep, self.reload
                )
                runner.key = key
                self._instances[key] = runner
            return runner

    def claim(self) -> bool:
        """Take this session's instance for this runner alone, until :meth:`close`
        or this process ends: ``False`` while another process holds it. A
        runner that resets what the instance holds (a test run) claims it
        first, so no other run is using it; the servers of one MCP client
        share theirs and claim nothing.

        Returns:
            Whether this runner holds the session now (asking again keeps it).
        """
        if self._claim is None:
            self._claim = self._lock(self.folder / self.CLAIM_FILE)
        return self._claim is not None

    def close(self) -> None:
        """End the instance this process started, letting it release the
        application first; another process's instance is that process's to end,
        and a kept one (:attr:`keep`) its lease's. A claim (:meth:`claim`) is let go."""
        for runner in list(self._instances.values()):
            runner.close()
        claim, self._claim = self._claim, None
        if claim is not None:
            if os.name == "nt":  # at once: Windows may let a closed handle's lock go late
                import msvcrt

                claim.seek(0)
                msvcrt.locking(claim.fileno(), msvcrt.LK_UNLCK, 1)
            claim.close()
        if self.keep is not None:
            return
        with self._guard:
            proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        self.queue.stop()
        try:
            proc.wait(self.STOP_GRACE)
        except subprocess.TimeoutExpired:
            self._proc = proc
            self._kill()
            self._proc = None

    @classmethod
    def serve(
        cls,
        folder: Union[str, Path],
        scene_files: str = "pythontk:PureSceneFiles",
        roots: Sequence[str] = (),
        owner: Optional[int] = None,
        reload: Sequence[str] = (),
    ) -> int:
        """In the instance: serve the session's calls until asked to stop (or
        its lease lapses), or until code it loaded from under *roots* changes
        on disk.

        Parameters:
            folder: The session's folder (its queue).
            scene_files: The :class:`SceneFiles` class, ``"module:qualified.name"``,
                built once: it starts the application's Python API.
            roots: The import roots whose modules must stay current, beside
                every folder on its path outside its own Python's install.
            owner: The pid of the process that started it: it ends once that is gone.
            reload: Folders whose modules it imports afresh for each call,
                never restarting for them.

        Returns:
            The exit code the instance ends with: 0 when it served until told to end.
        """
        folder = Path(folder)
        deadline = time.monotonic() + cls.LOCK_WAIT
        lock = cls._lock(folder / cls.INSTANCE_LOCK)
        while (
            lock is None and time.monotonic() < deadline
        ):  # the one before it is still letting go
            time.sleep(0.05)
            lock = cls._lock(folder / cls.INSTANCE_LOCK)
        if lock is None:
            print("another instance serves this session", file=sys.stderr, flush=True)
            return 1
        from pythontk.core_utils.handoff.app_handoff import HandoffBridge

        queue = JobQueue(str(folder))
        handed = dict.fromkeys([*roots, *HandoffBridge.child_sys_path()])
        watched = [cls._key(root) for root in handed]
        _SessionOperationHostInternal._own = HandoffBridge.own_python_folders()
        _SessionOperationHostInternal._reloaded = [
            cls._key(folder).rstrip("\\/") + os.sep for folder in reload
        ]
        stamps: Dict[str, Optional[float]] = {}
        held: Dict[str, Any] = {}

        def setup() -> None:
            files, failure = cls._begin(scene_files)
            if failure is not None:
                raise failure
            held["files"] = files
            cls._stamp(watched, stamps)

        if owner:
            cls._watch_owner(int(owner), queue)
        try:
            queue.serve(
                lambda job: cls._job(job, held["files"], watched, stamps, queue),
                setup=setup,
                poll=cls.POLL,
                beat=cls.BEAT,
            )
        except Exception as exc:  # it could not start: the heartbeat says why too
            print(
                f"the session could not start: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        finally:
            cls._finish(held.get("files"))
            lock.close()
        return 0
