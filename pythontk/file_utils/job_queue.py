# !/usr/bin/python
# coding=utf-8
"""A folder of JSON jobs that one worker process takes in order.

For work that has to run inside another process -- a DCC opened once on a
scene that takes minutes to load -- without a network port anything else
on the machine could answer or be answered by. The caller submits jobs and
waits; the worker serves them. Both sides only read and write files in the
queue's folder.

Layout: ``<id>.job.json`` (queued) -> ``<id>.running.json`` -> ``<id>.done.json``
or ``<id>.failed.json``; ``worker.json``, the worker's heartbeat; ``stop``,
a request to end after the current job; ``lease.json``, how long a caller
still wants the worker (:meth:`JobQueue.lease`).
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import traceback
from typing import Any, Callable, Dict, Mapping, Optional

from pythontk.file_utils._file_utils import FileUtils


class JobQueue:
    """Submit jobs to a worker through a folder, and serve them on the worker's side.

    Parameters:
        root: The queue's folder (made on first use).

    Example:
        >>> queue = JobQueue(folder)
        >>> job = queue.submit({"op": "bake", "layers": ["key"]})
        >>> queue.wait(job, timeout=3600)        # the caller
        >>> queue.serve(handle_job, setup=open_scene)   # the worker process
    """

    WORKER = "worker.json"
    STOP = "stop"
    LEASE = "lease.json"
    _STATES = ("queued", "running", "done", "failed")
    _SUFFIX = {
        "queued": ".job.json",
        "running": ".running.json",
        "done": ".done.json",
        "failed": ".failed.json",
    }

    def __init__(self, root: str):
        self.root = str(root)
        self._count = 0

    # ------------------------------------------------------------------ files
    def _path(self, job_id: str, state: str) -> str:
        return os.path.join(self.root, job_id + self._SUFFIX[state])

    def _read(self, path: str) -> Optional[Dict[str, Any]]:
        # A file being replaced can refuse a reader for a moment on Windows.
        for attempt in range(5):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                return None
            except (OSError, ValueError):
                time.sleep(0.01 * (attempt + 1))
        return None

    def _write(self, path: str, data: Mapping[str, Any]) -> None:
        FileUtils.write_json(path, data, indent=1)

    # ------------------------------------------------------------------ the caller
    def submit(self, job: Mapping[str, Any]) -> str:
        """Queue *job* (JSON-able); returns its id. Ids sort in submission order."""
        self._count += 1  # padded: a coarse clock gives a burst one time
        job_id = f"{time.time_ns():020d}-{os.getpid()}-{self._count:06d}"
        self._write(self._path(job_id, "queued"), dict(job))
        return job_id

    def status(self, job_id: str) -> Optional[str]:
        """``"queued"``, ``"running"``, ``"done"``, ``"failed"``, or ``None`` for an unknown id."""
        for state in reversed(self._STATES):  # the latest a race could show both
            if os.path.exists(self._path(job_id, state)):
                return state
        return None

    def result(self, job_id: str) -> Dict[str, Any]:
        """A finished job's result.

        Raises:
            RuntimeError: the job failed (the worker's error and traceback).
            LookupError: the job has not finished, or is unknown.
        """
        done = self._read(self._path(job_id, "done"))
        if done is not None:
            return done.get("result") or {}
        failed = self._read(self._path(job_id, "failed"))
        if failed is not None:
            raise RuntimeError(f"Job {job_id} failed: {failed.get('error')}")
        raise LookupError(f"Job {job_id} is {self.status(job_id) or 'unknown'}.")

    def wait(
        self,
        job_id: str,
        timeout: Optional[float] = None,
        poll: float = 1.0,
        stale: float = 120.0,
    ) -> Dict[str, Any]:
        """Wait for a job's result (see :meth:`result`).

        Parameters:
            timeout: Seconds to wait at most (``None``: as long as the worker beats).
            poll: Seconds between looks.
            stale: A worker whose heartbeat is older than this has died, and a
                queue with no live worker for this long has none coming: the
                wait fails then instead of running out its timeout.

        Raises:
            TimeoutError: *timeout* passed.
            RuntimeError: the job failed, the worker stopped beating while
                running it, or no worker served the queue for *stale* seconds.
        """
        end = None if timeout is None else time.time() + timeout
        idle_since = None
        while True:
            try:
                return self.result(job_id)
            except LookupError:
                pass
            worker = self.worker()
            if self.alive(stale):
                idle_since = None
            elif (
                worker
                and worker.get("state") != "stopped"
                and self.status(job_id) == "running"
            ):  # it died with this job; one still queued waits for the next worker
                raise RuntimeError(
                    f"The worker (pid {worker.get('pid')}) stopped beating with job "
                    f"{job_id} {self.status(job_id)}."
                )
            else:
                idle_since = idle_since or time.time()
                if time.time() - idle_since > stale:
                    raise RuntimeError(
                        f"Job {job_id} is {self.status(job_id)} and no worker is serving "
                        f"{self.root} (launch one)."
                    )
            if end is not None and time.time() >= end:
                raise TimeoutError(f"Job {job_id} still {self.status(job_id)}.")
            time.sleep(poll)

    def worker(self) -> Optional[Dict[str, Any]]:
        """The worker's heartbeat: ``pid``, ``host``, ``state`` (starting, serving,
        stopped), ``job`` (the one running), ``started``, ``beat`` (epoch seconds)."""
        return self._read(os.path.join(self.root, self.WORKER))

    def alive(self, stale: float = 120.0) -> bool:
        """Whether a worker is up: it beat within *stale* seconds and has not stopped."""
        worker = self.worker()
        return bool(
            worker
            and worker.get("state") != "stopped"
            and time.time() - float(worker.get("beat", 0)) < stale
        )

    def lease(self, seconds: Optional[float]) -> None:
        """Want the worker for *seconds* more; a caller renews this while it does.

        Once the lease lapses with nothing queued the worker ends by itself:
        an idle worker goes after the last renewal, and so does one whose
        caller died. A queued job always outlives it. ``None`` withdraws the
        lease -- the worker serves until :meth:`stop`, as a queue never leased
        always does.
        """
        path = os.path.join(self.root, self.LEASE)
        if seconds is None:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            return
        self._write(
            path,
            {
                "until": time.time() + float(seconds),
                "pid": os.getpid(),
                "host": socket.gethostname(),
            },
        )

    def leased(self) -> Optional[float]:
        """Seconds the lease has left (negative once lapsed), or ``None`` with none."""
        lease = self._read(os.path.join(self.root, self.LEASE))
        return None if lease is None else float(lease.get("until", 0)) - time.time()

    def stop(self) -> None:
        """Ask the worker to end once its current job is done."""
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, self.STOP), "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))

    # ------------------------------------------------------------------ the worker
    def serve(
        self,
        handler: Callable[[Dict[str, Any]], Mapping[str, Any]],
        setup: Optional[Callable[[], None]] = None,
        poll: float = 1.0,
        beat: float = 5.0,
    ) -> int:
        """Take jobs in order and answer each with ``handler(job)``, until :meth:`stop`.

        A job that raises is recorded failed (with its traceback) and the next
        one runs. A job left running by a worker that died fails at start.

        Parameters:
            handler: ``job -> result`` (JSON-able).
            setup: Run first, while the heartbeat already says ``starting``
                (opening a scene can take minutes).
            poll: Seconds between looks for a job.
            beat: Seconds between heartbeats (written from a thread, so a long
                job keeps beating).

        Returns:
            How many jobs ran.

        Raises:
            RuntimeError: another worker is serving this queue.
        """
        self.begin(setup, beat)
        ran = 0
        try:
            while not self.stopping():
                if self.step(handler) is None:
                    time.sleep(poll)
                else:
                    ran += 1
        finally:
            self.end()
        return ran

    def begin(
        self, setup: Optional[Callable[[], None]] = None, beat: float = 5.0
    ) -> None:
        """Become this queue's worker, for a host that runs its own loop (:meth:`step`).

        The heartbeat starts (``starting``), a job a dead worker left running
        fails, *setup* runs, and the heartbeat says ``serving``. Pair with :meth:`end`.

        Raises:
            RuntimeError: another worker is serving this queue.
        """
        if self.alive(stale=max(3 * beat, 1.0)):
            raise RuntimeError(f"A worker is serving {self.root}: {self.worker()}")
        os.makedirs(self.root, exist_ok=True)
        try:
            os.remove(os.path.join(self.root, self.STOP))
        except FileNotFoundError:
            pass
        left = self.leased()
        if left is not None and left <= 0:  # a caller gone before this worker came up
            self.lease(None)
        for name in sorted(os.listdir(self.root)):  # a dead worker's job
            if name.endswith(self._SUFFIX["running"]):
                job_id = name[: -len(self._SUFFIX["running"])]
                self._write(
                    self._path(job_id, "failed"),
                    {"error": "the worker running it died before it finished"},
                )
                os.remove(os.path.join(self.root, name))
        self._state = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "state": "starting",
            "job": None,
            "started": time.time(),
        }
        self._lock = threading.Lock()
        self._done = threading.Event()

        def beating() -> None:
            while not self._done.wait(beat):
                self._beat()

        self._beat()
        self._thread = threading.Thread(target=beating, daemon=True)
        self._thread.start()
        if setup is not None:
            try:
                setup()
            except BaseException as error:  # no worker: say so, and why
                self._done.set()
                self._thread.join()
                self._beat(state="stopped", error=repr(error))
                raise
        self._beat(state="serving")

    def _beat(self, **fields: Any) -> None:
        with self._lock:
            self._state.update(fields)
            self._state["beat"] = time.time()
            try:
                self._write(os.path.join(self.root, self.WORKER), self._state)
            except OSError:  # a reader held it open (Windows): the next beat lands
                pass

    def stopping(self) -> bool:
        """Whether :meth:`stop` has been asked, or the :meth:`lease` lapsed with
        nothing queued."""
        if os.path.exists(os.path.join(self.root, self.STOP)):
            return True
        left = self.leased()
        return left is not None and left <= 0 and self._next() is None

    def step(
        self, handler: Callable[[Dict[str, Any]], Mapping[str, Any]]
    ) -> Optional[str]:
        """Run the oldest queued job (between :meth:`begin` and :meth:`end`); its id,
        or ``None`` when nothing is queued. A job that raises is recorded failed."""
        job_id = self._next()
        if job_id is None:
            return None
        running = self._path(job_id, "running")
        try:  # waits out a sync client or a scanner holding the file (Windows)
            FileUtils.replace_file(self._path(job_id, "queued"), running)
        except FileNotFoundError:  # taken or withdrawn meanwhile
            return None
        self._beat(job=job_id)
        started = time.time()
        try:
            result = handler(self._read(running) or {})
            self._write(
                self._path(job_id, "done"),
                {
                    "result": dict(result or {}),
                    "seconds": round(time.time() - started, 3),
                },
            )
        except Exception:
            self._write(self._path(job_id, "failed"), {"error": traceback.format_exc()})
        os.remove(running)
        self._beat(job=None)
        return job_id

    def end(self) -> None:
        """Stop being the worker: the heartbeat says ``stopped`` and a stop request is spent."""
        self._done.set()
        self._thread.join()
        self._beat(state="stopped", job=None)
        try:
            os.remove(os.path.join(self.root, self.STOP))
        except FileNotFoundError:
            pass

    def _next(self) -> Optional[str]:
        """The oldest queued job's id."""
        try:
            names = sorted(os.listdir(self.root))
        except FileNotFoundError:
            return None
        suffix = self._SUFFIX["queued"]
        for name in names:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return None
