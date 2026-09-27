# !/usr/bin/python
# coding=utf-8
"""Run a script in an external app, block until it exits, and collect an artifact.

The synchronous counterpart of :class:`pythontk.ScriptLaunchDeliverer` (which renders
a script and launches a *detached* app). Here the caller needs the result: write the
script, run the app attached via :meth:`pythontk.AppLauncher.run`, and judge success
by the **artifact** — the exit code is advisory only, because DCC standalone
interpreters (mayapy et al.) are known to crash in teardown *after* the real work
succeeded. Qt-free and DCC-free; the app-specific knowledge (which exe, which
template) stays with the caller.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from pythontk.core_utils.app_launcher import AppLauncher
from pythontk.core_utils.cancel_scope import OperationCancelled

logger = logging.getLogger(__name__)

# Output tail size embedded in failure messages — enough for a DCC traceback,
# small enough to keep exceptions readable.
_TAIL_CHARS = 4000


# How the run is judged. ``CREATED``: the artifact must not survive from a prior run
# (it is cleared first) and must exist non-empty after. ``REWRITTEN``: the artifact IS
# the input — the app loads it, edits it, and saves over it — so clearing it would
# destroy the very thing the app was asked to read; success is a *change* to its
# (mtime, size) instead. An app that exits 0 without ever reaching its save call
# leaves both untouched, which is the failure this distinction exists to catch.
CREATED = "created"
REWRITTEN = "rewritten"


class _ScriptRunnerInternal(object):
    """Internal helpers for ScriptRunner."""

    @staticmethod
    def _tail(text: str) -> str:
        return text[-_TAIL_CHARS:] if len(text) > _TAIL_CHARS else text

    @staticmethod
    def _stamp(path: str) -> tuple:
        """``(mtime, size)`` for *path*, or ``(0, 0)`` when it doesn't exist."""
        try:
            st = os.stat(path)
        except OSError:
            return (0, 0)
        return (st.st_mtime, st.st_size)


class ScriptRunner(_ScriptRunnerInternal):
    """Run a script in an external app, block, and collect its artifact.

    The synchronous counterpart to :class:`~pythontk.ScriptLaunchDeliverer`,
    which renders a script and launches a *detached* app: here the caller
    needs the result, so the app is waited on and its output returned.
    """

    @staticmethod
    def run_script_to_artifact(
        app_exe: str,
        script_text: str,
        *,
        artifact: str,
        launch_args: Optional[Callable[[str], Sequence[str]]] = None,
        timeout: Optional[float] = 600,
        script_suffix: str = ".py",
        script_prefix: str = "script_run",
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        expect: str = CREATED,
        on_output: Optional[Callable[[Optional[str]], Optional[bool]]] = None,
    ) -> ScriptRunResult:
        """Run *script_text* in *app_exe*, wait, and return the verified *artifact*.

        Parameters:
            app_exe: Executable to run (name or path; resolved by ``AppLauncher``).
            script_text: The (already rendered) script body to execute.
            artifact: Path the script is expected to produce (:data:`CREATED`) or to
                edit in place (:data:`REWRITTEN`).
            expect: :data:`CREATED` (default) — the artifact is this run's output, so a
                stale one is cleared first and success is "exists, non-empty".
                :data:`REWRITTEN` — the artifact is also the *input*, so it is left
                alone and success is "still exists, non-empty, and its (mtime, size)
                moved". Round-trip hand-offs (export → app edits the file → re-import)
                need the second: clearing would delete the app's input, and an app that
                exits cleanly without saving would otherwise read as success and the
                caller would silently re-ingest its own unmodified export.
            launch_args: Maps the written script's path to the app's argv (default
                ``[script_path]`` — interpreter style, e.g. ``mayapy script.py``).
            timeout: Max seconds before the child is killed (``subprocess.TimeoutExpired``
                propagates with ``script_path`` attached, script kept). ``None`` = no limit.
            script_suffix / script_prefix: Naming for the temp script file.
            cwd / env: Forwarded to the child process.
            on_output: Stream the child's output while it runs instead of only
                collecting it at exit (:meth:`pythontk.AppLauncher.run`): called with
                each line as it arrives and with ``None`` on quiet ticks; return
                ``False`` to stop the run. :meth:`ProgressRelay.reader` builds one that
                turns a template's ``::progress::`` markers into a progress bar. A
                stopped run removes its script and, under :data:`CREATED`, any partial
                artifact -- the caller asked for it to end, so nothing is kept for
                debugging.

        Returns:
            ScriptRunResult: on success (the temp script is removed).

        Raises:
            RuntimeError: when the artifact is missing, empty, or (under
                :data:`REWRITTEN`) unchanged — the message embeds the exit code and
                output tail, and the exception carries ``script_path`` (the script is
                kept for debugging).
            FileNotFoundError / subprocess.TimeoutExpired: from the launch itself.
            pythontk.OperationCancelled: when *on_output* returned ``False`` (or an
                ambient :class:`pythontk.CancelScope` was cancelled).
        """
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        if expect not in (CREATED, REWRITTEN):
            raise ValueError(
                f"expect must be {CREATED!r} or {REWRITTEN!r}, got {expect!r}"
            )

        if expect == CREATED:
            # A leftover artifact from a prior run would fake success — the existence
            # check below must judge THIS run's output only.
            if os.path.exists(artifact):
                os.remove(artifact)
            before = (0, 0)
        else:
            # The artifact is the app's INPUT. Snapshot it instead of clearing it.
            before = _ScriptRunnerInternal._stamp(artifact)

        tmp = TempArtifacts(script_prefix, policy="scoped")
        script_path = tmp.path(extension=script_suffix)
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(script_text)

        args = list(launch_args(script_path)) if launch_args else [script_path]
        start = time.time()
        # hide_window: the child's output is captured — a console window (which
        # Windows would otherwise pop for a console-subsystem exe like mayapy when
        # the parent is a GUI app) serves nothing.
        try:
            proc = AppLauncher.run(
                app_exe,
                args=args,
                cwd=cwd,
                timeout=timeout,
                env=env,
                hide_window=True,
                on_output=on_output,
            )
        except subprocess.TimeoutExpired as error:
            # Same debuggability contract as the missing-artifact RuntimeError: the
            # script is kept, and the exception says where.
            error.script_path = script_path
            raise
        except OperationCancelled:
            tmp.cleanup()
            if expect == CREATED:
                try:
                    os.remove(artifact)
                except OSError:
                    pass
            raise
        duration = time.time() - start
        output = (proc.stdout or "") + (proc.stderr or "")

        failure = None
        if not (os.path.isfile(artifact) and os.path.getsize(artifact) > 0):
            failure = (
                f"did not produce the expected artifact {artifact}"
                if expect == CREATED
                else f"left no usable file at {artifact}"
            )
        elif expect == REWRITTEN and _ScriptRunnerInternal._stamp(artifact) == before:
            # Exited cleanly and never saved. Treating this as success would hand the
            # caller back its own untouched export as though the app had edited it.
            failure = f"exited without writing {artifact} (unchanged on disk)"

        if failure:
            error = RuntimeError(
                f"{os.path.basename(str(app_exe))} {failure} "
                f"(exit code {proc.returncode}, {duration:.1f}s). "
                f"Script kept at {script_path}. Output tail:\n{_ScriptRunnerInternal._tail(output)}"
            )
            error.script_path = (
                script_path  # kept — scoped cleanup is skipped on failure
            )
            error.output = output
            error.returncode = proc.returncode
            raise error

        if proc.returncode != 0:
            logger.warning(
                f"Artifact produced but exit code was {proc.returncode} "
                "(tolerated: DCC teardown crashes are known)."
            )
        tmp.cleanup()
        return ScriptRunResult(
            artifact=artifact,
            returncode=proc.returncode,
            output=output,
            duration=duration,
            script_path=script_path,
        )


@dataclass
class ScriptRunResult:
    """What a successful :func:`run_script_to_artifact` returns.

    *output* is the combined stdout+stderr text (DCC warnings are diagnostic gold);
    *returncode* is advisory (see module docstring); *script_path* is where the
    rendered script was written (already removed on success, kept on failure).
    """

    artifact: str
    returncode: int
    output: str
    duration: float
    script_path: str


class ProgressRelay:
    """One progress callback fed by staged work: child-script markers and in-process steps.

    A blocking multi-stage run (convert in one app, bake in another, finish here) has
    several producers of progress and one bar. Each stage owns an equal slice of
    ``[0, total]`` and reports ``step`` of ``steps`` inside it, either in-process
    (:meth:`report`) or from a child script printing marker lines (:meth:`line`), read
    back through :meth:`reader` -- the ``on_output`` for
    :meth:`ScriptRunner.run_script_to_artifact`.

    *progress* has the ecosystem's ``progress(current, total, message) -> bool`` shape,
    the one uitk's ``Switchboard.progress_adapter`` adapts a footer bar to; ``current``
    is ``None`` on a keep-alive tick, which only pumps the receiver. A callback that
    returns ``False`` cancels: :meth:`report`, :meth:`tick` and the reader return
    ``False``, and the runner stops the child.

    The marker is one plain line, so a template that imports nothing can print it::

        print("::progress:: 3/7 Baking animation", flush=True)

    Parameters:
        progress: The receiver, or ``None`` (every report then just continues).
        stages: How many stages share the bar.
        total: The bar's maximum.
        throttle: Minimum seconds between keep-alive ticks. Markers always pass.
    """

    PREFIX = "::progress::"

    def __init__(
        self,
        progress: Optional[Callable[..., Optional[bool]]] = None,
        stages: int = 1,
        total: int = 100,
        throttle: float = 0.1,
    ):
        self.progress = progress
        self.stages = max(1, int(stages))
        self.total = int(total)
        self.throttle = float(throttle)
        self._value = 0
        self._last_emit = None

    @classmethod
    def line(cls, step: int, steps: int, text: str = "") -> str:
        """The marker line a child prints for *step* of *steps*."""
        return f"{cls.PREFIX} {int(step)}/{int(steps)} {text}".rstrip()

    @classmethod
    def parse(cls, line: Optional[str]) -> Optional[Tuple[int, int, str]]:
        """``(step, steps, text)`` from a marker anywhere in *line*, else ``None``."""
        index = line.find(cls.PREFIX) if line else -1
        if index < 0:
            return None
        head, _, text = line[index + len(cls.PREFIX) :].strip().partition(" ")
        step, _, steps = head.partition("/")
        try:
            return int(step), int(steps), text.strip()
        except ValueError:
            return None

    @property
    def value(self) -> int:
        """The bar position last reported."""
        return self._value

    def report(
        self, stage: int, step: float, steps: float, text: Optional[str] = None
    ) -> bool:
        """Report *step* of *steps* inside *stage* (0-based); ``False`` = cancel."""
        stage = min(max(int(stage), 0), self.stages - 1)
        fraction = min(max(float(step) / steps, 0.0), 1.0) if steps else 0.0
        value = int(round((stage + fraction) * self.total / self.stages))
        # Monotonic: a stage re-reporting an earlier step never moves the bar back.
        self._value = max(self._value, value)
        return self._emit(self._value, text)

    def tick(self) -> bool:
        """Keep the receiver alive between reports (throttled); ``False`` = cancel."""
        now = time.monotonic()
        if self._last_emit is not None and now - self._last_emit < self.throttle:
            return True
        return self._emit(None, None)

    def reader(self, stage: int, label: str = "") -> Callable[[Optional[str]], bool]:
        """An ``on_output`` for :meth:`ScriptRunner.run_script_to_artifact` scoped to
        *stage*: markers report (their text prefixed with *label*), every other line
        and every quiet poll ticks."""

        def on_output(line: Optional[str]) -> bool:
            marker = self.parse(line)
            if marker is None:
                return self.tick()
            step, steps, text = marker
            message = ": ".join(part for part in (label, text) if part) or None
            return self.report(stage, step, steps, message)

        return on_output

    def _emit(self, value: Optional[int], text: Optional[str]) -> bool:
        self._last_emit = time.monotonic()
        if self.progress is None:
            return True
        return self.progress(value, self.total, text) is not False


__all__ = ["ScriptRunner", "ScriptRunResult", "ProgressRelay", "CREATED", "REWRITTEN"]
