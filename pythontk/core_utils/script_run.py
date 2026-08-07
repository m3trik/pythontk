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
from typing import Callable, Optional, Sequence

from pythontk.core_utils.app_launcher import AppLauncher

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
    """ScriptRunner — module namespace."""

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

        Returns:
            ScriptRunResult: on success (the temp script is removed).

        Raises:
            RuntimeError: when the artifact is missing, empty, or (under
                :data:`REWRITTEN`) unchanged — the message embeds the exit code and
                output tail, and the exception carries ``script_path`` (the script is
                kept for debugging).
            FileNotFoundError / subprocess.TimeoutExpired: from the launch itself.
        """
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        if expect not in (CREATED, REWRITTEN):
            raise ValueError(f"expect must be {CREATED!r} or {REWRITTEN!r}, got {expect!r}")

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
                app_exe, args=args, cwd=cwd, timeout=timeout, env=env, hide_window=True
            )
        except subprocess.TimeoutExpired as error:
            # Same debuggability contract as the missing-artifact RuntimeError: the
            # script is kept, and the exception says where.
            error.script_path = script_path
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


__all__ = ["ScriptRunner", "ScriptRunResult", "CREATED", "REWRITTEN"]
