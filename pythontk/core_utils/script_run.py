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


def _tail(text: str) -> str:
    return text[-_TAIL_CHARS:] if len(text) > _TAIL_CHARS else text


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
) -> ScriptRunResult:
    """Run *script_text* in *app_exe*, wait, and return the verified *artifact*.

    Parameters:
        app_exe: Executable to run (name or path; resolved by ``AppLauncher``).
        script_text: The (already rendered) script body to execute.
        artifact: Path the script is expected to produce. Existence + non-zero
            size after the run is the success criterion.
        launch_args: Maps the written script's path to the app's argv (default
            ``[script_path]`` — interpreter style, e.g. ``mayapy script.py``).
        timeout: Max seconds before the child is killed (``subprocess.TimeoutExpired``
            propagates with ``script_path`` attached, script kept). ``None`` = no limit.
        script_suffix / script_prefix: Naming for the temp script file.
        cwd / env: Forwarded to the child process.

    Returns:
        ScriptRunResult: on success (the temp script is removed).

    Raises:
        RuntimeError: when the artifact is missing or empty — the message embeds
            the exit code and output tail, and the exception carries
            ``script_path`` (the script is kept for debugging).
        FileNotFoundError / subprocess.TimeoutExpired: from the launch itself.
    """
    from pythontk.file_utils.temp_artifacts import TempArtifacts

    # A leftover artifact from a prior run would fake success — the existence
    # check below must judge THIS run's output only.
    if os.path.exists(artifact):
        os.remove(artifact)

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

    if not (os.path.isfile(artifact) and os.path.getsize(artifact) > 0):
        error = RuntimeError(
            f"{os.path.basename(str(app_exe))} did not produce the expected artifact "
            f"{artifact} (exit code {proc.returncode}, {duration:.1f}s). "
            f"Script kept at {script_path}. Output tail:\n{_tail(output)}"
        )
        error.script_path = script_path  # kept — scoped cleanup is skipped on failure
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


__all__ = ["ScriptRunResult", "run_script_to_artifact"]
