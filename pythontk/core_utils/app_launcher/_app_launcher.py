# !/usr/bin/python
# coding=utf-8
"""The :class:`AppLauncher` facade: the three launch shapes, over its job mixins.

``launch`` (detached: it outlives this process), ``run`` (blocking, with a
timeout and optional line streaming) and ``spawn`` (a helper that runs beside
this process and dies with it), plus ``write_batch_script``. Install discovery,
the process environment, desktop sessions and windows, and the process table
are private mixins in sibling modules; every member resolves on this class, and
every cross-call is spelled ``AppLauncher.<name>`` so patching the facade
reaches it.
"""

import os
import subprocess
import platform
import logging

from pythontk.core_utils.app_launcher._desktop import _DesktopMixin
from pythontk.core_utils.app_launcher._discovery import _DiscoveryMixin
from pythontk.core_utils.app_launcher._environment import _EnvironmentMixin
from pythontk.core_utils.app_launcher._processes import _ProcessesMixin

logger = logging.getLogger(__name__)


class _AppLauncherInternal(object):
    """Internal helpers for AppLauncher: the launch internals, and a file read
    the discovery and environment mixins share."""

    #: Seconds to keep draining a child's pipe after the child itself has exited.
    #: A grandchild that inherited the handle can hold the pipe open forever, so
    #: end-of-file alone is not a safe exit condition.
    _EXIT_DRAIN_GRACE = 2.0

    @staticmethod
    def _read_lines(path):
        """A text file's lines without line ends; empty when it is unreadable."""
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read().splitlines()
        except OSError:
            return []

    @staticmethod
    def _command(executable_path, args) -> list:
        """``[executable, *args]``; *args* may be one string or a sequence."""
        cmd = [executable_path]
        if args:
            if isinstance(args, str):
                cmd.append(args)
            elif isinstance(args, (list, tuple)):
                cmd.extend(args)
        return cmd

    @staticmethod
    def _kill(proc) -> None:
        """Kill *proc* and reap it; a process already gone is not an error."""
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning(f"Process {proc.pid} did not exit after kill.")

    @staticmethod
    def _run_streaming(cmd, cwd, timeout, env, creationflags, on_output, poll_interval):
        """The :meth:`AppLauncher.run` body when *on_output* is given (see there)."""
        import queue
        import time

        from pythontk.core_utils.cancel_scope import CancelScope, OperationCancelled
        from pythontk.core_utils.process_stream import OutputStream, ProcessReader

        # Merged into ONE pipe so lines keep their relative order: a traceback on
        # stderr must land after the progress line that preceded it.
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            creationflags=creationflags,
        )
        stream = OutputStream()
        lines: "queue.Queue[str]" = queue.Queue()
        unsubscribe = stream.subscribe(lambda _source, line: lines.put(line))
        reader = ProcessReader(proc.stdout, stream, "stdout")
        reader.start()
        output = []
        deadline = None if timeout is None else time.monotonic() + timeout
        exited_at = None
        try:
            while True:
                try:
                    line = lines.get(timeout=poll_interval)
                    output.append(line)
                except queue.Empty:
                    line = None
                    if proc.poll() is not None:
                        exited_at = exited_at or time.monotonic()
                        if not reader.is_alive() or (
                            time.monotonic() - exited_at
                            > _AppLauncherInternal._EXIT_DRAIN_GRACE
                        ):
                            break
                if on_output(line) is False or not CancelScope.proceed():
                    _AppLauncherInternal._kill(proc)
                    raise OperationCancelled(
                        f"'{os.path.basename(cmd[0])}' cancelled by the caller"
                    )
                if deadline is not None and time.monotonic() > deadline:
                    _AppLauncherInternal._kill(proc)
                    raise subprocess.TimeoutExpired(
                        cmd, timeout, output="\n".join(output)
                    )
        finally:
            unsubscribe()
            stream.close()
        text = "\n".join(output)
        return subprocess.CompletedProcess(
            cmd, proc.wait(), stdout=text + "\n" if output else "", stderr=""
        )


class AppLauncher(
    _AppLauncherInternal,
    _DiscoveryMixin,
    _EnvironmentMixin,
    _DesktopMixin,
    _ProcessesMixin,
):
    """
    A utility class for launching applications on Windows and Linux.
    """

    @staticmethod
    def launch(app_identifier, args=None, cwd=None, detached=True, env=None):
        """
        Launches an application.

        :param app_identifier: The name or path of the application (e.g., 'notepad', 'firefox', 'C:/Apps/MyTool.exe').
        :param args: A list of arguments to pass to the application.
        :param cwd: The working directory for the application.
        :param detached: If True, launches as a separate process (application keeps running if script ends).
        :param env: Optional mapping of environment variables for the child process.
                    If None, the child inherits the current process's environment.
        :return: The subprocess.Popen object or None if launch failed.
        """
        system = platform.system().lower()
        executable_path = AppLauncher.find_app(app_identifier)

        if not executable_path:
            logger.warning(f"Application '{app_identifier}' not found.")
            return None

        cmd = AppLauncher._command(executable_path, args)

        try:
            logger.debug(f"Launching: {cmd} (Detached: {detached})")

            kwargs = {"cwd": cwd, "shell": False}

            if env is not None:
                kwargs["env"] = env

            if detached:
                if system == "windows":
                    # DETACHED_PROCESS to allow the script to exit while app keeps running
                    kwargs["creationflags"] = (
                        subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                else:
                    # Linux/Unix
                    kwargs["start_new_session"] = True

            return subprocess.Popen(cmd, **kwargs)

        except Exception as e:
            logger.error(f"Failed to launch '{app_identifier}': {e}")
            return None

    @staticmethod
    def run(
        app_identifier,
        args=None,
        cwd=None,
        timeout=None,
        output_file=None,
        env=None,
        hide_window=False,
        on_output=None,
        poll_interval=0.1,
    ):
        """Execute an application synchronously and return its result.

        Unlike :meth:`launch` (fire-and-forget), this method blocks until the
        process finishes and honours a *timeout*.

        :param app_identifier: Name or path of the application.
        :param args: Arguments to pass (str, list, or tuple).
        :param cwd: Working directory for the process.
        :param timeout: Maximum seconds to wait before raising
                        ``subprocess.TimeoutExpired``.  *None* = no limit.
        :param output_file: If given, stdout+stderr are redirected to this file
                        (combined) instead of captured in memory — use for
                        long-running tools whose logs are large (the returned
                        ``stdout``/``stderr`` are then ``None``). Otherwise output
                        is captured and returned as decoded text.
        :param env: Optional environment mapping for the child (else inherits).
        :param hide_window: Windows only — suppress the console window Windows
                        would otherwise pop for a console-subsystem child when
                        the parent is a GUI app (e.g. mayapy run from a DCC).
                        No effect on the captured/redirected output.
        :param on_output: Stream the output instead of collecting it at exit:
                        called on THIS thread with each line of the child's
                        merged stdout+stderr as it arrives (newline stripped),
                        and with ``None`` whenever *poll_interval* seconds pass
                        without one, so a caller driving a UI keeps it alive
                        through a long silent step. Return ``False`` to stop the
                        run: the child is killed and
                        :class:`pythontk.OperationCancelled` raised. An ambient
                        :class:`pythontk.CancelScope` is polled at the same
                        points. The returned ``stdout`` still carries the whole
                        output; ``stderr`` is ``""`` (merged). A child only
                        streams what it flushes: a Python child that prints
                        progress should pass ``flush=True``. Not combinable with
                        *output_file*.
        :param poll_interval: Seconds between idle ``on_output(None)`` calls.
        :return: A ``subprocess.CompletedProcess`` with *returncode* and, unless
                 *output_file* is set, *stdout*/*stderr* (decoded text).
        :raises FileNotFoundError: If the application cannot be found.
        :raises subprocess.TimeoutExpired: If *timeout* is exceeded.
        :raises pythontk.OperationCancelled: When *on_output* returns ``False``.
        """
        if on_output is not None and output_file:
            raise ValueError("on_output and output_file are mutually exclusive.")
        executable_path = AppLauncher.find_app(app_identifier)
        if not executable_path:
            raise FileNotFoundError(f"Application '{app_identifier}' not found.")

        cmd = AppLauncher._command(executable_path, args)

        creationflags = (
            subprocess.CREATE_NO_WINDOW if hide_window and os.name == "nt" else 0
        )
        logger.debug(f"Running (blocking): {cmd}")
        if on_output is not None:
            return AppLauncher._run_streaming(
                cmd, cwd, timeout, env, creationflags, on_output, poll_interval
            )
        if output_file:
            with open(output_file, "w", encoding="utf-8", errors="replace") as fh:
                return subprocess.run(
                    cmd,
                    cwd=cwd,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                    shell=False,
                    env=env,
                    creationflags=creationflags,
                )
        # errors="replace": the child writes in ITS encoding (mayapy: cp1252), which
        # need not be ours (Blender decodes UTF-8). A strict decode died in the
        # reader thread and returned no output at all -- tracebacks included.
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            shell=False,
            env=env,
            creationflags=creationflags,
        )

    @staticmethod
    def spawn(
        app_identifier,
        args=None,
        cwd=None,
        env=None,
        hide_window=True,
        bind_lifetime=True,
    ):
        """Start a helper process that runs beside this one and dies with it.

        The third shape beside :meth:`launch` (detached -- it outlives this
        process by design) and :meth:`run` (blocks until the child exits): the
        child keeps running while the caller reads its output, and it must not
        outlive the caller. A tunnel, a watcher, a local service a feature
        fronts -- anything that, orphaned by a crashed host, would go on
        serving (or exposing) something nobody is looking after any more.

        Parameters:
            app_identifier: Name or path of the executable.
            args: Arguments (a string, list or tuple).
            cwd: Working directory for the child.
            env: Environment mapping for the child; ``None`` inherits.
            hide_window: Windows -- no console window for a console child of a
                GUI host (a DCC).
            bind_lifetime: Tie the child to this process, so it is killed
                however this process ends, crash included (Windows: a
                kill-on-close Job Object; POSIX: the child leads its own
                process group and a ``sh`` watcher kills that group within
                about a second of this process going away). The caller's own
                stop and ``atexit`` still cover a normal exit.

        Returns:
            The ``subprocess.Popen``, with ``bound_to_parent`` set to whether
            the lifetime binding took. Its ``stdout`` is a BINARY pipe carrying
            stdout and stderr merged in order: pair it with
            :class:`pythontk.ProcessReader` and :class:`pythontk.OutputStream`,
            and keep it drained -- a child writing into a full pipe blocks.

        Raises:
            FileNotFoundError: The application cannot be found.
        """
        executable_path = AppLauncher.find_app(app_identifier)
        if not executable_path:
            raise FileNotFoundError(f"Application '{app_identifier}' not found.")
        cmd = AppLauncher._command(executable_path, args)
        creationflags = (
            subprocess.CREATE_NO_WINDOW if hide_window and os.name == "nt" else 0
        )
        logger.debug(f"Spawning: {cmd}")
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            creationflags=creationflags,
            # POSIX: its own process group, so the lifetime watcher can take
            # down whatever the child starts too.
            start_new_session=bool(bind_lifetime) and os.name != "nt",
        )
        proc.bound_to_parent = bool(bind_lifetime) and AppLauncher._bind_lifetime(
            proc.pid
        )
        return proc

    @staticmethod
    def write_batch_script(path, lines, shell=None):
        """Write a script for a shell to run: the ONE place its bytes are decided.

        Two shells, two byte contracts, and a hand-written file got both wrong:

        * ``"cmd"`` (a ``.bat``): cmd reads the file in the console **OEM**
          codepage, so a UTF-8 file turns a non-ASCII path into mojibake. Lines
          end in exact CRLF -- ``"\\r\\n".join`` through a text-mode handle
          doubles every CR on Windows. No ``errors="replace"``: a path cmd
          cannot spell raises ``UnicodeEncodeError`` rather than being rewritten
          to ``?`` and sending output (or a completion marker) to a wrong file.
        * ``"bash"``: UTF-8, LF, a ``#!/usr/bin/env bash`` shebang unless the
          first line already is one, and the executable bit.

        The caller writes the lines in the chosen shell's own syntax; line ends
        inside an entry are normalized, so a pre-joined block is fine.

        Parameters:
            path: Destination file (overwritten).
            lines: The script's lines without line ends, or one pre-joined
                block as a single string.
            shell: ``"cmd"`` or ``"bash"``; ``None`` picks the platform's own
                (``cmd`` on Windows, ``bash`` elsewhere).

        Returns:
            *path*.

        Raises:
            ValueError: An unknown *shell*.
            UnicodeEncodeError: A cmd script holds text the OEM codepage lacks.
        """
        shell = shell or ("cmd" if os.name == "nt" else "bash")
        if shell not in ("cmd", "bash"):
            raise ValueError(f"Unknown shell {shell!r}: expected 'cmd' or 'bash'.")
        if isinstance(lines, str):  # one block, not one character per line
            lines = [lines]
        split = [
            part
            for line in lines
            for part in str(line).replace("\r\n", "\n").replace("\r", "\n").split("\n")
        ]
        if shell == "cmd":
            # The OEM codec exists only on Windows; elsewhere a .bat is written
            # for another machine to run, and UTF-8 is the only honest default.
            encoding = "oem" if os.name == "nt" else "utf-8"
            data = ("\r\n".join(split) + "\r\n").encode(encoding)
        else:
            if not (split and split[0].startswith("#!")):
                split.insert(0, "#!/usr/bin/env bash")
            data = ("\n".join(split) + "\n").encode("utf-8")
        with open(path, "wb") as fh:
            fh.write(data)
        if shell == "bash" and os.name != "nt":
            os.chmod(path, os.stat(path).st_mode | 0o111)
        return path
