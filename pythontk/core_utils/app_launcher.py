# !/usr/bin/python
# coding=utf-8
import os
import glob
import subprocess
import shutil
import platform
import logging
import threading

logger = logging.getLogger(__name__)


class _AppLauncherInternal(object):
    """Internal helpers for AppLauncher."""

    #: Seconds to keep draining a child's pipe after the child itself has exited.
    #: A grandchild that inherited the handle can hold the pipe open forever, so
    #: end-of-file alone is not a safe exit condition.
    _EXIT_DRAIN_GRACE = 2.0

    #: The kill-on-close Job Object every :meth:`AppLauncher.spawn` child joins
    #: on Windows. Created on first use and deliberately never closed: its
    #: handle closes when THIS process ends -- a crash or a kill included -- and
    #: closing the last handle to such a job terminates everything in it. That
    #: is the whole mechanism; there is no watchdog or reaper to keep running.
    _lifetime_job = None
    _lifetime_lock = threading.Lock()

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
    def _bind_lifetime(pid: int) -> bool:
        """Make process *pid* die with this one; False where that cannot be arranged.

        Windows only (see :attr:`_lifetime_job`). Elsewhere there is no
        portable equivalent -- ``PR_SET_PDEATHSIG`` is Linux-only and needs a
        ``preexec_fn``, which is unsafe in a threaded host -- so a caller keeps
        its own stop and ``atexit`` for a normal exit, and a crash can orphan
        the child.
        """
        if os.name != "nt":
            return False
        import ctypes
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class _BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimits),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        extended_limit_information = 9  # JobObjectExtendedLimitInformation
        kill_on_job_close = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        set_quota_and_terminate = 0x0100 | 0x0001  # PROCESS_SET_QUOTA | _TERMINATE

        # A private handle on kernel32, so the argtypes declared here cannot
        # leak into another caller's use of the shared `ctypes.windll.kernel32`.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.POINTER(_ExtendedLimits),
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

        with _AppLauncherInternal._lifetime_lock:
            job = _AppLauncherInternal._lifetime_job
            if job is None:
                job = kernel32.CreateJobObjectW(None, None)
                if not job:
                    logger.debug(f"CreateJobObject failed: {ctypes.get_last_error()}")
                    return False
                limits = _ExtendedLimits()
                limits.BasicLimitInformation.LimitFlags = kill_on_job_close
                if not kernel32.SetInformationJobObject(
                    job,
                    extended_limit_information,
                    ctypes.byref(limits),
                    ctypes.sizeof(limits),
                ):
                    logger.debug(
                        f"SetInformationJobObject failed: {ctypes.get_last_error()}"
                    )
                    kernel32.CloseHandle(job)
                    return False
                _AppLauncherInternal._lifetime_job = job
        process = kernel32.OpenProcess(set_quota_and_terminate, False, pid)
        if not process:
            return False  # already gone, or not ours to manage
        try:
            if kernel32.AssignProcessToJobObject(job, process):
                return True
            logger.debug(
                f"AssignProcessToJobObject({pid}) failed: {ctypes.get_last_error()}"
            )
            return False
        finally:
            kernel32.CloseHandle(process)

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


class AppLauncher(_AppLauncherInternal):
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
    def process_environ():
        """The LIVE process environment -- what a child would actually inherit.

        ``os.environ`` is a snapshot taken at interpreter init (plus Python-side
        changes). A HOST app that embeds Python and sets a variable at the C
        level afterwards is invisible to the snapshot, yet every child process
        inherits the real value: Blender 5.x sets ``OCIO`` to its bundled
        v2.5 config during color-management init, so ``os.environ`` showed no
        ``OCIO`` while a child ``cmd /c echo %%OCIO%%`` printed Blender's config
        (measured live -- the mechanism behind Maya's color-management init
        failing on every bridge send). Windows: decode the live block via
        ``GetEnvironmentStringsW``. Elsewhere ``os.environ`` is returned as-is
        (the same C-level blindness exists, but no supported host currently
        exercises it off-Windows).
        """
        if platform.system().lower() != "windows":
            return dict(os.environ)
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetEnvironmentStringsW.restype = ctypes.c_void_p
        ptr = kernel32.GetEnvironmentStringsW()
        if not ptr:
            return dict(os.environ)
        try:
            env = {}
            offset = ptr
            while True:
                entry = ctypes.wstring_at(offset)
                if not entry:
                    break
                # Advance by the UTF-16 byte length + terminator, not len(entry):
                # a non-BMP character is ONE Python char but TWO UTF-16 code
                # units, and undercounting desyncs the rest of the block walk.
                offset += len(entry.encode("utf-16-le")) + 2
                key, sep, value = entry.partition("=")
                # Skip the hidden per-drive CWD entries ("=C:=C:\\..."), whose
                # key is empty -- children rebuild those themselves.
                if key and sep:
                    env[key] = value
            return env
        finally:
            kernel32.FreeEnvironmentStringsW(ctypes.c_void_p(ptr))

    @staticmethod
    def handoff_env(source_root):
        """Child env for launching a DIFFERENT app: this process's env, minus its
        app-private ``OCIO`` config.

        A DCC hand-off launches app B from inside app A, so B inherits A's whole
        environment -- including an ``OCIO`` var pointing INSIDE A's own install
        tree (e.g. Blender's bundled ``ocio_profile_version: 2.5`` config), which
        B's OpenColorIO runtime may be unable to load: Maya 2025 (OCIO 2.3) then
        fails color-management init on every bridge launch. A config *outside*
        the source install (a studio ACES pipeline config) is deliberate
        cross-app state and passes through untouched.

        Reads (and, when stripping, copies) the LIVE process environment via
        :meth:`process_environ`, never ``os.environ`` -- Blender sets ``OCIO``
        at the C level AFTER Python init, so the snapshot can neither see nor
        strip the very variable this hook exists for.

        :param source_root: The RUNNING app's installation root. ``OCIO`` is
                            stripped only when its path lies under this tree.
        :return: A copied env dict without ``OCIO``, or ``None`` (= inherit
                 unchanged) when there is nothing to strip.
        """
        from pathlib import Path

        environ = AppLauncher.process_environ()
        ocio = environ.get("OCIO")
        if not (ocio and source_root):
            return None
        try:
            private = Path(ocio).resolve().is_relative_to(Path(source_root).resolve())
        except (OSError, ValueError):
            return None
        if not private:
            return None
        env = dict(environ)
        del env["OCIO"]
        logger.info(
            f"Launch env: dropped OCIO ({ocio}) -- private to the running app's "
            f"install ({source_root}); the target app gets its own default."
        )
        return env

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
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
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
                kill-on-close Job Object). Off Windows there is no portable
                equivalent; the caller's own stop and ``atexit`` cover a
                normal exit.

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
        )
        proc.bound_to_parent = bool(bind_lifetime) and AppLauncher._bind_lifetime(
            proc.pid
        )
        return proc

    # ------------------------------------------------------------------ sessions
    @staticmethod
    def current_session_id():
        """Windows session id of the *current* process.

        Session 0 is the non-interactive *services* session — no window station,
        no display — so GUI / window-station-bound apps launched there often fail
        or hang (e.g. an SSH or scheduled-task context). A non-zero id is an
        interactive desktop session. Returns ``None`` off Windows or on failure.
        """
        if platform.system().lower() != "windows":
            return None
        try:
            import ctypes

            pid = ctypes.windll.kernel32.GetCurrentProcessId()
            sid = ctypes.c_ulong()
            if ctypes.windll.kernel32.ProcessIdToSessionId(pid, ctypes.byref(sid)):
                return int(sid.value)
        except Exception as e:
            logger.debug(f"current_session_id failed: {e}")
        return None

    @staticmethod
    def active_console_session_id():
        """Session id of the physically logged-in console (interactive desktop).

        This is the session a GUI app must run in to have a window station and be
        visible. Returns ``None`` off Windows, or when no user is logged on
        (``WTSGetActiveConsoleSessionId`` → ``0xFFFFFFFF``).
        """
        if platform.system().lower() != "windows":
            return None
        try:
            import ctypes

            # ctypes' default c_int restype surfaces the DWORD sentinel
            # 0xFFFFFFFF as -1; mask back to unsigned so the check fires.
            sid = ctypes.windll.kernel32.WTSGetActiveConsoleSessionId() & 0xFFFFFFFF
            if sid == 0xFFFFFFFF:
                return None
            return int(sid)
        except Exception as e:
            logger.debug(f"active_console_session_id failed: {e}")
        return None

    @staticmethod
    def is_interactive_session():
        """True if the current process is in an interactive session (non-zero —
        has a window station / can show GUIs). False for the services session 0
        (e.g. a non-interactive SSH context on Windows). ``None``-safe: a missing
        session id is treated as non-interactive only on Windows; off Windows
        (where the concept doesn't apply) it returns True."""
        if platform.system().lower() != "windows":
            return True
        sid = AppLauncher.current_session_id()
        return sid is not None and sid != 0

    @staticmethod
    def find_session_launcher(explicit=None):
        """Locate a helper able to launch a process into *another* interactive
        session — Sysinternals PsExec. pythontk does not bundle it; this only
        discovers it so :meth:`launch_in_session` can orchestrate it when present,
        keeping PsExec a runtime-optional tool rather than a dependency.

        Search order: *explicit* → ``$PSEXEC`` env → PATH (``PsExec64.exe`` /
        ``PsExec.exe``) → ``$PSTOOLS_HOME`` → conventional tool dirs. Returns the
        path or ``None``. Site-specific locations belong in ``$PSEXEC`` (full
        path) or ``$PSTOOLS_HOME`` (containing directory) rather than here.
        """
        candidates = []
        if explicit:
            candidates.append(explicit)
        if os.environ.get("PSEXEC"):
            candidates.append(os.environ["PSEXEC"])
        for name in ("PsExec64.exe", "PsExec.exe", "psexec64", "psexec"):
            w = shutil.which(name)
            if w:
                candidates.append(w)
        program_files = os.environ.get("ProgramFiles", "")
        for d in (
            os.environ.get("PSTOOLS_HOME", ""),
            os.path.join(program_files, "PSTools") if program_files else "",
            program_files,
            r"C:\tools",
            r"M:\tools",  # legacy site location; last resort
        ):
            if d:
                candidates += [
                    os.path.join(d, n) for n in ("PsExec64.exe", "PsExec.exe")
                ]
        for c in candidates:
            if c and os.path.isfile(c):
                return c
        return None

    @staticmethod
    def launch_in_session(
        app_identifier,
        args=None,
        session=None,
        cwd=None,
        launcher=None,
        accept_eula=True,
    ):
        """Launch an application into a specific interactive Windows session.

        Needed when the caller runs in the non-interactive services session 0
        (e.g. over SSH) yet the target is a GUI / window-station-bound app that
        must run on the logged-in desktop. Delegates to a session launcher
        (PsExec ``-i <session> -d``). If the caller is *already* in the target
        session, this skips PsExec and launches normally.

        :param session: Target session id. ``None`` → the active console session.
        :param launcher: Explicit PsExec path (else discovered, see
                         :meth:`find_session_launcher`).
        :return: ``subprocess.CompletedProcess`` of the launch (returncode 0 =
                 started). The target runs detached in the other session, so
                 monitor it out-of-band (process name / output files), not via
                 this return value.
        :raises RuntimeError: off Windows, when no interactive session exists, or
                              when no session launcher is found.
        """
        if platform.system().lower() != "windows":
            raise RuntimeError("launch_in_session is Windows-only.")
        target = (
            session if session is not None else AppLauncher.active_console_session_id()
        )
        if target is None:
            raise RuntimeError("No interactive console session to launch into.")
        if AppLauncher.current_session_id() == target:
            proc = AppLauncher.launch(app_identifier, args=args, cwd=cwd, detached=True)
            return subprocess.CompletedProcess(
                args=[app_identifier], returncode=(0 if proc is not None else 1)
            )
        psexec = AppLauncher.find_session_launcher(launcher)
        if not psexec:
            raise RuntimeError(
                "No session launcher (PsExec) found; set $PSEXEC or pass launcher=."
            )
        exe = AppLauncher.find_app(app_identifier) or app_identifier
        cmd = [psexec]
        if accept_eula:
            cmd.append("-accepteula")
        cmd += ["-i", str(target), "-d"]
        if cwd:
            cmd += ["-w", cwd]
        cmd.append(exe)
        if args:
            cmd += list(args) if isinstance(args, (list, tuple)) else [args]
        logger.debug(f"launch_in_session: {cmd}")
        return subprocess.run(cmd, capture_output=True, text=True)

    @staticmethod
    def wait_for_ready(process, timeout=15, check_fn=None):
        """
        Waits until the application is ready.

        :param process: The subprocess.Popen object returned by launch().
        :param timeout: Maximum seconds to wait.
        :param check_fn: Optional callable that returns True when ready. Signature: check_fn(process) -> bool.
                         If None, defaults to checking for a visible window owned by process.pid.
        :return: True if ready, False if timeout or process exited.
        """
        import time

        if not process:
            return False

        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if process died prematurely
            if process.poll() is not None:
                return False

            if check_fn:
                try:
                    if check_fn(process):
                        return True
                except Exception:
                    # Ignore errors in the callback (e.g. connection refused) and keep retrying
                    pass
            else:
                if AppLauncher._has_window(process.pid):
                    return True

            time.sleep(0.5)

        return False

    @staticmethod
    def _has_window(pid):
        """
        Checks if the given PID owns a visible window.
        """
        system = platform.system().lower()

        if system == "windows":
            import ctypes

            # Callback function type for EnumWindows
            WNDENUMPROC = ctypes.WINFUNCTYPE(
                ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p
            )
            user32 = ctypes.windll.user32

            found_windows = []

            def enum_windows_callback(hwnd, _):
                # Get the Process ID for this window handle
                lpdw_process_id = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(lpdw_process_id))

                if lpdw_process_id.value == pid:
                    # Check if window is visible (ignore background/hidden helper windows)
                    if user32.IsWindowVisible(hwnd):
                        # Ensure the window has a title (filters out unnamed tooltips/overlays)
                        if user32.GetWindowTextLengthW(hwnd) > 0:
                            found_windows.append(hwnd)
                            return False  # Stop enumeration
                return True  # Continue enumeration

            # Trigger the enumeration
            user32.EnumWindows(WNDENUMPROC(enum_windows_callback), 0)

            return len(found_windows) > 0

        elif system == "linux":
            # Basic fallback for Linux
            return True

        return True

    @staticmethod
    def get_window_titles(pid):
        """
        Returns a list of window titles owned by the given PID.
        """
        system = platform.system().lower()
        titles = []

        if system == "windows":
            import ctypes

            WNDENUMPROC = ctypes.WINFUNCTYPE(
                ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p
            )
            user32 = ctypes.windll.user32

            def enum_windows_callback(hwnd, _):
                lpdw_process_id = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(lpdw_process_id))

                if lpdw_process_id.value == pid:
                    if user32.IsWindowVisible(hwnd):
                        length = user32.GetWindowTextLengthW(hwnd)
                        if length > 0:
                            buff = ctypes.create_unicode_buffer(length + 1)
                            user32.GetWindowTextW(hwnd, buff, length + 1)
                            titles.append(buff.value)
                return True

            user32.EnumWindows(WNDENUMPROC(enum_windows_callback), 0)

        return titles

    @staticmethod
    def append_to_path(path, user_scope=True):
        """
        Appends a directory to the system PATH.

        :param path: The directory path to append.
        :param user_scope: If True, updates User environment variables (persistent).
                           If False, only updates current process environment (temporary).
        :return: True if successful.
        """
        if not path or not os.path.isdir(path):
            return False

        # Always update current process
        current_path = os.environ.get("PATH", "")
        if path.lower() not in current_path.lower().split(os.pathsep):
            os.environ["PATH"] = f"{current_path}{os.pathsep}{path}"

        if not user_scope:
            return True

        if platform.system().lower() == "windows":
            import winreg

            try:
                key_path = r"Environment"
                root_key = winreg.HKEY_CURRENT_USER

                with winreg.OpenKey(
                    root_key, key_path, 0, winreg.KEY_ALL_ACCESS
                ) as key:
                    try:
                        old_path, _ = winreg.QueryValueEx(key, "Path")
                    except OSError:
                        old_path = ""

                    if path.lower() in old_path.lower().split(os.pathsep):
                        return True  # Already there

                    new_path = f"{old_path}{os.pathsep}{path}" if old_path else path
                    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)

                    return True
            except Exception as e:
                logger.error(f"Failed to update registry PATH: {e}")
                return False

        # TODO: Linux implementation (e.g. .bashrc)
        return False

    @staticmethod
    def scan_for_executables(root_paths, executable_name, depth=3):
        """
        Scans directories for a specific executable.

        :param root_paths: List of root directories to search.
        :param executable_name: Name of the executable (e.g. 'maya.exe').
        :param depth: Max folder depth to search.
        :return: List of absolute paths found.
        """
        found = []
        if isinstance(root_paths, str):
            root_paths = [root_paths]

        for root in root_paths:
            if not os.path.exists(root):
                continue

            for dirpath, dirnames, filenames in os.walk(root):
                # Calculate current depth relative to root
                try:
                    rel_path = os.path.relpath(dirpath, root)
                    current_depth = (
                        0 if rel_path == "." else len(rel_path.split(os.sep))
                    )

                    if current_depth >= depth:
                        # Don't descend further, clear dirs to stop walk
                        dirnames[:] = []
                        continue
                except ValueError:
                    continue

                for f in filenames:
                    if f.lower() == executable_name.lower():
                        found.append(os.path.join(dirpath, f))

        return sorted(found, reverse=True)  # Sort typically gives newer versions

    @staticmethod
    def is_path_persisted(path):
        """
        Checks if the path is permanently stored in the system configuration (e.g. Windows Registry).
        useful to avoid prompting the user repeatedly if they haven't restarted their shell.
        """
        if not path:
            return False

        path_norm = os.path.normpath(path).lower()

        if platform.system().lower() == "windows":
            import winreg

            try:
                # Check User Environment
                key_path = r"Environment"
                try:
                    with winreg.OpenKey(
                        winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ
                    ) as key:
                        val, _ = winreg.QueryValueEx(key, "Path")
                        if path_norm in [
                            os.path.normpath(p).lower()
                            for p in val.split(os.pathsep)
                            if p
                        ]:
                            return True
                except OSError:
                    pass

                # Check System Environment (ReadOnly usually)
                key_path_lm = (
                    r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
                )
                try:
                    with winreg.OpenKey(
                        winreg.HKEY_LOCAL_MACHINE, key_path_lm, 0, winreg.KEY_READ
                    ) as key:
                        val, _ = winreg.QueryValueEx(key, "Path")
                        if path_norm in [
                            os.path.normpath(p).lower()
                            for p in val.split(os.pathsep)
                            if p
                        ]:
                            return True
                except OSError:
                    pass
            except Exception:
                pass

        return False

    @staticmethod
    def _program_files_roots():
        """Return the 64-bit and 32-bit ``Program Files`` roots from the environment."""
        return (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        )

    @staticmethod
    def scan_install_dirs(scan_globs):
        """Yield existing files matching *scan_globs*; pattern order is a priority.

        Every match for the first pattern is yielded (newest install dir
        first, via reverse sort) before any match for the second pattern,
        and so on — mirroring how :meth:`resolve_app_path` ranks its stages
        (first hit wins). The previous pooled sort made the *filename* an
        accidental tiebreaker: RizomUV's ``rizomuv_RS.exe`` outranked the
        intended ``Rizomuv_VS.exe`` inside the same install dir purely by
        ASCII order, inverting the caller's tuple priority.

        A ``{program_files}`` token in a pattern expands to both Program Files roots,
        so callers write one portable pattern instead of hardcoding the 64/32-bit
        dirs::

            AppLauncher.scan_install_dirs([r"{program_files}\\Autodesk\\Maya*\\bin\\maya.exe"])

        :param scan_globs: An ordered iterable of glob patterns (each may use the token).
        :return: A generator of absolute file paths — highest-priority pattern
                 first, newest first within each pattern, duplicates skipped.
        """
        pf64, pf32 = AppLauncher._program_files_roots()
        seen = set()
        for pattern in scan_globs:
            if "{program_files}" in pattern:
                expanded = [
                    pattern.format(program_files=pf64),
                    pattern.format(program_files=pf32),
                ]
            else:
                expanded = [pattern]
            candidates = []
            for pat in expanded:
                candidates.extend(glob.glob(pat))
            for c in sorted(set(candidates), reverse=True):
                if c not in seen and os.path.isfile(c):
                    seen.add(c)
                    yield c

    @staticmethod
    def resolve_app_path(
        *, env_vars=(), location_env_vars=(), app_names=(), scan_globs=()
    ):
        """Resolve a target application executable; the first hit wins.

        Consolidates the ``$ENV -> find_app -> install-dir scan`` discovery that
        app hand-off callers (Maya / Blender / RizomUV / Painter / Toolbag) would
        otherwise each re-implement. Resolution order:

        1. *env_vars* -- each name whose ``os.environ`` value points at an existing
           file (e.g. ``BLENDER_EXE`` / ``MAYA_EXE``).
        2. *location_env_vars* -- each ``(env_var, suffix)`` where
           ``<env_value>/<suffix>`` exists (e.g.
           ``("MAYA_LOCATION", ("bin", "maya.exe"))``). *suffix* may be a string
           or a path-segment sequence.
        3. *app_names* -- each name via :meth:`find_app` (PATH / Windows App Paths).
        4. *scan_globs* -- :meth:`scan_install_dirs` over the patterns (newest wins).

        :return: The absolute path, or ``None`` when nothing resolves.
        """
        for var in env_vars:
            p = os.environ.get(var)
            if p and os.path.isfile(p):
                return p

        for var, suffix in location_env_vars:
            loc = os.environ.get(var)
            if not loc:
                continue
            parts = [suffix] if isinstance(suffix, str) else list(suffix)
            p = os.path.join(loc, *parts)
            if os.path.isfile(p):
                return p

        for name in app_names:
            found = AppLauncher.find_app(name)
            if found:
                return found

        for found in AppLauncher.scan_install_dirs(scan_globs):
            return found
        return None

    @staticmethod
    def find_app(app_identifier):
        """
        Attempts to locate the executable for the given application identifier.

        :param app_identifier: The name or path of the application.
        :return: The absolute path to the executable, or None if not found.
        """
        # 1. Check if it's a valid absolute path
        if os.path.isabs(app_identifier) and os.path.exists(app_identifier):
            return app_identifier

        # 2. Check if it's in the PATH
        path_in_env = shutil.which(app_identifier)
        if path_in_env:
            return path_in_env

        system = platform.system().lower()

        # 3. Windows Registry (App Paths)
        if system == "windows":
            return AppLauncher._find_windows_app(app_identifier)

        # 4. Linux specific searches can go here (e.g. iterate /usr/share/applications)

        return None

    @staticmethod
    def get_running_processes(process_name):
        """
        Returns a list of PIDs of running processes matching the given name.
        Uses tasklist (Windows) or pgrep (Linux/Unix) to avoid external dependencies like psutil.

        :param process_name: The name of the process (e.g. 'notepad.exe', 'maya').
        :return: List of integer PIDs.
        """
        pids = []
        system = platform.system().lower()

        try:
            if system == "windows":
                # 'tasklist /FO CSV /NH' returns "Image Name","PID","Session Name","Session#","Mem Usage"
                # Filter by name using /FI to rely on system filter.
                # List args (no shell=True): immune to quoting/injection issues
                # if process_name ever contains shell metacharacters.
                cmd = [
                    "tasklist",
                    "/FO",
                    "CSV",
                    "/NH",
                    "/FI",
                    f"IMAGENAME eq {process_name}",
                ]
                output = subprocess.check_output(cmd).decode(errors="ignore")

                import csv
                import io

                reader = csv.reader(io.StringIO(output))
                for row in reader:
                    if len(row) >= 2:
                        try:
                            # Verify the name matches because tasklist might return empty or "INFO: No tasks..."
                            # row[0] is image name, row[1] is PID
                            if row[0].lower() == process_name.lower():
                                pids.append(int(row[1]))
                        except ValueError:
                            pass

            else:
                # Linux / Unix
                cmd = ["pgrep", "-f", process_name]
                output = subprocess.check_output(cmd).decode(errors="ignore")
                for line in output.splitlines():
                    if line.strip().isdigit():
                        pids.append(int(line.strip()))

        except subprocess.CalledProcessError:
            # pgrep returns non-zero if no process found
            pass
        except Exception as e:
            logger.debug(f"Error finding processes: {e}")

        return pids

    @staticmethod
    def close_process(pid, force=False):
        """
        Terminates the process with the given PID.

        :param pid: Process ID to terminate.
        :param force: If True, force kill (SIGKILL/TerminateProcess).
        :return: True if successful, False otherwise.
        """
        import signal

        try:
            if platform.system().lower() == "windows":
                # Use taskkill
                args = ["taskkill", "/PID", str(pid)]
                if force:
                    args.append("/F")
                # Hide output
                subprocess.check_call(
                    args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return True
            else:
                os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
                return True
        except Exception as e:
            logger.debug(f"Failed to close process {pid}: {e}")
            return False

    @staticmethod
    def _find_windows_app(app_name):
        """
        Looks up the application in the Windows Registry App Paths, then falls
        back to scanning the Program Files install roots up to two levels deep
        (covers vendors that don't register App Paths — Adobe Substance 3D
        Painter, Houdini, Reaper, etc.).
        """
        import winreg

        if not app_name.endswith(".exe"):
            app_name += ".exe"

        reg_paths = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths",  # 32-bit apps on 64-bit OS
        ]

        for reg_path in reg_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                    try:
                        with winreg.OpenKey(key, app_name) as app_key:
                            value, _ = winreg.QueryValueEx(
                                app_key, None
                            )  # Default value contains path
                            if value and os.path.exists(value):
                                return value
                    except FileNotFoundError:
                        continue  # App not in this registry path key
            except OSError:
                continue

        # Also check CURRENT_USER
        for reg_path in reg_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path) as key:
                    try:
                        with winreg.OpenKey(key, app_name) as app_key:
                            value, _ = winreg.QueryValueEx(app_key, None)
                            if value and os.path.exists(value):
                                return value
                    except FileNotFoundError:
                        continue
            except OSError:
                continue

        # Program Files glob fallback
        program_files_roots = [
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("ProgramW6432"),
        ]
        seen = set()
        for root in program_files_roots:
            if not root or root in seen:
                continue
            seen.add(root)
            for pattern in (
                os.path.join(root, app_name),
                os.path.join(root, "*", app_name),
                os.path.join(root, "*", "*", app_name),
            ):
                for match in glob.glob(pattern):
                    if os.path.isfile(match):
                        return match

        return None
