# !/usr/bin/python
# coding=utf-8
import os
import glob
import subprocess
import shutil
import platform
import logging

logger = logging.getLogger(__name__)


class AppLauncher:
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

        cmd = [executable_path]
        if args:
            if isinstance(args, str):
                cmd.append(args)
            elif isinstance(args, (list, tuple)):
                cmd.extend(args)

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
    def run(app_identifier, args=None, cwd=None, timeout=None, output_file=None, env=None):
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
        :return: A ``subprocess.CompletedProcess`` with *returncode* and, unless
                 *output_file* is set, *stdout*/*stderr* (decoded text).
        :raises FileNotFoundError: If the application cannot be found.
        :raises subprocess.TimeoutExpired: If *timeout* is exceeded.
        """
        executable_path = AppLauncher.find_app(app_identifier)
        if not executable_path:
            raise FileNotFoundError(f"Application '{app_identifier}' not found.")

        cmd = [executable_path]
        if args:
            if isinstance(args, str):
                cmd.append(args)
            elif isinstance(args, (list, tuple)):
                cmd.extend(args)

        logger.debug(f"Running (blocking): {cmd}")
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
                )
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            env=env,
        )

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

            sid = ctypes.windll.kernel32.WTSGetActiveConsoleSessionId()
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
        ``PsExec.exe``) → common tool dirs. Returns the path or ``None``.
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
        for d in (r"M:\tools", r"C:\tools", os.environ.get("ProgramFiles", "")):
            if d:
                candidates += [os.path.join(d, n) for n in ("PsExec64.exe", "PsExec.exe")]
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
        """Yield existing files matching *scan_globs*, newest (reverse-sorted) first.

        A ``{program_files}`` token in a pattern expands to both Program Files roots,
        so callers write one portable pattern instead of hardcoding the 64/32-bit
        dirs::

            AppLauncher.scan_install_dirs([r"{program_files}\\Autodesk\\Maya*\\bin\\maya.exe"])

        :param scan_globs: An iterable of glob patterns (each may use the token).
        :return: A generator of absolute file paths, newest first.
        """
        pf64, pf32 = AppLauncher._program_files_roots()
        candidates = []
        for pattern in scan_globs:
            if "{program_files}" in pattern:
                expanded = [
                    pattern.format(program_files=pf64),
                    pattern.format(program_files=pf32),
                ]
            else:
                expanded = [pattern]
            for pat in expanded:
                candidates.extend(glob.glob(pat))
        for c in sorted(set(candidates), reverse=True):
            if os.path.isfile(c):
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
                    "tasklist", "/FO", "CSV", "/NH",
                    "/FI", f"IMAGENAME eq {process_name}",
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
