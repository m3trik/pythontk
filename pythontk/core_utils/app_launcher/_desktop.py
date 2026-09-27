# !/usr/bin/python
# coding=utf-8
"""The interactive desktop: the session a GUI runs in, and a process's windows.

Windows sessions (``current_session_id``, ``active_console_session_id``,
``is_interactive_session``; ``launch_in_session`` puts a GUI app on the
logged-in desktop from the services session 0, through a discovered PsExec)
and window lookup (``wait_for_ready``, ``get_window_titles``: ``EnumWindows``
on Windows, the X11 client over the process tree on Linux).

One job of :class:`AppLauncher`, composed in ``_app_launcher.py``. A cross-call
goes through the facade (imported on use: that module imports this one for its
base), so a caller's ``patch.object(AppLauncher, ...)`` reaches it.
"""

import os
import shutil
import platform
import subprocess
import logging

logger = logging.getLogger(__name__)


class _DesktopMixin:
    """Desktop sessions and window lookup; a private part of :class:`AppLauncher`."""

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
        """True if the current process can show GUIs.

        Windows: an interactive session (non-zero id, with a window station) --
        False for the services session 0 (a non-interactive SSH or scheduled
        task context; a missing id counts as non-interactive). Linux: a display
        server to talk to (``DISPLAY`` or ``WAYLAND_DISPLAY``) -- False over a
        plain SSH login. Elsewhere (macOS): True.
        """
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        system = platform.system().lower()
        if system == "linux":
            return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if system != "windows":
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
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

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
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

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
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

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
            # X11/XWayland windows of the process or its descendants (a Linux
            # launcher is often a script that forks the real binary). No X
            # server to ask (pure Wayland, SSH): unknown, so not a reason to wait.
            titles = AppLauncher._x11_titles(pid)
            return True if titles is None else bool(titles)

        return True

    @staticmethod
    def _x11_titles(pid):
        """X11 window titles of *pid* and its descendants; None without a server."""
        from pythontk.core_utils.x11 import X11
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        return X11.window_titles(AppLauncher.process_tree(pid))

    @staticmethod
    def get_window_titles(pid):
        """
        Returns a list of window titles owned by the given PID.
        """
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

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

        elif system == "linux":
            titles = AppLauncher._x11_titles(pid) or []

        return titles
