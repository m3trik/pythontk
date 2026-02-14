# !/usr/bin/python
# coding=utf-8
import os
import sys
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
    def launch(app_identifier, args=None, cwd=None, detached=True):
        """
        Launches an application.

        :param app_identifier: The name or path of the application (e.g., 'notepad', 'firefox', 'C:/Apps/MyTool.exe').
        :param args: A list of arguments to pass to the application.
        :param cwd: The working directory for the application.
        :param detached: If True, launches as a separate process (application keeps running if script ends).
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
    def run(app_identifier, args=None, cwd=None, timeout=None):
        """Execute an application synchronously and return its result.

        Unlike :meth:`launch` (fire-and-forget), this method blocks until the
        process finishes, captures stdout/stderr, and honours a *timeout*.

        :param app_identifier: Name or path of the application.
        :param args: Arguments to pass (str, list, or tuple).
        :param cwd: Working directory for the process.
        :param timeout: Maximum seconds to wait before raising
                        ``subprocess.TimeoutExpired``.  *None* = no limit.
        :return: A ``subprocess.CompletedProcess`` with *returncode*,
                 *stdout*, and *stderr* (decoded text).
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
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )

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
                # Filter by name using /FI to rely on system filter
                cmd = f'tasklist /FO CSV /NH /FI "IMAGENAME eq {process_name}"'
                # run via subprocess
                output = subprocess.check_output(cmd, shell=True).decode(
                    errors="ignore"
                )

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
        Looks up the application in the Windows Registry App Paths.
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

        return None
