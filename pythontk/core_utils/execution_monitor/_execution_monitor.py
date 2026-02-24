# !/usr/bin/python
# coding=utf-8
import sys
import ctypes
import threading
import _thread
import os
import time
import subprocess
import tempfile
from functools import wraps


class ExecutionMonitor:
    """Utilities for monitoring and handling long-running executions."""

    _x11_lib = None
    _x11_display = None

    @staticmethod
    def _force_interrupt_main_thread():
        """Raise SystemExit in the main thread to force-stop the current operation.

        Unlike the previous implementation this does NOT kill the host process.
        It injects a SystemExit (a BaseException, rarely caught by user code) into
        the main thread via ``PyThreadState_SetAsyncExc``.  If the first attempt
        does not take effect (e.g. main thread is stuck in a C extension) we retry
        up to 3 times before falling back to ``_thread.interrupt_main()``.
        """
        main_tid = threading.main_thread().ident
        exc_type = ctypes.py_object(SystemExit)
        tid = ctypes.c_ulong(main_tid)

        for _ in range(3):
            ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, exc_type)
            if ret == 1:
                return  # Successfully scheduled
            elif ret > 1:
                # Oops – clear the exception to avoid corruption
                ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
                break
            # ret == 0 means thread id not found; retry after a short pause
            time.sleep(0.05)

        # Fallback: cooperative interrupt (same as Cancel but still better than
        # killing the host application).
        _thread.interrupt_main()

    @staticmethod
    def _force_kill_process():
        """Force kill the current process (including the host application).

        WARNING: This terminates the entire process (e.g. Maya, the Python host).
        Use only as an absolute last resort when the operation cannot be stopped
        any other way.
        """
        try:
            import signal

            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            try:
                os._exit(1)
            except Exception:
                pass

    @staticmethod
    def is_escape_pressed():
        """Check if the Escape key is currently pressed (Windows & Linux)."""
        try:
            if sys.platform == "win32":
                # VK_ESCAPE is 0x1B
                # GetAsyncKeyState returns a 16-bit integer.
                # The most significant bit indicates whether the key is currently up or down.
                return ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000

            elif sys.platform.startswith("linux"):
                try:
                    if ExecutionMonitor._x11_lib is None:
                        ExecutionMonitor._x11_lib = ctypes.cdll.LoadLibrary(
                            "libX11.so.6"
                        )
                        ExecutionMonitor._x11_lib.XOpenDisplay.restype = ctypes.c_void_p
                        ExecutionMonitor._x11_lib.XKeysymToKeycode.restype = (
                            ctypes.c_ubyte
                        )
                        ExecutionMonitor._x11_lib.XQueryKeymap.argtypes = [
                            ctypes.c_void_p,
                            ctypes.c_char * 32,
                        ]

                    if ExecutionMonitor._x11_display is None:
                        ExecutionMonitor._x11_display = (
                            ExecutionMonitor._x11_lib.XOpenDisplay(None)
                        )

                    if not ExecutionMonitor._x11_display:
                        return False

                    # XK_Escape is 0xFF1B
                    keycode = ExecutionMonitor._x11_lib.XKeysymToKeycode(
                        ExecutionMonitor._x11_display, 0xFF1B
                    )

                    keys = (ctypes.c_char * 32)()
                    ExecutionMonitor._x11_lib.XQueryKeymap(
                        ExecutionMonitor._x11_display, keys
                    )

                    byte_index = keycode // 8
                    bit_index = keycode % 8

                    # In Python 3, accessing c_char array returns bytes of length 1
                    key_byte = keys[byte_index]
                    # Convert to int
                    key_val = key_byte[0] if isinstance(key_byte, bytes) else key_byte

                    return (key_val & (1 << bit_index)) != 0
                except Exception:
                    return False
        except ImportError:
            return False

        return False

    _interpreter_override = None

    @classmethod
    def set_interpreter(cls, path):
        """Set a custom Python interpreter to use for subprocesses.

        Args:
            path (str): Absolute path to the python executable.
        """
        cls._interpreter_override = path

    @staticmethod
    def _get_python_executable():
        """
        Get the path to the Python interpreter.
        Returns _interpreter_override if set, otherwise attempts to resolve the interpreter from sys.executable.
        """
        if ExecutionMonitor._interpreter_override:
            return ExecutionMonitor._interpreter_override

        executable = sys.executable
        if not executable:
            return sys.executable

        # If the executable looks like a python interpreter, return it.
        name = os.path.basename(executable).lower()
        name_no_ext = os.path.splitext(name)[0]
        if (
            "python" in name_no_ext
            or name_no_ext.endswith("py")
            or name_no_ext == "hython"
        ):
            return executable

        # Otherwise, look for a companion interpreter in the same directory.
        dir_path = os.path.dirname(executable)

        # 1. Try generic naming convention: {app}py.exe (e.g. app -> apppy)
        # Handle 'batch' variations (e.g. appbatch -> apppy)
        base_name = name_no_ext.replace("batch", "")
        candidates = [base_name + "py"]

        # 2. Try standard python executable names
        candidates.extend(["python", "python3", "hython"])

        extensions = [".exe"] if sys.platform == "win32" else [""]

        for cand in candidates:
            for ext in extensions:
                path = os.path.join(dir_path, cand + ext)
                if os.path.exists(path):
                    return path

        return executable

    @staticmethod
    def _start_gif_process(gif_path):
        """Starts a subprocess to display a GIF using tkinter."""
        try:
            # Locate the _gif_viewer.py script in the same directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_dir, "_gif_viewer.py")

            if not os.path.exists(script_path):
                return None

            executable = ExecutionMonitor._get_python_executable()

            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            process = subprocess.Popen(
                [executable, script_path, gif_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                startupinfo=startupinfo,
            )
            return process
        except Exception:
            return None

    @staticmethod
    def _stop_gif_process(process):
        """Stops the GIF subprocess."""
        if process:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()

    @staticmethod
    def on_long_execution(
        threshold, callback, interval=None, allow_escape_cancel=False, indicator=None
    ):
        """
        Decorator that triggers a callback if the decorated function
        takes longer than `threshold` seconds to execute.

        Args:
            threshold (float): Time in seconds before callback is triggered.
            callback (callable): Function to call if threshold is exceeded.
                                 If the callback returns False, a KeyboardInterrupt will be raised
                                 in the main thread to attempt to abort the execution.
            interval (float|bool, optional): If True, repeats every `threshold` seconds.
                                             If float, repeats every `interval` seconds.
            allow_escape_cancel (bool): If True, holding Escape will interrupt the main thread immediately.
            indicator (bool|str, optional): If True, displays the default 'task_indicator.gif'.
                                            If a string, displays the GIF at the specified path.
                                            Runs in a separate process to ensure animation during blocking tasks.
        """
        # If interval is True, use threshold as the interval
        repeat_interval = threshold if interval is True else interval

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                stop_event = threading.Event()

                gif_process = None
                resolved_gif_path = indicator
                if resolved_gif_path:
                    if resolved_gif_path is True:
                        # Use default bundled GIF
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        resolved_gif_path = os.path.join(
                            current_dir, "task_indicator.gif"
                        )

                    if os.path.exists(resolved_gif_path):
                        gif_process = ExecutionMonitor._start_gif_process(
                            resolved_gif_path
                        )

                def wait_for_stop_or_timeout(duration):
                    """Returns True if stopped (event set or aborted), False if timeout."""
                    if not allow_escape_cancel:
                        return stop_event.wait(duration)

                    # Polling
                    remaining = duration
                    step = 0.1
                    while remaining > 0:
                        wait_time = min(step, remaining)
                        if stop_event.wait(wait_time):
                            return True
                        if ExecutionMonitor.is_escape_pressed():
                            _thread.interrupt_main()
                            return True  # Aborted
                        remaining -= wait_time
                    return False

                def timer_func():
                    # Wait for the initial threshold
                    if not wait_for_stop_or_timeout(threshold):
                        result = callback()
                        if result is False:
                            _thread.interrupt_main()
                            # If interrupt_main fails to stop the process (e.g. stuck in C extension),
                            # we can try to raise it again after a short delay or escalate.
                            return
                        elif result == "FORCE_KILL":
                            ExecutionMonitor._force_kill_process()
                            return
                        elif result == "FORCE_INTERRUPT":
                            ExecutionMonitor._force_interrupt_main_thread()
                            return
                        elif result == "STOP_MONITORING":
                            return

                        # If repeat_interval is set, keep repeating
                        if repeat_interval:
                            while not wait_for_stop_or_timeout(repeat_interval):
                                result = callback()
                                if result is False:
                                    _thread.interrupt_main()
                                    return
                                elif result == "FORCE_KILL":
                                    ExecutionMonitor._force_kill_process()
                                    return
                                elif result == "FORCE_INTERRUPT":
                                    ExecutionMonitor._force_interrupt_main_thread()
                                    return
                                elif result == "STOP_MONITORING":
                                    return

                t = threading.Thread(target=timer_func)
                t.daemon = True
                t.start()

                try:
                    result = func(*args, **kwargs)
                finally:
                    stop_event.set()
                    if gif_process:
                        ExecutionMonitor._stop_gif_process(gif_process)
                return result

            return wrapper

        return decorator

    @staticmethod
    def _start_dialog_process(title, message):
        """Starts a subprocess to display a custom dialog using tkinter.

        Returns:
            subprocess.Popen: The dialog process, or None on error.
        """
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_dir, "_dialog_viewer.py")

            if not os.path.exists(script_path):
                return None

            executable = ExecutionMonitor._get_python_executable()

            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            process = subprocess.Popen(
                [executable, script_path, title, message],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                startupinfo=startupinfo,
            )
            return process
        except Exception:
            return None

    @staticmethod
    def show_long_execution_dialog(title, message, force_action=None):
        """Show a dialog to ask the user how to proceed with a long operation.

        Uses a subprocess-based tkinter dialog for custom button labels (VS Code style).

        Args:
            title (str): Dialog window title.
            message (str): Body text.
            force_action (str|None): Controls the force button.
                ``None``  – no force button (default, 2-button dialog).
                ``"interrupt"`` – show *Force Stop* (raises SystemExit in main thread).
                ``"kill"``      – show *Force Quit* (terminates the host process).

        Returns:
            bool/str: True to continue waiting, False to abort,
                      ``"FORCE_INTERRUPT"`` or ``"FORCE_KILL"`` for the force action.
        """
        import sys

        # Map force_action to button label and return sentinel
        if force_action == "interrupt":
            force_label = "Force Stop"
            force_sentinel = "FORCE_INTERRUPT"
        elif force_action == "kill":
            force_label = "Force Quit"
            force_sentinel = "FORCE_KILL"
        else:
            force_label = None
            force_sentinel = None

        if sys.platform == "win32":
            # Use custom tkinter dialog for better button labels
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                script_path = os.path.join(current_dir, "_dialog_viewer.py")

                if os.path.exists(script_path):
                    executable = ExecutionMonitor._get_python_executable()

                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE

                    cmd = [executable, script_path, title, message]
                    if force_label:
                        cmd.append(force_label)

                    result = subprocess.run(cmd, startupinfo=startupinfo)

                    # Exit codes: 0=Keep Waiting, 10=Cancel, 2=Force, 3=Closed
                    if result.returncode == 0 or result.returncode == 3:
                        return True  # Keep Waiting (or window closed)
                    elif result.returncode == 10:
                        return False  # Cancel/Stop Operation
                    elif result.returncode == 2:
                        return force_sentinel
                    return True
            except Exception:
                pass

            # Fallback to MessageBox if custom dialog fails
            try:
                import ctypes

                if force_label:
                    # MB_YESNOCANCEL | MB_ICONWARNING | MB_SYSTEMMODAL | MB_TOPMOST
                    flags = 0x03 | 0x30 | 0x1000 | 0x40000
                    fallback_msg = f"{message}\n\nYes: Keep Waiting\nNo: Cancel\nCancel: {force_label}"
                else:
                    # MB_YESNO | MB_ICONWARNING | MB_SYSTEMMODAL | MB_TOPMOST
                    flags = 0x04 | 0x30 | 0x1000 | 0x40000
                    fallback_msg = f"{message}\n\nYes: Keep Waiting\nNo: Cancel"

                response = ctypes.windll.user32.MessageBoxW(
                    0, fallback_msg, title, flags
                )

                if response == 6:  # IDYES
                    return True
                if response == 7:  # IDNO
                    return False
                if response == 2 and force_sentinel:  # IDCANCEL
                    return force_sentinel
            except ImportError:
                return True

        elif sys.platform.startswith("linux"):
            import subprocess
            import shutil

            # Try Zenity (GNOME/Standard)
            if shutil.which("zenity"):
                cmd = [
                    "zenity",
                    "--question",
                    "--title",
                    title,
                    "--text",
                    message,
                    "--ok-label",
                    "Keep Waiting",
                    "--cancel-label",
                    "Cancel",
                ]
                if force_label:
                    cmd.extend(["--extra-button", force_label])
                try:
                    result = subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    if force_label and result.stdout.strip() == force_label:
                        return force_sentinel
                    if result.returncode == 0:
                        return True  # Keep Waiting
                    else:
                        return False  # Cancel
                except Exception:
                    pass

            # Try KDialog (KDE)
            elif shutil.which("kdialog"):
                if force_label:
                    cmd = [
                        "kdialog",
                        "--title",
                        title,
                        "--yesnocancel",
                        message,
                        "--yes-label",
                        "Keep Waiting",
                        "--no-label",
                        "Cancel",
                        "--cancel-label",
                        force_label,
                    ]
                else:
                    cmd = [
                        "kdialog",
                        "--title",
                        title,
                        "--yesno",
                        message,
                        "--yes-label",
                        "Keep Waiting",
                        "--no-label",
                        "Cancel",
                    ]
                try:
                    result = subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    if result.returncode == 0:
                        return True
                    elif result.returncode == 1:
                        return False
                    elif result.returncode == 2 and force_sentinel:
                        return force_sentinel
                except Exception:
                    pass

        return True  # Default to continue if platform not supported

    @staticmethod
    def execution_monitor(
        threshold,
        message,
        logger=None,
        allow_escape_cancel=False,
        show_dialog: bool = True,
        force_action: str | None = None,
        watchdog_timeout: float | None = None,
        watchdog_heartbeat_interval: float = 1.0,
        watchdog_check_interval: float = 1.0,
        watchdog_kill_tree: bool = True,
        watchdog_heartbeat_path: str | None = None,
        indicator: bool | str | None = None,
    ):
        """
        Decorator that monitors execution time and (optionally) prompts the user via a native
        dialog if the threshold is exceeded. Can also (optionally) enable an external heartbeat
        watchdog that can force-kill the process if the host application hard-hangs.

        Args:
            threshold (float): Time in seconds before warning.
            message (str): Message to display in the dialog/logs.
            logger (logging.Logger, optional): Logger to use for status updates.
            allow_escape_cancel (bool): If True, holding Escape will interrupt the main thread immediately.
            show_dialog (bool): If False, do not show a blocking dialog; only log warnings.
            force_action (str|None): Controls the force button in the dialog.
                ``None`` (default) – no force button; dialog shows only Keep Waiting / Cancel.
                ``"interrupt"``    – adds a *Force Stop* button that raises ``SystemExit`` in
                                     the main thread without killing the host process.
                ``"kill"``         – adds a *Force Quit* button that terminates the entire
                                     host process (use as a last resort).
            watchdog_timeout (float|None): If set, starts an external watchdog that kills this process
                if the heartbeat stalls for longer than this many seconds.
            watchdog_heartbeat_interval (float): Heartbeat write interval in seconds.
            watchdog_check_interval (float): Watchdog polling interval in seconds.
            watchdog_kill_tree (bool): If True, attempt to kill child processes too.
            watchdog_heartbeat_path (str|None): Optional heartbeat file path override.
            indicator (bool|str, optional): If True, displays the default 'task_indicator.gif'.
                                            If a string, displays the GIF at the specified path.
        """

        _dialog_shown = [False]

        def callback():
            full_msg = f"{message} (taking longer than {threshold}s...)"
            if logger:
                logger.warning(full_msg)

            if not show_dialog or _dialog_shown[0]:
                # Non-interactive mode or dialog already shown: keep waiting silently.
                return True

            _dialog_shown[0] = True
            esc_hint = (
                "\n\nPress and hold Esc at any time to cancel the operation."
                if allow_escape_cancel
                else ""
            )
            result = ExecutionMonitor.show_long_execution_dialog(
                "Long Execution Warning",
                f"{full_msg}\n\nThe operation is not responding.\n"
                "You can keep waiting or cancel the operation."
                f"{esc_hint}",
                force_action=force_action,
            )

            if logger:
                if result is True:
                    logger.info("Continuing execution (Keep Waiting).")
                elif result is False:
                    logger.warning("Operation cancelled by user.")
                elif result == "FORCE_INTERRUPT":
                    logger.critical("Force stopping operation by user request.")
                elif result == "FORCE_KILL":
                    logger.critical("Force quitting application by user request.")

            return result

        monitored = ExecutionMonitor.on_long_execution(
            threshold,
            callback,
            interval=True,
            allow_escape_cancel=allow_escape_cancel,
            indicator=indicator,
        )

        if watchdog_timeout is None:
            return monitored

        # Layer the external watchdog outside the timer-based monitor.
        return ExecutionMonitor.external_watchdog(
            timeout=float(watchdog_timeout),
            message=message,
            heartbeat_interval=float(watchdog_heartbeat_interval),
            check_interval=float(watchdog_check_interval),
            kill_tree=bool(watchdog_kill_tree),
            logger=logger,
            heartbeat_path=watchdog_heartbeat_path,
        )(monitored)

    @staticmethod
    def _default_heartbeat_path(tag: str = "execution_monitor") -> str:
        safe_tag = "".join(
            c if c.isalnum() or c in ("-", "_", ".") else "_" for c in (tag or "")
        )
        filename = f"{safe_tag}_hb_{os.getpid()}.txt"
        return os.path.join(tempfile.gettempdir(), filename)

    @staticmethod
    def _start_heartbeat_writer(heartbeat_path: str, interval: float, logger=None):
        """Start a daemon thread that touches/writes a heartbeat file periodically."""
        stop_event = threading.Event()

        def _write_once():
            try:
                # Ensure directory exists (temp should, but be safe)
                os.makedirs(os.path.dirname(heartbeat_path), exist_ok=True)
                with open(heartbeat_path, "w", encoding="utf-8") as f:
                    f.write(str(time.time()))
                # Ensure mtime updates even if contents identical
                os.utime(heartbeat_path, None)
            except Exception as e:
                if logger:
                    logger.warning(f"Heartbeat write failed: {e}")

        def _loop():
            # Initial heartbeat immediately
            _write_once()
            while not stop_event.wait(max(0.05, float(interval))):
                _write_once()

        t = threading.Thread(target=_loop, name="ExecutionMonitorHeartbeat")
        t.daemon = True
        t.start()

        def stop():
            stop_event.set()
            try:
                # Best-effort cleanup
                if os.path.exists(heartbeat_path):
                    os.remove(heartbeat_path)
            except Exception:
                pass

        return stop

    @staticmethod
    def _spawn_watchdog_subprocess(
        pid: int,
        heartbeat_path: str,
        timeout: float,
        check_interval: float,
        kill_tree: bool,
        logger=None,
    ):
        """Spawn a small external watchdog process that kills `pid` if heartbeat stalls."""

        # Use an inline Python program to avoid PYTHONPATH/import issues.
        # Args: pid heartbeat_path timeout check_interval kill_tree stop_file
        stop_file = heartbeat_path + ".stop"
        code = r"""
import os, sys, time, subprocess, signal

pid = int(sys.argv[1])
heartbeat_path = sys.argv[2]
timeout = float(sys.argv[3])
check_interval = float(sys.argv[4])
kill_tree = sys.argv[5].lower() in ("1","true","yes","y")
stop_file = sys.argv[6]

is_win = sys.platform == "win32"

def process_alive(p):
    if is_win:
        # Avoid heavy polling; attempt a lightweight OpenProcess.
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, p)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return True
    else:
        try:
            os.kill(p, 0)
            return True
        except Exception:
            return False

def kill_process(p):
    if is_win:
        cmd = ["taskkill", "/PID", str(p), "/F"]
        if kill_tree:
            cmd.insert(3, "/T")
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    else:
        try:
            if kill_tree:
                try:
                    os.killpg(os.getpgid(p), signal.SIGKILL)
                    return
                except Exception:
                    pass
            os.kill(p, signal.SIGKILL)
        except Exception:
            pass

start = time.time()

while True:
    if os.path.exists(stop_file):
        break

    # If process is gone, exit
    if not process_alive(pid):
        break

    now = time.time()

    # If heartbeat doesn't exist yet, allow a grace period
    if not os.path.exists(heartbeat_path):
        if now - start > timeout:
            kill_process(pid)
            break
        time.sleep(check_interval)
        continue

    try:
        age = now - os.path.getmtime(heartbeat_path)
    except Exception:
        age = timeout + 1.0

    if age > timeout:
        kill_process(pid)
        break

    time.sleep(check_interval)
"""

        args = [
            sys.executable,
            "-c",
            code,
            str(int(pid)),
            str(heartbeat_path),
            str(float(timeout)),
            str(float(check_interval)),
            "1" if kill_tree else "0",
            str(stop_file),
        ]

        try:
            # Detach from console on Windows to reduce UI interference.
            creationflags = 0
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except Exception as e:
            if logger:
                logger.warning(f"Failed to start watchdog subprocess: {e}")
            return None, None

        def stop():
            # Signal watchdog to exit
            try:
                with open(stop_file, "w", encoding="utf-8") as f:
                    f.write("stop")
            except Exception:
                pass
            try:
                if proc and proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
            try:
                if os.path.exists(stop_file):
                    os.remove(stop_file)
            except Exception:
                pass

        return proc, stop

    @staticmethod
    def external_watchdog(
        timeout: float,
        message: str = "Operation appears to have stalled",
        heartbeat_interval: float = 1.0,
        check_interval: float = 1.0,
        kill_tree: bool = True,
        logger=None,
        heartbeat_path: str | None = None,
    ):
        """Decorator that starts an OS-level watchdog for the current process.

        This is meant for cases where the host application can hard-hang (e.g. Maya). The
        watchdog runs in a separate process and will force-kill this process if the heartbeat
        file stops updating for longer than `timeout`.

        Notes:
            - Works on Windows and Linux.
            - If the entire process is frozen, the watchdog can still kill it.
            - This is an aggressive safety valve; prefer cooperative cancellation when possible.
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                hb_path = heartbeat_path or ExecutionMonitor._default_heartbeat_path(
                    "watchdog"
                )
                if logger:
                    logger.info(
                        f"Starting external watchdog (timeout={timeout}s, heartbeat={hb_path})"
                    )

                stop_hb = ExecutionMonitor._start_heartbeat_writer(
                    hb_path, heartbeat_interval, logger=logger
                )
                proc, stop_watchdog = ExecutionMonitor._spawn_watchdog_subprocess(
                    os.getpid(),
                    hb_path,
                    timeout,
                    check_interval,
                    kill_tree,
                    logger=logger,
                )

                try:
                    return func(*args, **kwargs)
                finally:
                    # Best-effort shutdown of watchdog + heartbeat
                    try:
                        stop_hb()
                    except Exception:
                        pass
                    try:
                        if stop_watchdog:
                            stop_watchdog()
                    except Exception:
                        pass

            return wrapper

        return decorator
