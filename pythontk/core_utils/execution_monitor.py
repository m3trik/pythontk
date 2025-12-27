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

    @staticmethod
    def on_long_execution(
        threshold, callback, interval=None, allow_escape_cancel=False
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
        """
        # If interval is True, use threshold as the interval
        repeat_interval = threshold if interval is True else interval

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                stop_event = threading.Event()

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
                                elif result == "STOP_MONITORING":
                                    return

                t = threading.Thread(target=timer_func)
                t.daemon = True
                t.start()

                try:
                    result = func(*args, **kwargs)
                finally:
                    stop_event.set()
                return result

            return wrapper

        return decorator

    @staticmethod
    def show_long_execution_dialog(title, message):
        """Show a native blocking dialog to ask the user how to proceed with a long operation.

        Returns:
            bool/str: True to continue waiting, False to abort, "STOP_MONITORING" to ignore.
        """
        import sys

        if sys.platform == "win32":
            try:
                import ctypes

                # MB_YESNOCANCEL | MB_ICONWARNING | MB_SYSTEMMODAL | MB_TOPMOST
                # 0x03 | 0x30 | 0x1000 | 0x40000
                flags = 0x03 | 0x30 | 0x1000 | 0x40000

                response = ctypes.windll.user32.MessageBoxW(0, message, title, flags)

                # IDYES=6, IDNO=7, IDCANCEL=2
                if response == 6:
                    return True
                if response == 7:
                    return False
                if response == 2:
                    return "STOP_MONITORING"
            except ImportError:
                return True

        elif sys.platform.startswith("linux"):
            import subprocess
            import shutil

            # Try Zenity (GNOME/Standard)
            if shutil.which("zenity"):
                # Zenity returns 0 for OK, 1 for Cancel/Close.
                # Extra buttons print their label to stdout.
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
                    "Stop Operation",
                    "--extra-button",
                    "Stop Monitoring",
                ]
                try:
                    result = subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    if result.returncode == 0:
                        if result.stdout.strip() == "Stop Monitoring":
                            return "STOP_MONITORING"
                        return True  # Keep Waiting
                    else:
                        return False  # Stop Operation
                except Exception:
                    pass

            # Try KDialog (KDE)
            elif shutil.which("kdialog"):
                # kdialog --yesnocancel "text" --title "title"
                # Returns: 0 (Yes), 1 (No), 2 (Cancel)
                # We map: Yes->Wait, No->Stop, Cancel->Stop Monitoring
                cmd = [
                    "kdialog",
                    "--title",
                    title,
                    "--yesnocancel",
                    message,
                    "--yes-label",
                    "Keep Waiting",
                    "--no-label",
                    "Stop Operation",
                    "--cancel-label",
                    "Stop Monitoring",
                ]
                try:
                    result = subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    if result.returncode == 0:
                        return True
                    elif result.returncode == 1:
                        return False
                    elif result.returncode == 2:
                        return "STOP_MONITORING"
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
        watchdog_timeout: float | None = None,
        watchdog_heartbeat_interval: float = 1.0,
        watchdog_check_interval: float = 1.0,
        watchdog_kill_tree: bool = True,
        watchdog_heartbeat_path: str | None = None,
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
            watchdog_timeout (float|None): If set, starts an external watchdog that kills this process
                if the heartbeat stalls for longer than this many seconds.
            watchdog_heartbeat_interval (float): Heartbeat write interval in seconds.
            watchdog_check_interval (float): Watchdog polling interval in seconds.
            watchdog_kill_tree (bool): If True, attempt to kill child processes too.
            watchdog_heartbeat_path (str|None): Optional heartbeat file path override.
        """

        def callback():
            full_msg = f"{message} (taking longer than {threshold}s...)"
            if logger:
                logger.warning(full_msg)

            if not show_dialog:
                # Non-interactive mode: keep waiting, continue monitoring.
                return True

            result = ExecutionMonitor.show_long_execution_dialog(
                "Long Execution Warning",
                f"{full_msg}\n\n"
                "Do you want to continue waiting?\n\n"
                "Yes:\tKeep waiting (ask again later).\n"
                "No:\tStop the operation.\n"
                "Cancel:\tKeep waiting (don't ask again).",
            )

            if logger:
                if result is True:
                    logger.info("Continuing execution (monitoring active).")
                elif result is False:
                    logger.warning("Aborting execution by user request.")
                elif result == "STOP_MONITORING":
                    logger.info("Continuing execution (monitoring disabled).")

            return result

        monitored = ExecutionMonitor.on_long_execution(
            threshold, callback, interval=True, allow_escape_cancel=allow_escape_cancel
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
