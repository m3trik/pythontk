# !/usr/bin/python
# coding=utf-8
import threading
import _thread
from functools import wraps


class ExecutionMonitor:
    """Utilities for monitoring and handling long-running executions."""

    _x11_lib = None
    _x11_display = None

    @staticmethod
    def is_escape_pressed():
        """Check if the Escape key is currently pressed (Windows & Linux)."""
        import sys

        try:
            if sys.platform == "win32":
                import ctypes

                # VK_ESCAPE is 0x1B
                # GetAsyncKeyState returns a 16-bit integer.
                # The most significant bit indicates whether the key is currently up or down.
                return ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000

            elif sys.platform.startswith("linux"):
                import ctypes

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
    def execution_monitor(threshold, message, logger=None, allow_escape_cancel=False):
        """
        Decorator that monitors execution time and prompts the user via a native dialog
        if the threshold is exceeded.

        Args:
            threshold (float): Time in seconds before warning.
            message (str): Message to display in the dialog/logs.
            logger (logging.Logger, optional): Logger to use for status updates.
            allow_escape_cancel (bool): If True, holding Escape will interrupt the main thread immediately.
        """

        def callback():
            full_msg = f"{message} (taking longer than {threshold}s...)"
            if logger:
                logger.warning(full_msg)

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

        return ExecutionMonitor.on_long_execution(
            threshold, callback, interval=True, allow_escape_cancel=allow_escape_cancel
        )
