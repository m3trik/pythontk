# !/usr/bin/python
# coding=utf-8
from __future__ import annotations

import sys
import ctypes
import glob
import inspect
import threading
import _thread
import os
import time
import subprocess
from functools import wraps

from pythontk.core_utils.execution_monitor import _sidecar


class _EscapeHoldDetector:
    """Stateful Esc-hold probe: ``True`` once Esc is held for *hold_seconds*.

    Raw key state is the wrong signal to cancel on. Esc is the most overloaded
    key in a DCC (dismiss popup, exit tool, stop playback) and
    ``GetAsyncKeyState`` is system-wide, so a single positive sample means
    almost nothing. Requiring a sustained hold *and* window ownership turns it
    into a deliberate gesture.

    Usable two ways with the same instance: called repeatedly from a monitor
    thread, or registered as a :class:`~pythontk.CancelScope` pull source and
    polled at the operation's own checkpoints.
    """

    def __init__(self, hold_seconds: float = 0.4, require_foreground: bool = True):
        self.hold_seconds = max(0.0, float(hold_seconds))
        self.require_foreground = bool(require_foreground)
        self._held_since = None

    def reset(self):
        """Forget any in-progress hold."""
        self._held_since = None

    def __call__(self) -> bool:
        if not ExecutionMonitor.is_escape_pressed():
            self._held_since = None
            return False
        if self.require_foreground and not ExecutionMonitor.is_foreground_process():
            self._held_since = None
            return False

        now = time.monotonic()
        if self._held_since is None:
            self._held_since = now
        return (now - self._held_since) >= self.hold_seconds


class ExecutionMonitor:
    """Utilities for monitoring and handling long-running executions.

    Cancellation policy
    -------------------
    This class no longer cancels by injecting exceptions into the main thread.
    ``_thread.interrupt_main`` and ``PyThreadState_SetAsyncExc`` deliver at an
    arbitrary bytecode boundary, cannot be revoked once armed, and cannot
    interrupt a native call at all — so a cancel would routinely land *after*
    the operation finished, inside whatever unrelated code ran next.

    Pass a :class:`~pythontk.CancelScope` (``cancel_scope=``) and every cancel
    affordance here — the dialog's *Cancel* button, the Esc-hold poller — only
    sets that scope's flag, which the operation consumes at a checkpoint it
    chose. The async-exception paths survive solely behind the explicit
    *Force Stop* / *Force Quit* buttons, which the user opts into knowing the
    operation cannot be stopped safely.

    Out-of-process UI
    -----------------
    The cursor indicator, the long-execution dialog and the external watchdog
    all run as sidecar processes (``_sidecar.py``) under the interpreter
    :meth:`_get_python_executable` resolves — a Tk window in this process
    could not animate or answer while the main thread is blocked. Hosts whose
    bundled python is not discoverable pin it with :meth:`set_interpreter`.
    """

    _x11_lib = None
    _x11_display = None
    _interpreter_override = None

    #: Prefix namespace for the watchdog heartbeat files (a ``TempArtifacts``
    #: store: a crash leftover is reclaimed by a later run's stale sweep).
    _TEMP_PREFIX = "execution_monitor"

    @staticmethod
    def escape_hold_source(hold_seconds: float = 0.4, require_foreground: bool = True):
        """Build an Esc-hold probe for use as a ``CancelScope`` pull source.

        Parameters:
            hold_seconds (float): Sustained hold required before reporting True.
            require_foreground (bool): Ignore Esc unless the focused window
                belongs to this process.

        Returns:
            Callable[[], bool]: Stateful probe; keep one instance per scope.

        Example:
            scope.add_source(ptk.ExecutionMonitor.escape_hold_source())
        """
        return _EscapeHoldDetector(hold_seconds, require_foreground)

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
    def is_foreground_process():
        """True when the focused window belongs to this process.

        Key-state probing via ``GetAsyncKeyState`` is **system-wide**: it reports
        physical key state regardless of which application has focus, so Esc
        pressed in a browser would cancel a background operation here. Gating on
        window ownership removes that entire false-positive class.

        Returns ``True`` on platforms where ownership can't be determined, so
        the gate never *removes* an existing cancel affordance.
        """
        if sys.platform != "win32":
            return True
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return False
            pid = ctypes.c_ulong(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            return pid.value == os.getpid()
        except Exception:
            return True

    @staticmethod
    def is_escape_pressed() -> bool:
        """Check if the Escape key is currently pressed (Windows & Linux).

        Reports raw physical key state — it does **not** consider window focus.
        Callers driving cancellation should gate on :meth:`is_foreground_process`
        and require a sustained hold; see :meth:`escape_hold_source`.
        """
        try:
            if sys.platform == "win32":
                # VK_ESCAPE is 0x1B; the most significant bit of the 16-bit
                # GetAsyncKeyState result is the "currently down" flag.
                return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)

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
        except Exception:
            # Key-state probing is best-effort; an unexpected failure here must
            # not propagate — it would kill the monitor thread (and with it the
            # threshold callback), not just Esc support.
            return False

        return False

    @classmethod
    def set_interpreter(cls, path):
        """Set a custom Python interpreter to use for subprocesses.

        Parameters:
            path (str): Absolute path to the python executable.
        """
        cls._interpreter_override = path

    @staticmethod
    def _looks_like_python(path: str) -> bool:
        name = os.path.splitext(os.path.basename(path))[0].lower()
        return "python" in name or name.endswith("py") or name == "hython"

    @staticmethod
    def _get_python_executable():
        """Path of a python interpreter for the sidecar processes, or ``None``.

        :meth:`set_interpreter` wins. Otherwise ``sys.executable`` when it is
        itself a python; else a companion beside it (``maya.exe`` → ``mayapy``,
        ``3dsmax`` → ``3dsmaxpy``, a sibling ``python``/``hython``); else the
        python bundled under ``sys.prefix`` (Blender: ``<ver>/python/bin``).

        ``None``, never the host binary: every caller hands the result a python
        *script* to run, and inside a host with no discoverable interpreter the
        "spinner" used to be a second copy of the host application.
        """
        if ExecutionMonitor._interpreter_override:
            return ExecutionMonitor._interpreter_override

        executable = sys.executable or ""
        if executable and ExecutionMonitor._looks_like_python(executable):
            return executable

        ext = ".exe" if sys.platform == "win32" else ""
        candidates = []
        if executable:
            dir_path = os.path.dirname(executable)
            # {app}py beside the app (mayabatch → mayapy), then generic names.
            base = os.path.splitext(os.path.basename(executable))[0].lower()
            base = base.replace("batch", "")
            for name in (base + "py", "python", "python3", "hython"):
                candidates.append(os.path.join(dir_path, name + ext))
        prefixes = dict.fromkeys(
            p for p in (sys.prefix, sys.base_prefix, sys.exec_prefix) if p
        )
        for prefix in prefixes:
            candidates.append(os.path.join(prefix, "python" + ext))
            candidates.append(os.path.join(prefix, "bin", "python" + ext))
            candidates.append(os.path.join(prefix, "bin", "python3" + ext))
            candidates.extend(
                sorted(glob.glob(os.path.join(prefix, "bin", "python3.*")))
            )

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    @staticmethod
    def _get_cursor_pos():
        """Get the current cursor position, or None if unavailable."""
        try:
            if sys.platform == "win32":

                class POINT(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

                pt = POINT()
                if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                    return (pt.x, pt.y)
        except Exception:
            pass
        return None

    @staticmethod
    def _hidden_startupinfo():
        """STARTUPINFO that suppresses the console window (None off Windows).

        Shared with ``package_manager``'s pip runner, which needs the
        STARTUPINFO alone (it captures the output).
        """
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            return startupinfo
        return None

    @staticmethod
    def _hidden_popen_kwargs() -> dict:
        """``Popen`` kwargs that keep a sidecar silent and windowless.

        A console interpreter (``python.exe``, ``mayapy.exe``) flashes a console
        window otherwise; both the STARTUPINFO hide and ``CREATE_NO_WINDOW`` are
        set because each covers cases the other does not.
        """
        kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kwargs["startupinfo"] = ExecutionMonitor._hidden_startupinfo()
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return kwargs

    @staticmethod
    def _helper_script_path(name):
        """Absolute path to a helper file shipped beside this module, or None."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        return path if os.path.exists(path) else None

    @staticmethod
    def _sidecar_command(command: str, *args: str):
        """``[python, _sidecar.py, command, *args]`` or ``None`` when no script
        or interpreter is available."""
        script = ExecutionMonitor._helper_script_path("_sidecar.py")
        executable = ExecutionMonitor._get_python_executable()
        if not script or not executable:
            return None
        return [executable, script, command, *args]

    @staticmethod
    def _parent_pid_arg() -> str:
        """The flag that makes a sidecar WINDOW close itself when we die."""
        return f"--parent-pid={os.getpid()}"

    @staticmethod
    def _start_indicator_process(indicator=True):
        """Start the sidecar busy indicator near the cursor; returns the ``Popen``
        or ``None``.

        Parameters:
            indicator (bool|str): True shows the canvas-drawn spinner.
                A string is treated as a path to an animated GIF (absolute,
                or relative to this package — e.g. ``"task_indicator.gif"``);
                if the file is missing, falls back to the canvas spinner.
        """
        try:
            extra = []
            if isinstance(indicator, str):
                gif_path = (
                    indicator
                    if os.path.exists(indicator)
                    else ExecutionMonitor._helper_script_path(indicator)
                )
                if gif_path:
                    extra.append(f"--gif={gif_path}")
            pos = ExecutionMonitor._get_cursor_pos()
            if pos:
                # '=' form is required: on a monitor left/above the primary the
                # coordinates are negative, and a separate "-122,-341" token is
                # parsed by argparse as an option flag (exit 2, no spinner).
                extra.append(f"--pos={pos[0]},{pos[1]}")

            cmd = ExecutionMonitor._sidecar_command(
                "indicator", ExecutionMonitor._parent_pid_arg(), *extra
            )
            if cmd is None:
                return None
            return subprocess.Popen(cmd, **ExecutionMonitor._hidden_popen_kwargs())
        except Exception:
            return None

    @staticmethod
    def _stop_indicator_process(process):
        """Stop the indicator subprocess."""
        if process:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()

    @staticmethod
    def _wait_child(process, finished=None, poll_interval: float = 0.1):
        """Wait for a sidecar; returns its exit code, or ``None`` if *finished*
        was set first — the child is then terminated, because whatever it was
        about to answer no longer applies."""
        while True:
            code = process.poll()
            if code is not None:
                return code
            if finished is None:
                time.sleep(poll_interval)
            elif finished.wait(poll_interval):
                try:
                    process.terminate()
                except Exception:
                    pass
                return None

    @staticmethod
    def _accepts_finished(callback) -> bool:
        """True if *callback* can take the ``finished`` keyword."""
        try:
            params = inspect.signature(callback).parameters
        except (TypeError, ValueError):
            return False
        return "finished" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    @staticmethod
    def _handle_callback_result(result, cancel_scope=None):
        """Act on a monitor-callback result. Returns True to stop monitoring.

        False           – request cancellation: set *cancel_scope* when one was
                          supplied (safe, cooperative), else fall back to the
                          legacy ``interrupt_main`` for callers that have not
                          adopted scopes yet.
        "FORCE_INTERRUPT" – raise SystemExit in the main thread.
        "FORCE_KILL"    – terminate the host process (last resort).
        "STOP_MONITORING" – stop silently.
        Anything else   – keep monitoring.
        """
        if result is False:
            if cancel_scope is not None:
                cancel_scope.cancel("dialog")
            else:
                _thread.interrupt_main()
            return True
        elif result == "FORCE_KILL":
            ExecutionMonitor._force_kill_process()
            return True
        elif result == "FORCE_INTERRUPT":
            ExecutionMonitor._force_interrupt_main_thread()
            return True
        elif result == "STOP_MONITORING":
            return True
        return False

    @staticmethod
    def on_long_execution(
        threshold,
        callback,
        interval=None,
        allow_escape_cancel=False,
        indicator=None,
        cancel_scope=None,
        escape_hold_seconds=0.4,
    ):
        """Decorator that triggers a callback if the decorated function takes
        longer than `threshold` seconds to execute.

        Parameters:
            threshold (float): Time in seconds before callback is triggered.
            callback (callable): Function to call if threshold is exceeded.
                Returning False requests cancellation (see
                :meth:`_handle_callback_result`). A callback that declares a
                ``finished`` parameter receives the ``threading.Event`` set
                when the function returns, so a blocking prompt can dismiss
                itself. An answer that arrives after the function finished is
                discarded either way.
            interval (float|bool, optional): If True, repeats every `threshold`
                seconds. If float, repeats every `interval` seconds.
            allow_escape_cancel (bool): If True, a sustained Escape hold (while
                this process owns the focused window) requests cancellation.
            indicator (bool|str, optional): If True, displays a spinner overlay
                near the cursor. A string is a path to an animated GIF to show
                instead. Runs in a separate process so it animates during
                blocking tasks.
            cancel_scope (CancelScope, optional): Scope to flag instead of
                interrupting the main thread. Strongly preferred — see the
                class docstring for why the interrupt path is unsafe. With a
                scope, an Esc request does not end monitoring: the flag is only
                a request, and the threshold callback is where an operation
                that ignores it gets explained (and force-stopped).
            escape_hold_seconds (float): Sustained hold required before Esc counts.
        """
        # If interval is True, use threshold as the interval
        repeat_interval = threshold if interval is True else interval
        pass_finished = ExecutionMonitor._accepts_finished(callback)

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                finished = threading.Event()

                indicator_process = None
                if indicator:
                    indicator_process = ExecutionMonitor._start_indicator_process(
                        indicator
                    )

                escape_detector = (
                    _EscapeHoldDetector(escape_hold_seconds)
                    if allow_escape_cancel
                    else None
                )

                def request_cancel(reason):
                    """Request cancellation the safest way available."""
                    if cancel_scope is not None:
                        cancel_scope.cancel(reason)
                    else:
                        # Legacy path for callers that predate CancelScope.
                        _thread.interrupt_main()

                def wait_for_stop_or_timeout(duration):
                    """Returns "stop" if the function finished, "abort" if the
                    legacy Esc interrupt was sent (once, here), or None on
                    timeout. With a scope an Esc hold only sets the flag (again
                    after a reset — the flag itself is the one-shot guard) and
                    the wait carries on."""
                    if escape_detector is None:
                        return "stop" if finished.wait(duration) else None

                    # Polling
                    remaining = duration
                    step = 0.1
                    while remaining > 0:
                        wait_time = min(step, remaining)
                        if finished.wait(wait_time):
                            return "stop"
                        if cancel_scope is None:
                            if escape_detector():
                                request_cancel("escape")
                                return "abort"
                        elif escape_detector() and not cancel_scope.cancelled:
                            # Detector first: it tracks the hold, and must see
                            # the release even while a request is pending.
                            request_cancel("escape")
                        remaining -= wait_time
                    return None

                def invoke_callback():
                    """Run the callback and act on its answer; True to stop.

                    A failing callback must not take the monitor thread (and
                    the Esc watch) down with it. An answer that arrives after
                    the function already returned is moot — acting on it would
                    cancel or interrupt whatever runs next.
                    """
                    try:
                        result = (
                            callback(finished=finished) if pass_finished else callback()
                        )
                    except Exception:
                        return False
                    if finished.is_set():
                        return True
                    return ExecutionMonitor._handle_callback_result(
                        result, cancel_scope
                    )

                def timer_func():
                    # Wait for the initial threshold
                    if wait_for_stop_or_timeout(threshold):
                        return  # Function finished, or the legacy Esc fired.

                    stopped = invoke_callback()

                    # If repeat_interval is set, keep repeating
                    while not stopped and repeat_interval:
                        if wait_for_stop_or_timeout(repeat_interval):
                            return
                        stopped = invoke_callback()

                    # Monitoring is over, but the Esc-cancel contract lasts for
                    # the whole execution — keep polling until the function ends.
                    if escape_detector is not None:
                        while wait_for_stop_or_timeout(60.0) is None:
                            pass

                t = threading.Thread(target=timer_func, name="ExecutionMonitor")
                t.daemon = True
                t.start()

                try:
                    result = func(*args, **kwargs)
                finally:
                    finished.set()
                    if indicator_process:
                        ExecutionMonitor._stop_indicator_process(indicator_process)
                return result

            return wrapper

        return decorator

    @staticmethod
    def _run_dialog_sidecar(title, message, force_label=None, finished=None):
        """Run the Tk dialog sidecar; returns its exit code, or ``None`` when it
        is unavailable, crashed, or was dismissed because *finished* fired."""
        try:
            extra = [ExecutionMonitor._parent_pid_arg()]
            if force_label:
                extra.append(f"--force-label={force_label}")
            cmd = ExecutionMonitor._sidecar_command(
                "dialog", *extra, "--", title, message
            )
            if cmd is None:
                return None
            process = subprocess.Popen(cmd, **ExecutionMonitor._hidden_popen_kwargs())
            code = ExecutionMonitor._wait_child(process, finished)
        except Exception:
            return None
        known = (
            _sidecar.DIALOG_KEEP_WAITING,
            _sidecar.DIALOG_CANCEL,
            _sidecar.DIALOG_FORCE,
            _sidecar.DIALOG_CLOSED,
        )
        return code if code in known else None

    @staticmethod
    def show_long_execution_dialog(title, message, force_action=None, finished=None):
        """Show a dialog to ask the user how to proceed with a long operation.

        The Tk sidecar dialog (custom button labels, VS Code style) is the
        primary path on every platform; native fallbacks when it cannot run:
        ``MessageBoxW`` on Windows, zenity / kdialog on Linux.

        Parameters:
            title (str): Dialog window title.
            message (str): Body text.
            force_action (str|None): Controls the force button.
                ``None``  – no force button (default, 2-button dialog).
                ``"interrupt"`` – show *Force Stop* (raises SystemExit in main thread).
                ``"kill"``      – show *Force Quit* (terminates the host process).
            finished (threading.Event|None): Set when the operation ends; the
                sidecar dialog is then dismissed and ``"STOP_MONITORING"``
                returned — there is nothing left to ask. The native fallbacks
                block and cannot be dismissed early.

        Returns:
            bool/str: True to continue waiting, False to abort,
                ``"FORCE_INTERRUPT"`` or ``"FORCE_KILL"`` for the force action,
                ``"STOP_MONITORING"`` if the operation finished first.
        """
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

        if finished is not None and finished.is_set():
            return "STOP_MONITORING"  # nothing left to ask
        code = ExecutionMonitor._run_dialog_sidecar(
            title, message, force_label, finished
        )
        if code in (_sidecar.DIALOG_KEEP_WAITING, _sidecar.DIALOG_CLOSED):
            return True
        if code == _sidecar.DIALOG_CANCEL:
            return False
        if code == _sidecar.DIALOG_FORCE:
            return force_sentinel or True
        if finished is not None and finished.is_set():
            return "STOP_MONITORING"

        if sys.platform == "win32":
            try:
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
            except Exception:
                return True

        elif sys.platform.startswith("linux"):
            # NOTE: do not re-import subprocess here — a local import would
            # shadow the module-level one for the WHOLE function, making the
            # sidecar branch above raise UnboundLocalError (silently caught, so
            # the custom dialog never showed).
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
        cancel_scope=None,
        escape_hold_seconds: float = 0.4,
    ):
        """Decorator that monitors execution time and (optionally) prompts the
        user via a native dialog if the threshold is exceeded. Can also
        (optionally) enable an external heartbeat watchdog that can force-kill
        the process if the host application hard-hangs.

        Parameters:
            threshold (float): Time in seconds before warning.
            message (str): Message to display in the dialog/logs.
            logger (logging.Logger, optional): Logger to use for status updates.
            allow_escape_cancel (bool): If True, a sustained Esc hold requests
                cancellation (flags *cancel_scope*; interrupts the main thread
                only on the legacy no-scope path).
            show_dialog (bool): If False, do not show a blocking dialog; only log warnings.
            force_action (str|None): Controls the force button in the dialog.
                ``None`` (default) – no force button; dialog shows only Keep Waiting / Cancel.
                ``"interrupt"``    – adds a *Force Stop* button that raises ``SystemExit`` in
                                     the main thread without killing the host process.
                ``"kill"``         – adds a *Force Quit* button that terminates the entire
                                     host process (use as a last resort).
            watchdog_timeout (float|None): If set, starts an external watchdog that kills this process
                if the heartbeat stalls for longer than this many seconds. See
                :meth:`external_watchdog` for what "stalls" can mean.
            watchdog_heartbeat_interval (float): Heartbeat write interval in seconds.
            watchdog_check_interval (float): Watchdog polling interval in seconds.
            watchdog_kill_tree (bool): If True, attempt to kill child processes too.
            watchdog_heartbeat_path (str|None): Optional heartbeat file path override.
            indicator (bool|str, optional): If True, displays a spinner overlay near the cursor;
                a string is a path to an animated GIF to show instead.
            cancel_scope (CancelScope, optional): Scope flagged by the dialog's *Cancel*
                button and by Esc, instead of interrupting the main thread. Also lets
                the dialog tell the truth about whether cancelling can take effect.
            escape_hold_seconds (float): Sustained Esc hold required to request cancel.
        """

        _dialog_shown = [False]

        def callback(finished=None):
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
            # Be honest about what Cancel can do. An operation that has never
            # reached a checkpoint has no cooperative point to stop at, so a
            # cancel request would sit unconsumed — saying "cancelled" there
            # would be a lie the user acts on.
            if cancel_scope is not None and not cancel_scope.has_ticked:
                effect_hint = (
                    "\n\nNote: this operation has not reported any cancellable "
                    "points yet, so Cancel will only take effect if and when it "
                    "reaches one."
                )
            else:
                effect_hint = ""
            result = ExecutionMonitor.show_long_execution_dialog(
                "Long Execution Warning",
                f"{full_msg}\n\nThe operation is not responding.\n"
                "You can keep waiting or cancel the operation."
                f"{esc_hint}{effect_hint}",
                force_action=force_action,
                finished=finished,
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
                elif result == "STOP_MONITORING":
                    logger.debug("Operation finished while the dialog was open.")

            return result

        def decorator(func):
            monitored = ExecutionMonitor.on_long_execution(
                threshold,
                callback,
                interval=True,
                allow_escape_cancel=allow_escape_cancel,
                indicator=indicator,
                cancel_scope=cancel_scope,
                escape_hold_seconds=escape_hold_seconds,
            )(func)

            if watchdog_timeout is not None:
                # Layer the external watchdog outside the timer-based monitor.
                monitored = ExecutionMonitor.external_watchdog(
                    timeout=float(watchdog_timeout),
                    message=message,
                    heartbeat_interval=float(watchdog_heartbeat_interval),
                    check_interval=float(watchdog_check_interval),
                    kill_tree=bool(watchdog_kill_tree),
                    logger=logger,
                    heartbeat_path=watchdog_heartbeat_path,
                )(monitored)

            @wraps(func)
            def wrapper(*args, **kwargs):
                # Re-arm the once-per-run dialog guard for THIS invocation;
                # without this the dialog only ever showed once per session.
                _dialog_shown[0] = False
                return monitored(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def _default_heartbeat_path(tag: str = "watchdog") -> str:
        """A heartbeat file path in the ``execution_monitor_`` temp namespace.

        Allocated through ``TempArtifacts`` rather than a bare temp path: when
        the watchdog fires, this process is killed and no ``finally`` runs, so
        the file's only reclamation is the next run's age-gated sweep.
        """
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        safe_tag = "".join(
            c if c.isalnum() or c in ("-", "_", ".") else "_" for c in (tag or "")
        )
        # An hour: a live heartbeat is rewritten every second, and a watchdog
        # timeout is seconds to minutes — anything older is a leftover.
        store = TempArtifacts(
            ExecutionMonitor._TEMP_PREFIX, policy="detached", max_age_days=1 / 24
        )
        return store.path(".txt", name=f"{safe_tag}_hb_{os.getpid()}")

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
            # Join first — a still-running writer could recreate the file
            # right after we delete it.
            t.join(timeout=2.0)
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
        """Spawn the sidecar watchdog that kills `pid` if the heartbeat stalls.

        Runs under :meth:`_get_python_executable`, never ``sys.executable``:
        inside Maya that is ``maya.exe``, and the safety valve for a hung Maya
        would have launched a second Maya.
        """
        stop_file = heartbeat_path + ".stop"
        args = [
            str(int(pid)),
            str(heartbeat_path),
            f"--timeout={float(timeout)}",
            f"--check-interval={float(check_interval)}",
            "--stop-file",
            str(stop_file),
        ]
        if kill_tree:
            args.append("--kill-tree")
        cmd = ExecutionMonitor._sidecar_command("watchdog", *args)
        if cmd is None:
            if logger:
                logger.warning(
                    "External watchdog skipped: no python interpreter resolved "
                    "(see ExecutionMonitor.set_interpreter)."
                )
            return None, None

        try:
            proc = subprocess.Popen(cmd, **ExecutionMonitor._hidden_popen_kwargs())
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
                    proc.wait(timeout=1)
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
            - The heartbeat is written by a Python thread, so a long native call
              that holds the GIL (many DCC commands do) starves it exactly like a
              hang would. Size `timeout` for the longest legitimate native call,
              not for "how long a hang takes to notice".
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
