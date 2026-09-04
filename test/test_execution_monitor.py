import unittest
import time
import threading
import sys
import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch
from pythontk.core_utils.execution_monitor._execution_monitor import ExecutionMonitor
from pythontk import ExecutionMonitor as PublicExecutionMonitor
from pythontk import CancelScope

from conftest import BaseTestCase

_SIDECAR = os.path.join(
    os.path.dirname(os.path.abspath(sys.modules[ExecutionMonitor.__module__].__file__)),
    "_sidecar.py",
)
_HAS_DISPLAY = sys.platform == "win32" or bool(os.environ.get("DISPLAY"))


class TestExecutionMonitor(BaseTestCase):
    def test_on_long_execution_fast_function(self):
        """Test that callback is not triggered for fast functions."""
        callback = MagicMock()

        @ExecutionMonitor.on_long_execution(threshold=0.5, callback=callback)
        def fast_func():
            return "success"

        result = fast_func()
        self.assertEqual(result, "success")
        callback.assert_not_called()

    def test_on_long_execution_slow_function(self):
        """Test that callback is triggered for slow functions."""
        callback = MagicMock(return_value=True)

        @ExecutionMonitor.on_long_execution(threshold=0.1, callback=callback)
        def slow_func():
            time.sleep(0.3)
            return "success"

        result = slow_func()
        self.assertEqual(result, "success")
        # Callback should have been called at least once
        self.assertTrue(callback.called)

    def test_on_long_execution_interval(self):
        """Test that callback is triggered repeatedly with interval."""
        callback = MagicMock(return_value=True)

        # Threshold 0.1, Interval 0.1
        @ExecutionMonitor.on_long_execution(
            threshold=0.1, callback=callback, interval=0.1
        )
        def very_slow_func():
            time.sleep(0.35)
            return "success"

        result = very_slow_func()
        self.assertEqual(result, "success")
        # Should be called roughly 3 times (0.1, 0.2, 0.3)
        self.assertGreaterEqual(callback.call_count, 2)

    def test_on_long_execution_no_interval(self):
        """Test that callback is triggered only once if interval is not set."""
        callback = MagicMock(return_value=True)

        # Threshold 0.1, No interval
        @ExecutionMonitor.on_long_execution(
            threshold=0.1, callback=callback, interval=None
        )
        def long_func_no_interval():
            time.sleep(0.35)  # Sleep 3.5x threshold
            return "success"

        result = long_func_no_interval()
        self.assertEqual(result, "success")
        # Should be called exactly once
        self.assertEqual(callback.call_count, 1)

    def test_on_long_execution_abort(self):
        """Test that returning False from callback raises KeyboardInterrupt."""
        # Callback returns False to signal abort
        callback = MagicMock(return_value=False)

        @ExecutionMonitor.on_long_execution(threshold=0.1, callback=callback)
        def abortable_func():
            try:
                # Sleep long enough to trigger callback
                time.sleep(3.0)
            except KeyboardInterrupt:
                return "interrupted"
            return "finished"

        # We need to be careful here because _thread.interrupt_main() schedules an exception
        # in the main thread. The sleep above should be interrupted.
        result = abortable_func()
        self.assertEqual(result, "interrupted")
        callback.assert_called()

    def test_on_long_execution_stop_monitoring(self):
        """Test that returning 'STOP_MONITORING' stops further callbacks."""
        # First call returns STOP_MONITORING, subsequent calls shouldn't happen
        callback = MagicMock(side_effect=["STOP_MONITORING", True, True])

        @ExecutionMonitor.on_long_execution(
            threshold=0.1, callback=callback, interval=0.1
        )
        def monitored_func():
            time.sleep(0.4)
            return "success"

        result = monitored_func()
        self.assertEqual(result, "success")
        # Should be called exactly once
        self.assertEqual(callback.call_count, 1)

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.is_foreground_process",
        return_value=True,
    )
    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.is_escape_pressed"
    )
    def test_on_long_execution_escape_cancel_legacy_interrupt(
        self, mock_is_escape, _mock_fg
    ):
        """Without a scope, Esc still falls back to the legacy interrupt.

        Back-compat only: callers that have not adopted ``CancelScope`` keep the
        old (unsafe) behaviour rather than silently losing cancellation.
        """
        mock_is_escape.side_effect = [False] * 5 + [True] * 100

        callback = MagicMock()

        @ExecutionMonitor.on_long_execution(
            threshold=0.5,
            callback=callback,
            allow_escape_cancel=True,
            escape_hold_seconds=0,
        )
        def escape_func():
            try:
                # Sleep in small chunks to allow interrupt to happen
                for _ in range(50):
                    time.sleep(0.1)
            except KeyboardInterrupt:
                return "interrupted"
            return "finished"

        result = escape_func()
        self.assertEqual(result, "interrupted")
        # Callback shouldn't be called because we interrupted before threshold
        callback.assert_not_called()

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.is_foreground_process",
        return_value=True,
    )
    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.is_escape_pressed"
    )
    def test_escape_cancels_scope_without_interrupting_main(
        self, mock_is_escape, _mock_fg
    ):
        """With a scope, Esc flags it — and never injects an exception."""
        mock_is_escape.return_value = True
        scope = CancelScope("test")
        callback = MagicMock()

        @ExecutionMonitor.on_long_execution(
            threshold=5.0,
            callback=callback,
            allow_escape_cancel=True,
            escape_hold_seconds=0,
            cancel_scope=scope,
        )
        def cooperative_func():
            # Cooperative consumer: notices the flag at its own checkpoint.
            for _ in range(50):
                if not scope.tick():
                    return "cancelled"
                time.sleep(0.05)
            return "finished"

        result = cooperative_func()
        self.assertEqual(result, "cancelled")
        self.assertTrue(scope.cancelled)
        self.assertEqual(scope.reason, "escape")

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.is_escape_pressed"
    )
    def test_escape_ignored_when_not_foreground(self, mock_is_escape):
        """Esc pressed while another app has focus must not cancel."""
        mock_is_escape.return_value = True
        scope = CancelScope("test")

        with patch.object(
            ExecutionMonitor, "is_foreground_process", return_value=False
        ):

            @ExecutionMonitor.on_long_execution(
                threshold=5.0,
                callback=MagicMock(),
                allow_escape_cancel=True,
                escape_hold_seconds=0,
                cancel_scope=scope,
            )
            def func():
                time.sleep(0.5)
                return "finished"

            self.assertEqual(func(), "finished")
        self.assertFalse(scope.cancelled)

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.is_foreground_process",
        return_value=True,
    )
    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.is_escape_pressed"
    )
    def test_escape_requires_sustained_hold(self, mock_is_escape, _mock_fg):
        """A momentary Esc tap (its dozen other DCC meanings) must not cancel."""
        # Pressed for one sample, then released for the rest.
        mock_is_escape.side_effect = [True] + [False] * 200
        scope = CancelScope("test")

        @ExecutionMonitor.on_long_execution(
            threshold=5.0,
            callback=MagicMock(),
            allow_escape_cancel=True,
            escape_hold_seconds=1.0,
            cancel_scope=scope,
        )
        def func():
            time.sleep(0.6)
            return "finished"

        self.assertEqual(func(), "finished")
        self.assertFalse(scope.cancelled)

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.show_long_execution_dialog"
    )
    def test_dialog_cancel_flags_scope_not_interrupt(self, mock_dialog):
        """The dialog's Cancel button sets the scope; no async exception."""
        mock_dialog.return_value = False  # Cancel
        scope = CancelScope("test")

        @ExecutionMonitor.execution_monitor(
            threshold=0.1, message="Testing", cancel_scope=scope
        )
        def monitored_func():
            for _ in range(50):
                if not scope.tick():
                    return "cancelled"
                time.sleep(0.05)
            return "finished"

        self.assertEqual(monitored_func(), "cancelled")
        self.assertEqual(scope.reason, "dialog")

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.show_long_execution_dialog"
    )
    def test_dialog_warns_when_operation_has_no_checkpoints(self, mock_dialog):
        """An operation that never ticks must not imply Cancel will work."""
        mock_dialog.return_value = True  # Keep waiting
        scope = CancelScope("monolith")

        @ExecutionMonitor.execution_monitor(
            threshold=0.1, message="Testing", cancel_scope=scope
        )
        def monolithic_func():
            time.sleep(0.5)  # never reaches a checkpoint
            return "done"

        monolithic_func()
        mock_dialog.assert_called()
        body = mock_dialog.call_args[0][1]
        self.assertIn("has not reported any cancellable", body)

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.show_long_execution_dialog"
    )
    def test_execution_monitor_decorator(self, mock_dialog):
        """Test the high-level execution_monitor decorator."""
        mock_dialog.return_value = True  # Continue waiting
        logger = MagicMock()

        @ExecutionMonitor.execution_monitor(
            threshold=0.1, message="Testing", logger=logger
        )
        def monitored_func():
            time.sleep(2.0)
            return "done"

        result = monitored_func()
        self.assertEqual(result, "done")
        mock_dialog.assert_called()
        logger.warning.assert_called()
        logger.info.assert_called_with("Continuing execution (Keep Waiting).")

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.show_long_execution_dialog"
    )
    def test_execution_monitor_abort(self, mock_dialog):
        """Test execution_monitor abort behavior."""
        mock_dialog.return_value = False  # Abort
        logger = MagicMock()

        @ExecutionMonitor.execution_monitor(
            threshold=0.1, message="Testing", logger=logger
        )
        def monitored_func():
            try:
                time.sleep(3.0)
            except KeyboardInterrupt:
                return "interrupted"
            return "finished"

        result = monitored_func()
        self.assertEqual(result, "interrupted")
        logger.warning.assert_any_call("Operation cancelled by user.")

    def test_is_escape_pressed_windows(self):
        """Test is_escape_pressed on Windows (mocked)."""
        with patch("sys.platform", "win32"):
            with patch("ctypes.windll.user32.GetAsyncKeyState") as mock_get_key:
                # Case 1: Key pressed (Most significant bit set)
                mock_get_key.return_value = 0x8000
                self.assertTrue(ExecutionMonitor.is_escape_pressed())

                # Case 2: Key not pressed
                mock_get_key.return_value = 0
                self.assertFalse(ExecutionMonitor.is_escape_pressed())

    # Patch target for the module-under-test's own references.
    _EM_MOD = "pythontk.core_utils.execution_monitor._execution_monitor"

    def _dialog_subprocess_mock(self, returncode):
        """A stand-in for the em-module's ``subprocess`` whose ``Popen`` reports
        *returncode* from the sidecar dialog. A full MagicMock also supplies
        STARTUPINFO/flags, so the primary path is exercised even on non-Windows
        CI (where the real subprocess module lacks them)."""
        sp = MagicMock()
        proc = sp.Popen.return_value
        proc.poll.return_value = returncode
        proc.returncode = returncode
        # ``communicate`` is only reached by the native fallbacks; keep it inert.
        proc.communicate.return_value = ("", "")
        return sp

    def _no_sidecar(self):
        """Pin the NATIVE fallback branch: make the sidecar script 'missing'.

        The sidecar dialog is the primary path on every platform; the zenity /
        kdialog / MessageBoxW tests below are about the fallbacks and must not
        depend on whether this machine can run Tk."""
        return patch(
            f"{self._EM_MOD}.ExecutionMonitor._helper_script_path", return_value=None
        )

    def test_show_long_execution_dialog_windows_custom_dialog(self):
        """The custom dialog viewer (primary win32 path) maps exit codes to
        results: 0/3 -> keep waiting, 10 -> cancel, 2 -> force sentinel.

        Regression: a local ``import subprocess`` in the linux branch used to
        shadow the module-level import for the WHOLE function, so this path
        always died with UnboundLocalError (silently caught) and fell through
        to MessageBoxW — the custom dialog never showed on Windows.
        """
        with patch("sys.platform", "win32"):
            with patch("ctypes.windll.user32.MessageBoxW") as mock_msg_box:
                mock_msg_box.return_value = 6  # would mean True via fallback
                with patch(
                    f"{self._EM_MOD}.subprocess", self._dialog_subprocess_mock(10)
                ):
                    # rc 10 = Cancel. The fallback would have returned True —
                    # False proves the primary path produced the answer.
                    self.assertFalse(
                        ExecutionMonitor.show_long_execution_dialog("Title", "Msg")
                    )
                    mock_msg_box.assert_not_called()
                with patch(
                    f"{self._EM_MOD}.subprocess", self._dialog_subprocess_mock(0)
                ):
                    self.assertTrue(
                        ExecutionMonitor.show_long_execution_dialog("Title", "Msg")
                    )
                with patch(
                    f"{self._EM_MOD}.subprocess", self._dialog_subprocess_mock(2)
                ):
                    self.assertEqual(
                        ExecutionMonitor.show_long_execution_dialog(
                            "Title", "Msg", force_action="kill"
                        ),
                        "FORCE_KILL",
                    )

    def _force_messagebox_fallback(self):
        """Make the custom-dialog script 'missing' so the win32 branch takes
        the MessageBoxW fallback (the sidecar dialog is the primary path)."""
        return self._no_sidecar()

    def test_show_long_execution_dialog_windows(self):
        """Test the MessageBoxW fallback on Windows — no force button by default."""
        with patch("sys.platform", "win32"):
            with self._force_messagebox_fallback():
                with patch("ctypes.windll.user32.MessageBoxW") as mock_msg_box:
                    # Default (force_action=None) uses MB_YESNO (no Cancel button)
                    # IDYES=6 -> True
                    mock_msg_box.return_value = 6
                    self.assertTrue(
                        ExecutionMonitor.show_long_execution_dialog("Title", "Msg")
                    )

                    # IDNO=7 -> False
                    mock_msg_box.return_value = 7
                    self.assertFalse(
                        ExecutionMonitor.show_long_execution_dialog("Title", "Msg")
                    )

    def test_show_long_execution_dialog_windows_force_kill(self):
        """Test the MessageBoxW fallback with force_action='kill' returns FORCE_KILL."""
        with patch("sys.platform", "win32"):
            with self._force_messagebox_fallback():
                with patch("ctypes.windll.user32.MessageBoxW") as mock_msg_box:
                    # IDCANCEL=2 -> "FORCE_KILL"
                    mock_msg_box.return_value = 2
                    self.assertEqual(
                        ExecutionMonitor.show_long_execution_dialog(
                            "Title", "Msg", force_action="kill"
                        ),
                        "FORCE_KILL",
                    )

    def test_show_long_execution_dialog_windows_force_interrupt(self):
        """Test the MessageBoxW fallback with force_action='interrupt' returns FORCE_INTERRUPT."""
        with patch("sys.platform", "win32"):
            with self._force_messagebox_fallback():
                with patch("ctypes.windll.user32.MessageBoxW") as mock_msg_box:
                    # IDCANCEL=2 -> "FORCE_INTERRUPT"
                    mock_msg_box.return_value = 2
                    self.assertEqual(
                        ExecutionMonitor.show_long_execution_dialog(
                            "Title", "Msg", force_action="interrupt"
                        ),
                        "FORCE_INTERRUPT",
                    )

    def test_is_escape_pressed_linux(self):
        """Test is_escape_pressed on Linux (mocked)."""
        with patch("sys.platform", "linux"):
            with patch("ctypes.cdll.LoadLibrary") as mock_load_lib:
                mock_x11 = MagicMock()
                mock_load_lib.return_value = mock_x11

                # Setup X11 mocks
                mock_x11.XOpenDisplay.return_value = 1  # Valid display
                mock_x11.XKeysymToKeycode.return_value = (
                    9  # Keycode for Escape (example)
                )

                # Mock XQueryKeymap to return a keymap where the bit for keycode 9 is set
                # Keycode 9 -> Byte 1 (9 // 8), Bit 1 (9 % 8)
                # We need to populate the buffer passed to XQueryKeymap
                def side_effect_query_keymap(display, keys_buffer):
                    # keys_buffer is a c_char * 32
                    # We want to set the bit at index 1
                    # In Python ctypes array, we can set values by index
                    # 1 << 1 = 2
                    keys_buffer[1] = b"\x02"

                mock_x11.XQueryKeymap.side_effect = side_effect_query_keymap

                # Reset static variables in ExecutionMonitor to force re-initialization
                ExecutionMonitor._x11_lib = None
                ExecutionMonitor._x11_display = None

                self.assertTrue(ExecutionMonitor.is_escape_pressed())

                # Test not pressed
                def side_effect_query_keymap_empty(display, keys_buffer):
                    keys_buffer[1] = b"\x00"

                mock_x11.XQueryKeymap.side_effect = side_effect_query_keymap_empty

                self.assertFalse(ExecutionMonitor.is_escape_pressed())

    def test_show_long_execution_dialog_linux_zenity(self):
        """Test show_long_execution_dialog on Linux using Zenity (mocked)."""
        with patch("sys.platform", "linux"), self._no_sidecar():
            with patch("shutil.which") as mock_which:
                with patch("subprocess.run") as mock_run:
                    # Case 1: Zenity available
                    mock_which.side_effect = lambda x: (
                        "/usr/bin/zenity" if x == "zenity" else None
                    )

                    # Zenity returns 0 (OK) -> True
                    mock_run.return_value = MagicMock(returncode=0, stdout="")
                    self.assertTrue(
                        ExecutionMonitor.show_long_execution_dialog("Title", "Msg")
                    )

                    # Zenity returns 1 (Cancel) -> False
                    mock_run.return_value = MagicMock(returncode=1, stdout="")
                    self.assertFalse(
                        ExecutionMonitor.show_long_execution_dialog("Title", "Msg")
                    )

                    # With force_action='interrupt', extra button returns "FORCE_INTERRUPT"
                    mock_process = MagicMock()
                    mock_process.returncode = 0
                    mock_process.stdout = "Force Stop\n"
                    mock_run.return_value = mock_process

                    self.assertEqual(
                        ExecutionMonitor.show_long_execution_dialog(
                            "Title", "Msg", force_action="interrupt"
                        ),
                        "FORCE_INTERRUPT",
                    )

                    # With force_action='kill', extra button returns "FORCE_KILL"
                    mock_process.stdout = "Force Quit\n"
                    mock_run.return_value = mock_process

                    self.assertEqual(
                        ExecutionMonitor.show_long_execution_dialog(
                            "Title", "Msg", force_action="kill"
                        ),
                        "FORCE_KILL",
                    )

    def test_show_long_execution_dialog_linux_kdialog(self):
        """Test show_long_execution_dialog on Linux using KDialog (mocked)."""
        with patch("sys.platform", "linux"), self._no_sidecar():
            with patch("shutil.which") as mock_which:
                with patch("subprocess.run") as mock_run:
                    # Case 1: Zenity NOT available, KDialog available
                    mock_which.side_effect = lambda x: (
                        "/usr/bin/kdialog" if x == "kdialog" else None
                    )

                    # KDialog returns 0 (Yes) -> True
                    mock_run.return_value = MagicMock(returncode=0)
                    self.assertTrue(
                        ExecutionMonitor.show_long_execution_dialog("Title", "Msg")
                    )

                    # KDialog returns 1 (No) -> False
                    mock_run.return_value = MagicMock(returncode=1)
                    self.assertFalse(
                        ExecutionMonitor.show_long_execution_dialog("Title", "Msg")
                    )

                    # With force_action='kill': KDialog returns 2 (Cancel) -> "FORCE_KILL"
                    mock_run.return_value = MagicMock(returncode=2)
                    self.assertEqual(
                        ExecutionMonitor.show_long_execution_dialog(
                            "Title", "Msg", force_action="kill"
                        ),
                        "FORCE_KILL",
                    )

    def test_on_long_execution_exception_propagation(self):
        """Test that exceptions in the monitored function are propagated."""
        callback = MagicMock()

        @ExecutionMonitor.on_long_execution(threshold=0.1, callback=callback)
        def error_func():
            raise ValueError("Something went wrong")

        with self.assertRaises(ValueError):
            error_func()

        # Callback shouldn't be called as execution was fast (immediate error)
        callback.assert_not_called()

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.is_escape_pressed"
    )
    def test_on_long_execution_escape_ignored(self, mock_is_escape):
        """Test that holding Escape is ignored if allow_escape_cancel is False."""
        # Mock escape being pressed
        mock_is_escape.return_value = True

        callback = MagicMock(return_value=True)

        @ExecutionMonitor.on_long_execution(
            threshold=0.1, callback=callback, allow_escape_cancel=False
        )
        def ignore_escape_func():
            time.sleep(0.2)
            return "finished"

        result = ignore_escape_func()
        self.assertEqual(result, "finished")
        # Callback should be called because of timeout, not escape
        self.assertTrue(callback.called)

    def test_on_long_execution_slow_exception(self):
        """Test that exceptions in a slow monitored function are propagated and monitor stops."""
        callback = MagicMock(return_value=True)

        @ExecutionMonitor.on_long_execution(threshold=0.1, callback=callback)
        def slow_error_func():
            time.sleep(0.2)
            raise ValueError("Slow error")

        with self.assertRaises(ValueError):
            slow_error_func()

        # Callback should have been called
        self.assertTrue(callback.called)

    def test_on_long_execution_blocking_on_thread_join(self):
        """Test that monitor can interrupt a main thread blocked on thread.join()."""
        callback = MagicMock(return_value=False)

        @ExecutionMonitor.on_long_execution(threshold=0.1, callback=callback)
        def blocking_func():
            def worker():
                time.sleep(1.0)

            t = threading.Thread(target=worker)
            t.start()
            try:
                t.join()  # This blocks main thread
            except KeyboardInterrupt:
                return "interrupted"
            return "finished"

        result = blocking_func()
        self.assertEqual(result, "interrupted")
        callback.assert_called()

    def test_on_long_execution_from_thread_interrupts_main(self):
        """Test that using the monitor in a thread interrupts the main thread (limitation of _thread.interrupt_main)."""
        callback = MagicMock(return_value=False)  # Return False to trigger interrupt

        @ExecutionMonitor.on_long_execution(threshold=0.1, callback=callback)
        def thread_func():
            time.sleep(0.5)
            return "finished"

        # We need to coordinate to catch the interrupt in the main thread
        interrupt_caught = threading.Event()

        def run_thread():
            thread_func()

        t = threading.Thread(target=run_thread)
        t.start()

        try:
            # Sleep in main thread to allow interrupt to be received
            # We need to sleep longer than the threshold
            time.sleep(1.0)
        except KeyboardInterrupt:
            interrupt_caught.set()

        t.join()

        self.assertTrue(
            interrupt_caught.is_set(), "Main thread should have been interrupted"
        )
        callback.assert_called()

    def test_default_heartbeat_path_sanitizes_tag(self):
        """The heartbeat file is a TempArtifacts allocation: it lives in the
        ``execution_monitor_`` prefix namespace so a crash leftover (the
        watchdog killed us, so no ``finally`` ran) is reclaimed by the next
        run's stale sweep instead of leaking forever."""
        with tempfile.TemporaryDirectory() as td:
            with patch("tempfile.gettempdir", return_value=td):
                p = ExecutionMonitor._default_heartbeat_path("a/b:{c}\\d")
                self.assertEqual(os.path.dirname(p), td)
                self.assertTrue(os.path.basename(p).startswith("execution_monitor_"), p)
                self.assertIn(f"hb_{os.getpid()}.txt", p)
                self.assertNotIn("/", os.path.basename(p))
                self.assertNotIn(":", os.path.basename(p))

    def test_start_heartbeat_writer_writes_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as td:
            hb = os.path.join(td, "hb.txt")
            stop = ExecutionMonitor._start_heartbeat_writer(hb, interval=0.05)

            try:
                time.sleep(0.2)
                self.assertTrue(os.path.exists(hb))
                first_mtime = os.path.getmtime(hb)
                time.sleep(0.2)
                second_mtime = os.path.getmtime(hb)
                self.assertGreaterEqual(second_mtime, first_mtime)
            finally:
                stop()

            self.assertFalse(os.path.exists(hb))

    def test_spawn_watchdog_subprocess_builds_args_and_stop(self):
        with tempfile.TemporaryDirectory() as td:
            hb = os.path.join(td, "hb.txt")
            # Create heartbeat so watchdog doesn't immediately consider it missing.
            with open(hb, "w", encoding="utf-8") as f:
                f.write("0")

            fake_proc = MagicMock()
            fake_proc.poll.return_value = None

            with patch("subprocess.Popen", return_value=fake_proc) as popen:
                proc, stop = ExecutionMonitor._spawn_watchdog_subprocess(
                    pid=1234,
                    heartbeat_path=hb,
                    timeout=5.0,
                    check_interval=0.5,
                    kill_tree=True,
                )

                self.assertIs(proc, fake_proc)
                self.assertIsNotNone(stop)

                # Verify Popen called with the sidecar watchdog and our args
                popen.assert_called()
                called_args, called_kwargs = popen.call_args
                argv = called_args[0]
                self.assertEqual(argv[0], ExecutionMonitor._get_python_executable())
                self.assertTrue(argv[1].endswith("_sidecar.py"), argv)
                self.assertEqual(argv[2], "watchdog")
                self.assertEqual(argv[3], "1234")
                self.assertEqual(argv[4], hb)
                self.assertIn("--timeout=5.0", argv)
                self.assertIn("--check-interval=0.5", argv)
                self.assertIn("--kill-tree", argv)
                stop_file = argv[argv.index("--stop-file") + 1]
                self.assertTrue(stop_file.endswith(".stop"))

                # stop() should write stop file, terminate process, and cleanup stop file
                self.assertFalse(os.path.exists(stop_file))
                stop()
                fake_proc.terminate.assert_called()
                self.assertFalse(os.path.exists(stop_file))

    def test_spawn_watchdog_subprocess_windows_sets_no_window(self):
        with tempfile.TemporaryDirectory() as td:
            hb = os.path.join(td, "hb.txt")
            with open(hb, "w", encoding="utf-8") as f:
                f.write("0")

            fake_proc = MagicMock()
            fake_proc.poll.return_value = None

            with patch("sys.platform", "win32"):
                with patch("subprocess.CREATE_NO_WINDOW", 123, create=True):
                    with patch("subprocess.STARTUPINFO", MagicMock(), create=True):
                        with patch("subprocess.STARTF_USESHOWWINDOW", 1, create=True):
                            with patch("subprocess.SW_HIDE", 0, create=True):
                                with patch(
                                    "subprocess.Popen", return_value=fake_proc
                                ) as popen:
                                    ExecutionMonitor._spawn_watchdog_subprocess(
                                        pid=1234,
                                        heartbeat_path=hb,
                                        timeout=5.0,
                                        check_interval=0.5,
                                        kill_tree=False,
                                    )
                                    _, kwargs = popen.call_args
                                    self.assertEqual(kwargs.get("creationflags"), 123)
                                    self.assertIn("startupinfo", kwargs)

    def test_external_watchdog_decorator_starts_and_stops(self):
        stop_hb = MagicMock()
        stop_wd = MagicMock()
        proc = MagicMock()

        with patch(
            "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor._start_heartbeat_writer",
            return_value=stop_hb,
        ) as start_hb:
            with patch(
                "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor._spawn_watchdog_subprocess",
                return_value=(proc, stop_wd),
            ) as spawn_wd:

                @ExecutionMonitor.external_watchdog(timeout=1.0, heartbeat_interval=0.1)
                def fn():
                    return 123

                self.assertEqual(fn(), 123)
                start_hb.assert_called()
                spawn_wd.assert_called()
                stop_hb.assert_called()
                stop_wd.assert_called()

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.show_long_execution_dialog"
    )
    def test_execution_monitor_no_dialog_logs_only(self, mock_dialog):
        """When show_dialog=False, no UI is shown but warnings still log."""
        logger = MagicMock()

        @ExecutionMonitor.execution_monitor(
            threshold=0.1,
            message="Testing",
            logger=logger,
            show_dialog=False,
        )
        def monitored_func():
            time.sleep(0.25)
            return "done"

        self.assertEqual(monitored_func(), "done")
        mock_dialog.assert_not_called()
        self.assertTrue(logger.warning.called)

    def test_execution_monitor_watchdog_timeout_wraps(self):
        """When watchdog_timeout is set, the external watchdog is enabled."""
        stop_hb = MagicMock()
        stop_wd = MagicMock()
        proc = MagicMock()

        with patch(
            "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor._start_heartbeat_writer",
            return_value=stop_hb,
        ) as start_hb:
            with patch(
                "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor._spawn_watchdog_subprocess",
                return_value=(proc, stop_wd),
            ) as spawn_wd:

                @ExecutionMonitor.execution_monitor(
                    threshold=0.1,
                    message="Testing",
                    logger=None,
                    show_dialog=False,
                    watchdog_timeout=2.0,
                    watchdog_heartbeat_interval=0.1,
                    watchdog_check_interval=0.2,
                )
                def monitored_func():
                    return "ok"

                self.assertEqual(monitored_func(), "ok")
                start_hb.assert_called()
                spawn_wd.assert_called()
                stop_hb.assert_called()
                stop_wd.assert_called()

    def test_force_stop_does_not_terminate_process(self):
        """Verify Force Stop raises SystemExit in main thread instead of killing the process.

        Bug: _force_kill_process() called os.kill(os.getpid(), SIGTERM) which
        terminated the entire host application (e.g. Maya) rather than just
        stopping the monitored operation.
        Fixed: 2026-02-11
        """
        with patch("ctypes.pythonapi") as mock_api:
            mock_api.PyThreadState_SetAsyncExc.return_value = 1  # success

            ExecutionMonitor._force_interrupt_main_thread()

            mock_api.PyThreadState_SetAsyncExc.assert_called_once()
            call_args = mock_api.PyThreadState_SetAsyncExc.call_args
            # Second argument should be SystemExit
            self.assertIs(call_args[0][1].value, SystemExit)

    @patch("os.kill")
    @patch("os._exit")
    def test_force_stop_never_calls_os_kill(self, mock_exit, mock_kill):
        """Verify that Force Stop never invokes os.kill or os._exit.

        Bug: Force Quit killed the parent application via os.kill(os.getpid()).
        Fixed: 2026-02-11
        """
        with patch("ctypes.pythonapi") as mock_api:
            mock_api.PyThreadState_SetAsyncExc.return_value = 1

            ExecutionMonitor._force_interrupt_main_thread()

            mock_kill.assert_not_called()
            mock_exit.assert_not_called()

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.show_long_execution_dialog"
    )
    def test_execution_monitor_dialog_rearms_per_invocation(self, mock_dialog):
        """The long-execution dialog must show again on a later invocation.

        Bug: ``_dialog_shown`` lived in the decorator-factory scope, so once
        the dialog appeared for one call, every subsequent call of the same
        decorated function ran dialog-less for the life of the session.
        """
        mock_dialog.return_value = True

        @ExecutionMonitor.execution_monitor(threshold=0.1, message="Testing")
        def monitored_func():
            time.sleep(0.3)
            return "done"

        self.assertEqual(monitored_func(), "done")
        first_count = mock_dialog.call_count
        self.assertGreaterEqual(first_count, 1)

        self.assertEqual(monitored_func(), "done")
        self.assertGreater(
            mock_dialog.call_count,
            first_count,
            "Dialog did not re-arm for the second invocation",
        )

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.is_foreground_process",
        return_value=True,
    )
    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.is_escape_pressed"
    )
    def test_escape_cancel_stays_active_after_callback(self, mock_is_escape, _mock_fg):
        """Esc must keep working after the one-shot callback has fired.

        Bug: with ``interval=None`` the monitor thread exited right after the
        first callback, taking the Esc poll with it — the documented
        "press-and-hold Esc at any time" contract silently ended at the
        threshold.
        """
        callback_fired = threading.Event()
        scope = CancelScope("test")

        def cb():
            callback_fired.set()
            return True  # keep waiting, no repeat interval

        # Esc is only "pressed" after the callback has already fired.
        mock_is_escape.side_effect = lambda: callback_fired.is_set()

        @ExecutionMonitor.on_long_execution(
            threshold=0.1,
            callback=cb,
            interval=None,
            allow_escape_cancel=True,
            escape_hold_seconds=0,
            cancel_scope=scope,
        )
        def escape_after_callback_func():
            for _ in range(50):
                if not scope.tick():
                    return "cancelled"
                time.sleep(0.1)
            return "finished"

        self.assertEqual(escape_after_callback_func(), "cancelled")
        self.assertTrue(callback_fired.is_set())

    def test_start_heartbeat_writer_stop_joins_writer_thread(self):
        """stop() must join the writer thread so it cannot recreate the
        heartbeat file after cleanup deletes it."""
        with tempfile.TemporaryDirectory() as td:
            hb = os.path.join(td, "hb.txt")
            stop = ExecutionMonitor._start_heartbeat_writer(hb, interval=0.05)
            time.sleep(0.15)
            stop()

            alive = [
                t
                for t in threading.enumerate()
                if t.name == "ExecutionMonitorHeartbeat" and t.is_alive()
            ]
            self.assertEqual(alive, [], "Heartbeat writer thread still alive")
            self.assertFalse(os.path.exists(hb))

    def test_start_indicator_process_gif_indicator(self):
        """indicator=<gif path> launches the sidecar indicator with that gif."""
        gif = os.path.join(
            os.path.dirname(sys.modules[self._EM_MOD].__file__),
            "task_indicator.gif",
        )
        fake_proc = MagicMock()
        with patch(f"{self._EM_MOD}.subprocess.Popen", return_value=fake_proc) as popen:
            proc = ExecutionMonitor._start_indicator_process(indicator=gif)
            self.assertIs(proc, fake_proc)
            argv = popen.call_args[0][0]
            self.assertTrue(argv[1].endswith("_sidecar.py"), argv)
            self.assertEqual(argv[2], "indicator")
            self.assertIn(f"--gif={gif}", argv)
            self.assertIn(f"--parent-pid={os.getpid()}", argv)

    def test_start_indicator_process_bad_gif_falls_back_to_spinner(self):
        """A nonexistent gif path falls back to the canvas spinner."""
        fake_proc = MagicMock()
        with patch(f"{self._EM_MOD}.subprocess.Popen", return_value=fake_proc) as popen:
            ExecutionMonitor._start_indicator_process(indicator="no_such_file.gif")
            argv = popen.call_args[0][0]
            self.assertTrue(argv[1].endswith("_sidecar.py"), argv)
            self.assertFalse(any(a.startswith("--gif=") for a in argv), argv)

    def test_start_indicator_process_skipped_without_interpreter(self):
        """No resolvable python: no subprocess at all (never a host binary)."""
        with patch(
            f"{self._EM_MOD}.ExecutionMonitor._get_python_executable",
            return_value=None,
        ):
            with patch(f"{self._EM_MOD}.subprocess.Popen") as popen:
                self.assertIsNone(ExecutionMonitor._start_indicator_process(True))
                popen.assert_not_called()

    def test_force_stop_retries_and_falls_back(self):
        """If PyThreadState_SetAsyncExc fails, fall back to _thread.interrupt_main."""
        with patch("ctypes.pythonapi") as mock_api:
            # Return 0 (thread not found) for all 3 attempts
            mock_api.PyThreadState_SetAsyncExc.return_value = 0

            with patch("_thread.interrupt_main") as mock_interrupt:
                ExecutionMonitor._force_interrupt_main_thread()
                # Should have retried 3 times then fallen back
                self.assertEqual(mock_api.PyThreadState_SetAsyncExc.call_count, 3)
                mock_interrupt.assert_called_once()

    def test_callback_result_ignored_once_function_finished(self):
        """A callback answer that arrives AFTER the operation ended is moot.

        The dialog blocks the monitor thread while it waits on the user. When
        the operation finishes in the meantime, whatever the user then clicks
        must not be acted on: on the legacy path a late ``False`` used to call
        ``interrupt_main`` after the function had returned, so the
        KeyboardInterrupt landed in whatever unrelated code ran next.
        """
        func_done = threading.Event()

        def late_cancel(finished=None):
            func_done.wait(5.0)  # the "user" only answers after the op ends
            return False

        @ExecutionMonitor.on_long_execution(threshold=0.1, callback=late_cancel)
        def quick_after_threshold():
            time.sleep(0.3)
            return "done"

        self.assertEqual(quick_after_threshold(), "done")
        func_done.set()
        try:
            time.sleep(0.6)  # a pending interrupt would land here
        except KeyboardInterrupt:
            self.fail("late callback result interrupted the main thread")

    def test_on_long_execution_passes_finished_event_to_callback(self):
        """A callback that declares ``finished`` receives the completion event,
        so a blocking prompt can dismiss itself when the operation ends."""
        seen = {}

        def cb(finished):
            seen["finished"] = finished
            return True

        @ExecutionMonitor.on_long_execution(threshold=0.1, callback=cb)
        def f():
            time.sleep(0.3)

        f()
        self.assertIsInstance(seen.get("finished"), threading.Event)
        self.assertTrue(seen["finished"].is_set())

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.show_long_execution_dialog"
    )
    def test_execution_monitor_hands_dialog_the_finished_event(self, mock_dialog):
        """The long-execution dialog is told when the operation finishes."""
        mock_dialog.return_value = True

        @ExecutionMonitor.execution_monitor(threshold=0.1, message="Testing")
        def f():
            time.sleep(0.3)

        f()
        self.assertIsInstance(
            mock_dialog.call_args.kwargs.get("finished"), threading.Event
        )

    def test_show_long_execution_dialog_dismisses_when_finished(self):
        """The sidecar dialog is terminated once the operation finishes and
        the answer reports STOP_MONITORING rather than a fabricated choice."""
        finished = threading.Event()
        sp = MagicMock()
        proc = sp.Popen.return_value
        proc.poll.return_value = None  # dialog still open
        # The operation ends while the dialog is up (an event set BEFORE the
        # call is the separate short-circuit: no dialog is launched at all).
        threading.Timer(0.3, finished.set).start()
        with patch(f"{self._EM_MOD}.subprocess", sp):
            result = ExecutionMonitor.show_long_execution_dialog(
                "Title", "Msg", finished=finished
            )
        self.assertEqual(result, "STOP_MONITORING")
        proc.terminate.assert_called()

    def test_show_long_execution_dialog_skips_when_already_finished(self):
        """An operation that ended before the prompt gets no prompt at all."""
        finished = threading.Event()
        finished.set()
        with patch(f"{self._EM_MOD}.subprocess") as sp:
            result = ExecutionMonitor.show_long_execution_dialog(
                "Title", "Msg", finished=finished
            )
        self.assertEqual(result, "STOP_MONITORING")
        sp.Popen.assert_not_called()

    def test_show_long_execution_dialog_sidecar_first_on_every_platform(self):
        """The Tk sidecar is the primary dialog on Linux too (the zenity /
        kdialog paths are fallbacks): a clean exit code is honoured before any
        native tool is looked up."""
        with patch("sys.platform", "linux"):
            with patch("shutil.which", return_value=None) as which:
                with patch(
                    f"{self._EM_MOD}.subprocess", self._dialog_subprocess_mock(10)
                ):
                    self.assertFalse(
                        ExecutionMonitor.show_long_execution_dialog("T", "M")
                    )
                    which.assert_not_called()

    def test_spawn_watchdog_uses_resolved_interpreter(self):
        """The watchdog must run under the resolved python, not
        ``sys.executable``: inside Maya that is ``maya.exe``, and the safety
        valve for a hung Maya would have launched a second Maya."""
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        with tempfile.TemporaryDirectory() as td:
            hb = os.path.join(td, "hb.txt")
            with patch.object(
                ExecutionMonitor, "_interpreter_override", "X:/mayapy.exe"
            ):
                with patch("subprocess.Popen", return_value=fake_proc) as popen:
                    ExecutionMonitor._spawn_watchdog_subprocess(
                        pid=1234,
                        heartbeat_path=hb,
                        timeout=5.0,
                        check_interval=0.5,
                        kill_tree=True,
                    )
                    self.assertEqual(popen.call_args[0][0][0], "X:/mayapy.exe")

    def test_spawn_watchdog_skipped_without_interpreter(self):
        """No resolvable python: no watchdog, and the caller is told."""
        logger = MagicMock()
        with patch.object(
            ExecutionMonitor, "_get_python_executable", return_value=None
        ):
            with patch("subprocess.Popen") as popen:
                proc, stop = ExecutionMonitor._spawn_watchdog_subprocess(
                    pid=1,
                    heartbeat_path="hb",
                    timeout=1.0,
                    check_interval=0.5,
                    kill_tree=False,
                    logger=logger,
                )
        self.assertIsNone(proc)
        self.assertIsNone(stop)
        popen.assert_not_called()
        logger.warning.assert_called()

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.is_foreground_process",
        return_value=True,
    )
    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.is_escape_pressed"
    )
    def test_escape_with_scope_keeps_threshold_callback(self, mock_is_escape, _fg):
        """With a scope, an Esc request must not end monitoring.

        The flag is only a request; an operation with no checkpoint ignores it
        and keeps running, and the threshold dialog is then the one place the
        user is told why nothing happened (and offered Force Stop). The old
        thread exited on the first Esc, so that dialog never came.
        """
        mock_is_escape.return_value = True
        scope = CancelScope("test")
        callback = MagicMock(return_value=True)

        @ExecutionMonitor.on_long_execution(
            threshold=0.3,
            callback=callback,
            allow_escape_cancel=True,
            escape_hold_seconds=0,
            cancel_scope=scope,
        )
        def monolith():
            time.sleep(0.8)  # never reaches a checkpoint
            return "finished"

        self.assertEqual(monolith(), "finished")
        self.assertTrue(scope.cancelled)
        self.assertEqual(scope.reason, "escape")
        callback.assert_called()

    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.is_foreground_process",
        return_value=True,
    )
    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.is_escape_pressed"
    )
    def test_callback_exception_keeps_escape_watch_alive(self, mock_is_escape, _fg):
        """A failing callback must not take the monitor thread (and the Esc
        watch) down with it."""
        fired = threading.Event()
        scope = CancelScope("test")

        def broken_callback():
            fired.set()
            raise RuntimeError("logger sink exploded")

        mock_is_escape.side_effect = lambda: fired.is_set()

        @ExecutionMonitor.on_long_execution(
            threshold=0.1,
            callback=broken_callback,
            allow_escape_cancel=True,
            escape_hold_seconds=0,
            cancel_scope=scope,
        )
        def func():
            for _ in range(50):
                if not scope.tick():
                    return "cancelled"
                time.sleep(0.1)
            return "finished"

        self.assertEqual(func(), "cancelled")

    def test_is_escape_pressed_returns_bool(self):
        with patch("sys.platform", "win32"):
            with patch("ctypes.windll.user32.GetAsyncKeyState", return_value=0x8000):
                self.assertIs(ExecutionMonitor.is_escape_pressed(), True)


class TestSidecar(unittest.TestCase):
    """The out-of-process Tk half (``_sidecar.py``): indicator, dialog, watchdog."""

    @staticmethod
    def _dead_pid():
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait(timeout=10)
        return proc.pid

    def test_sidecar_is_self_contained(self):
        """Runs under any host python: no pythontk import, Tk imported lazily
        (the watchdog subcommand must work where tkinter is absent)."""
        import ast

        with open(_SIDECAR, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        top_level = set()
        for node in tree.body:  # module-level statements only
            if isinstance(node, ast.Import):
                top_level.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_level.add((node.module or "").split(".")[0])
        self.assertNotIn("tkinter", top_level, "tkinter must be imported lazily")
        self.assertNotIn("pythontk", top_level)
        everywhere = {
            (n.module or "").split(".")[0]
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
        } | {
            alias.name.split(".")[0]
            for n in ast.walk(tree)
            if isinstance(n, ast.Import)
            for alias in n.names
        }
        self.assertNotIn("pythontk", everywhere)

    def test_watchdog_kills_stalled_process(self):
        """End-to-end: a heartbeat that stops updating gets its owner killed."""
        with tempfile.TemporaryDirectory() as td:
            hb = os.path.join(td, "hb.txt")
            with open(hb, "w", encoding="utf-8") as f:
                f.write("0")
            victim = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"]
            )
            try:
                wd = subprocess.Popen(
                    [
                        sys.executable,
                        _SIDECAR,
                        "watchdog",
                        str(victim.pid),
                        hb,
                        "--timeout=0.5",
                        "--check-interval=0.1",
                        f"--stop-file={hb}.stop",
                    ]
                )
                try:
                    victim.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.fail("watchdog did not kill the stalled process")
                self.assertEqual(wd.wait(timeout=5), 0)
            finally:
                if victim.poll() is None:
                    victim.kill()

    def test_watchdog_exits_on_stop_file(self):
        with tempfile.TemporaryDirectory() as td:
            hb = os.path.join(td, "hb.txt")
            stop = hb + ".stop"
            with open(hb, "w", encoding="utf-8") as f:
                f.write("0")
            with open(stop, "w", encoding="utf-8") as f:
                f.write("stop")
            wd = subprocess.Popen(
                [
                    sys.executable,
                    _SIDECAR,
                    "watchdog",
                    str(os.getpid()),
                    hb,
                    "--timeout=60",
                    "--check-interval=0.1",
                    f"--stop-file={stop}",
                ]
            )
            self.assertEqual(wd.wait(timeout=10), 0)

    @unittest.skipUnless(_HAS_DISPLAY, "Requires a display (headless CI has no GUI)")
    def test_indicator_exits_when_parent_dies(self):
        """A borderless topmost overlay has no close button; an orphan left by
        a host crash could only be removed from the task manager. It watches
        its parent and closes itself."""
        proc = subprocess.Popen(
            [sys.executable, _SIDECAR, "indicator", f"--parent-pid={self._dead_pid()}"],
            stderr=subprocess.PIPE,
        )
        try:
            _, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            self.fail("indicator outlived its (dead) parent")
        self.assertEqual(proc.returncode, 0, err.decode(errors="replace"))

    @unittest.skipUnless(_HAS_DISPLAY, "Requires a display (headless CI has no GUI)")
    def test_dialog_exits_when_parent_dies(self):
        proc = subprocess.Popen(
            [
                sys.executable,
                _SIDECAR,
                "dialog",
                "T",
                "M",
                "--force-label=Force Stop",
                f"--parent-pid={self._dead_pid()}",
            ],
            stderr=subprocess.PIPE,
        )
        try:
            _, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            self.fail("dialog outlived its (dead) parent")
        # Closed by the parent watch == "window closed" (3), i.e. keep waiting.
        self.assertEqual(proc.returncode, 3, err.decode(errors="replace"))

    @unittest.skipUnless(_HAS_DISPLAY, "Requires a display (headless CI has no GUI)")
    def test_dialog_window_fits_its_content(self):
        """The dialog sizes to its message. It used to be a fixed 450x180, and
        the fullest message (Esc hint + no-checkpoint note + a force button)
        needs ~244px: Tk's packer then dropped the button row entirely, so
        the user got a warning with nothing to click."""
        import importlib.util
        import tkinter as tk

        spec = importlib.util.spec_from_file_location("_sidecar", _SIDECAR)
        sidecar = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sidecar)

        message = (
            "Slot 'tb016' on 'widget' (taking longer than 60.0s...)\n\n"
            "The operation is not responding.\n"
            "You can keep waiting or cancel the operation."
            "\n\nPress and hold Esc at any time to cancel the operation."
            "\n\nNote: this operation has not reported any cancellable points "
            "yet, so Cancel will only take effect if and when it reaches one."
        )
        root = tk.Tk()
        root.withdraw()
        try:
            sidecar.build_dialog(root, "Title", message, force_label="Force Stop")
            sidecar.fit_and_center(root)
            root.update_idletasks()
            size = root.geometry().split("+")[0]
            width, height = (int(v) for v in size.split("x"))
            self.assertGreaterEqual(height, root.winfo_reqheight())
            self.assertGreaterEqual(width, root.winfo_reqwidth())
        finally:
            root.destroy()


class TestExecutionMonitorPythonExecutable(unittest.TestCase):
    """`_get_python_executable` DCC-host resolution (maya/max/nuke/generic)."""

    def _test_resolution(self, current_exe, file_system, expected_result):
        """Helper to test resolution logic.

        Parameters:
            current_exe: value of sys.executable.
            file_system: set/list of paths that "exist" for this case.
            expected_result: path that should be returned.
        """
        with patch("sys.executable", current_exe):
            with patch("os.path.exists") as mock_exists:

                def side_effect(path):
                    path = path.lower().replace("\\", "/")
                    return any(
                        f.lower().replace("\\", "/") == path for f in file_system
                    )

                mock_exists.side_effect = side_effect

                result = ExecutionMonitor._get_python_executable()
                if expected_result is None:
                    self.assertIsNone(result)
                    return
                self.assertEqual(
                    result.lower().replace("\\", "/"),
                    expected_result.lower().replace("\\", "/"),
                )

    def test_standard_python(self):
        """Standard python should return itself."""
        exe = r"C:\Python39\python.exe"
        self._test_resolution(exe, {exe}, exe)

    def test_maya(self):
        """maya.exe should find mayapy.exe."""
        exe = r"C:\Program Files\Autodesk\Maya2025\bin\maya.exe"
        mayapy = r"C:\Program Files\Autodesk\Maya2025\bin\mayapy.exe"
        self._test_resolution(exe, {exe, mayapy}, mayapy)

    def test_mayabatch(self):
        """mayabatch.exe should find mayapy.exe."""
        exe = r"C:\Program Files\Autodesk\Maya2025\bin\mayabatch.exe"
        mayapy = r"C:\Program Files\Autodesk\Maya2025\bin\mayapy.exe"
        self._test_resolution(exe, {exe, mayapy}, mayapy)

    def test_3dsmax(self):
        """3dsmax.exe should find 3dsmaxpy.exe."""
        exe = r"C:\Max\3dsmax.exe"
        maxpy = r"C:\Max\3dsmaxpy.exe"
        self._test_resolution(exe, {exe, maxpy}, maxpy)

    def test_generic_app_bundled_python(self):
        """SomeApp.exe with a sibling python.exe."""
        exe = r"C:\App\SomeApp.exe"
        python = r"C:\App\python.exe"
        self._test_resolution(exe, {exe, python}, python)

    def test_unknown_app_no_python(self):
        """UnknownApp.exe with no python anywhere resolves to None.

        Returning the host binary itself was never usable: the callers hand the
        result a *python script* to run, so inside a host with no discoverable
        interpreter (Blender 4.x before ``set_interpreter``) the "spinner" was a
        second copy of the host application.
        """
        exe = r"C:\App\UnknownApp.exe"
        with patch("sys.prefix", r"C:\App"), patch("sys.base_prefix", r"C:\App"):
            self._test_resolution(exe, {exe}, None)

    def test_embedded_python_under_sys_prefix(self):
        """A host binary whose bundled python lives under ``sys.prefix``
        (Blender: ``<ver>/python/bin/python.exe``) resolves to that python."""
        exe = r"C:\Blender\blender.exe"
        python = r"C:\Blender\4.2\python\bin\python.exe"
        with patch("sys.prefix", r"C:\Blender\4.2\python"):
            self._test_resolution(exe, {exe, python}, python)

    def test_nuke(self):
        """Nuke13.0.exe with a sibling python.exe."""
        exe = r"C:\Nuke\Nuke13.0.exe"
        python = r"C:\Nuke\python.exe"
        self._test_resolution(exe, {exe, python}, python)


class TestExecutionMonitorSpinner(unittest.TestCase):
    """Spinner subprocess lifecycle and the public re-export surface."""

    def test_public_import(self):
        """ExecutionMonitor is importable from the top-level pythontk package."""
        self.assertTrue(hasattr(PublicExecutionMonitor, "on_long_execution"))

    @unittest.skipUnless(
        sys.platform == "win32" or os.environ.get("DISPLAY"),
        "Requires a display (headless CI has no GUI)",
    )
    def test_spinner_process_start_stop(self):
        """Start and stop the indicator process directly."""
        process = ExecutionMonitor._start_indicator_process()
        self.assertIsNotNone(process)

        time.sleep(1)
        self.assertIsNone(process.poll())  # still running

        ExecutionMonitor._stop_indicator_process(process)
        self.assertIsNotNone(process.poll())  # stopped

    @unittest.skipUnless(
        sys.platform == "win32" or os.environ.get("DISPLAY"),
        "Requires a display (headless CI has no GUI)",
    )
    def test_spinner_subprocess_stays_alive(self):
        """The spinner subprocess runs without crashing.

        Launches the sidecar indicator directly and confirms it stays alive
        for at least 2 seconds (proves the tkinter mainloop is running).
        """
        script_path = _SIDECAR

        if not os.path.exists(script_path):
            self.skipTest("Sidecar script not found")

        executable = ExecutionMonitor._get_python_executable()
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        proc = subprocess.Popen(
            [executable, script_path, "indicator"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
        )

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                _, err = proc.communicate(timeout=2)
                self.fail(
                    f"Spinner exited immediately (code={proc.returncode}). "
                    f"stderr: {err.decode(errors='replace')}"
                )
            time.sleep(0.2)

        proc.terminate()
        proc.wait(timeout=3)

    @unittest.skipUnless(
        sys.platform == "win32" or os.environ.get("DISPLAY"),
        "Requires a display (headless CI has no GUI)",
    )
    def test_spinner_accepts_negative_position(self):
        """--pos must accept negative coordinates (cursor on a monitor
        left/above the primary). A space-separated "--pos -122,-341" is
        parsed by argparse as an option flag and exits with code 2 —
        the launcher must use the '=' form.
        """
        script_path = _SIDECAR
        if not os.path.exists(script_path):
            self.skipTest("Sidecar script not found")

        executable = ExecutionMonitor._get_python_executable()
        proc = subprocess.Popen(
            [executable, script_path, "indicator", "--pos=-100,-200"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            time.sleep(1)
            if proc.poll() is not None:
                _, err = proc.communicate(timeout=2)
                self.fail(
                    f"Spinner rejected negative --pos (code={proc.returncode}). "
                    f"stderr: {err.decode(errors='replace')}"
                )
        finally:
            proc.terminate()
            proc.wait(timeout=3)

    def test_decorator_with_indicator(self):
        """The decorator runs the tkinter spinner indicator without error."""
        callback_called = [False]

        def my_callback():
            callback_called[0] = True
            return True  # Continue

        @ExecutionMonitor.on_long_execution(
            threshold=0.5, callback=my_callback, indicator=True
        )
        def long_task():
            time.sleep(1.5)
            return "Done"

        result = long_task()

        self.assertEqual(result, "Done")
        self.assertTrue(callback_called[0], "Callback should have been triggered")


if __name__ == "__main__":
    unittest.main()
