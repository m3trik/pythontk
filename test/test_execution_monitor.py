import unittest
import time
import threading
import sys
from unittest.mock import MagicMock, patch
from pythontk.core_utils.execution_monitor import ExecutionMonitor


from conftest import BaseTestCase


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
                time.sleep(1.0)
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

    @patch("pythontk.core_utils.execution_monitor.ExecutionMonitor.is_escape_pressed")
    def test_on_long_execution_escape_cancel(self, mock_is_escape):
        """Test that holding Escape interrupts execution if allowed."""
        # Mock escape being pressed after a short delay
        # We need is_escape_pressed to return False initially, then True
        mock_is_escape.side_effect = [False, False, True, True, True]

        callback = MagicMock()

        @ExecutionMonitor.on_long_execution(
            threshold=1.0, callback=callback, allow_escape_cancel=True
        )
        def escape_func():
            try:
                # Sleep in small chunks to allow interrupt to happen
                for _ in range(10):
                    time.sleep(0.1)
            except KeyboardInterrupt:
                return "interrupted"
            return "finished"

        result = escape_func()
        self.assertEqual(result, "interrupted")
        # Callback shouldn't be called because we interrupted before threshold
        callback.assert_not_called()

    @patch(
        "pythontk.core_utils.execution_monitor.ExecutionMonitor.show_long_execution_dialog"
    )
    def test_execution_monitor_decorator(self, mock_dialog):
        """Test the high-level execution_monitor decorator."""
        mock_dialog.return_value = True  # Continue waiting
        logger = MagicMock()

        @ExecutionMonitor.execution_monitor(
            threshold=0.1, message="Testing", logger=logger
        )
        def monitored_func():
            time.sleep(0.2)
            return "done"

        result = monitored_func()
        self.assertEqual(result, "done")
        mock_dialog.assert_called()
        logger.warning.assert_called()
        logger.info.assert_called_with("Continuing execution (monitoring active).")

    @patch(
        "pythontk.core_utils.execution_monitor.ExecutionMonitor.show_long_execution_dialog"
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
                time.sleep(0.5)
            except KeyboardInterrupt:
                return "interrupted"
            return "finished"

        result = monitored_func()
        self.assertEqual(result, "interrupted")
        logger.warning.assert_any_call("Aborting execution by user request.")

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

    def test_show_long_execution_dialog_windows(self):
        """Test show_long_execution_dialog on Windows (mocked)."""
        with patch("sys.platform", "win32"):
            with patch("ctypes.windll.user32.MessageBoxW") as mock_msg_box:
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

                # IDCANCEL=2 -> "STOP_MONITORING"
                mock_msg_box.return_value = 2
                self.assertEqual(
                    ExecutionMonitor.show_long_execution_dialog("Title", "Msg"),
                    "STOP_MONITORING",
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
        with patch("sys.platform", "linux"):
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

                    # Zenity returns 0 with "Stop Monitoring" stdout -> "STOP_MONITORING"
                    mock_run.return_value = MagicMock(
                        returncode=0, stdout="Stop Monitoring\n"
                    )
                    self.assertEqual(
                        ExecutionMonitor.show_long_execution_dialog("Title", "Msg"),
                        "STOP_MONITORING",
                    )

    def test_show_long_execution_dialog_linux_kdialog(self):
        """Test show_long_execution_dialog on Linux using KDialog (mocked)."""
        with patch("sys.platform", "linux"):
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

                    # KDialog returns 2 (Cancel) -> "STOP_MONITORING"
                    mock_run.return_value = MagicMock(returncode=2)
                    self.assertEqual(
                        ExecutionMonitor.show_long_execution_dialog("Title", "Msg"),
                        "STOP_MONITORING",
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

    @patch("pythontk.core_utils.execution_monitor.ExecutionMonitor.is_escape_pressed")
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


if __name__ == "__main__":
    unittest.main()
