import unittest
import time
import os
import sys
import subprocess
from pythontk import ExecutionMonitor


class TestExecutionMonitorRefactor(unittest.TestCase):
    def test_import(self):
        """Test that ExecutionMonitor can be imported from core_utils."""
        self.assertTrue(hasattr(ExecutionMonitor, "on_long_execution"))

    @unittest.skipUnless(
        sys.platform == "win32" or os.environ.get("DISPLAY"),
        "Requires a display (headless CI has no GUI)",
    )
    def test_spinner_process_start_stop(self):
        """Test starting and stopping the spinner process directly."""
        process = ExecutionMonitor._start_spinner_process()
        self.assertIsNotNone(process)

        # Let it run for a bit
        time.sleep(1)

        # Check if it is running (poll returns None if running)
        self.assertIsNone(process.poll())

        ExecutionMonitor._stop_spinner_process(process)

        # Check if it stopped
        self.assertIsNotNone(process.poll())

    @unittest.skipUnless(
        sys.platform == "win32" or os.environ.get("DISPLAY"),
        "Requires a display (headless CI has no GUI)",
    )
    def test_spinner_subprocess_stays_alive(self):
        """Verify the spinner subprocess runs without crashing.

        Launches _spinner.py directly and confirms it stays alive for at
        least 2 seconds (proves tkinter mainloop is running).
        """
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(
            current_dir,
            "pythontk",
            "core_utils",
            "execution_monitor",
            "_spinner.py",
        )

        if not os.path.exists(script_path):
            self.skipTest("Spinner script not found")

        executable = ExecutionMonitor._get_python_executable()
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        proc = subprocess.Popen(
            [executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
        )

        # Poll until alive or failed (up to 2s)
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

    def test_decorator_with_indicator(self):
        """Test the decorator with indicator=True."""

        # We need a callback for the threshold
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

        # Run the task
        result = long_task()

        self.assertEqual(result, "Done")
        self.assertTrue(callback_called[0], "Callback should have been triggered")


if __name__ == "__main__":
    unittest.main()
