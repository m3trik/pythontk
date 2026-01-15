import unittest
import time
import os
import sys
from pythontk import ExecutionMonitor


class TestExecutionMonitorRefactor(unittest.TestCase):
    def test_import(self):
        """Test that ExecutionMonitor can be imported from core_utils."""
        self.assertTrue(hasattr(ExecutionMonitor, "on_long_execution"))

    def test_gif_process_start_stop(self):
        """Test starting and stopping the GIF process directly."""
        # Get the path to the bundled GIF
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gif_path = os.path.join(
            current_dir,
            "pythontk",
            "core_utils",
            "execution_monitor",
            "task_indicator.gif",
        )

        if not os.path.exists(gif_path):
            print(f"Warning: GIF not found at {gif_path}, skipping process test.")
            return

        process = ExecutionMonitor._start_gif_process(gif_path)
        self.assertIsNotNone(process)

        # Let it run for a bit
        time.sleep(1)

        # Check if it is running (poll returns None if running)
        self.assertIsNone(process.poll())

        ExecutionMonitor._stop_gif_process(process)

        # Check if it stopped
        self.assertIsNotNone(process.poll())

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
