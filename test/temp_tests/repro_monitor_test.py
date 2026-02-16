import unittest
from unittest.mock import patch, MagicMock
import time
import logging
from pythontk.core_utils.execution_monitor._execution_monitor import ExecutionMonitor


class TestExecutionMonitorRepro(unittest.TestCase):
    @patch(
        "pythontk.core_utils.execution_monitor._execution_monitor.ExecutionMonitor.show_long_execution_dialog"
    )
    def test_execution_monitor_decorator(self, mock_dialog):
        """Test execution_monitor decorator logic."""
        mock_dialog.return_value = True  # Continue
        logger = MagicMock()

        print("Defining monitored_func...")

        @ExecutionMonitor.execution_monitor(
            threshold=0.1, message="Testing", logger=logger
        )
        def monitored_func():
            print("Inside monitored_func, sleeping...")
            time.sleep(1.0)
            print("Inside monitored_func, done sleeping.")
            return "done"

        print("Calling monitored_func...")
        result = monitored_func()
        print(f"Result: {result}")

        if mock_dialog.called:
            print("mock_dialog was called.")
        else:
            print("mock_dialog was NOT called.")

        self.assertEqual(result, "done")
        mock_dialog.assert_called()


if __name__ == "__main__":
    unittest.main()
