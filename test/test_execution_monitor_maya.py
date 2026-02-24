import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from pythontk.core_utils.execution_monitor._execution_monitor import ExecutionMonitor


class TestExecutionMonitorMaya(unittest.TestCase):
    @patch("sys.platform", "win32")
    def test_get_python_executable_maya(self):
        """Test detection of mayapy.exe when running in maya.exe."""

        # Mock sys.executable to look like maya.exe
        fake_maya_path = r"C:\Program Files\Autodesk\Maya2025\bin\maya.exe"
        expected_mayapy = r"C:\Program Files\Autodesk\Maya2025\bin\mayapy.exe"

        with patch("sys.executable", fake_maya_path):
            with patch("os.path.exists") as mock_exists:
                # Setup mock to return True for mayapy path
                def side_effect(path):
                    if path.lower() == expected_mayapy.lower():
                        return True
                    return False

                mock_exists.side_effect = side_effect

                result = ExecutionMonitor._get_python_executable()

                self.assertEqual(result.lower(), expected_mayapy.lower())

    @patch("sys.platform", "win32")
    def test_get_python_executable_mayabatch(self):
        """Test detection of mayapy.exe when running in mayabatch.exe."""

        fake_maya_path = r"C:\Program Files\Autodesk\Maya2025\bin\mayabatch.exe"
        expected_mayapy = r"C:\Program Files\Autodesk\Maya2025\bin\mayapy.exe"

        with patch("sys.executable", fake_maya_path):
            with patch("os.path.exists") as mock_exists:

                def side_effect(path):
                    if path.lower() == expected_mayapy.lower():
                        return True
                    return False

                mock_exists.side_effect = side_effect

                result = ExecutionMonitor._get_python_executable()

                self.assertEqual(result.lower(), expected_mayapy.lower())

    def test_get_python_executable_normal(self):
        """Test standard python executable is returned unchanged."""

        fake_python = r"C:\Python39\python.exe"

        with patch("sys.executable", fake_python):
            result = ExecutionMonitor._get_python_executable()
            self.assertEqual(result, fake_python)


if __name__ == "__main__":
    unittest.main()
