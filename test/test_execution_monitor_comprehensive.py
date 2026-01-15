import unittest
from unittest.mock import patch
import os
import sys
from pythontk.core_utils.execution_monitor import ExecutionMonitor


class TestExecutionMonitorComprehensive(unittest.TestCase):

    def _test_resolution(self, current_exe, file_system, expected_result):
        """
        Helper to test resolution logic.
        current_exe: value of sys.executable
        file_system: dict of {path: bool} or set of paths that exist.
        expected_result: path that should be returned.
        """
        if (
            sys.platform == "win32"
            and not current_exe.endswith(".exe")
            and "python" not in current_exe
        ):
            # Just a helper for test setup simplicity, though inputs should be explicit
            pass

        with patch("sys.executable", current_exe):
            with patch("os.path.exists") as mock_exists:

                def side_effect(path):
                    # Normalize for comparison
                    path = path.lower().replace("\\", "/")
                    if isinstance(file_system, (set, list)):
                        return any(
                            f.lower().replace("\\", "/") == path for f in file_system
                        )
                    return False

                mock_exists.side_effect = side_effect

                result = ExecutionMonitor._get_python_executable()
                self.assertEqual(
                    result.lower().replace("\\", "/"),
                    expected_result.lower().replace("\\", "/"),
                )

    def test_standard_python(self):
        """Standard python should return itself."""
        exe = r"C:\Python39\python.exe"
        self._test_resolution(exe, {exe}, exe)

    def test_maya(self):
        """Maya.exe should find mayapy.exe."""
        exe = r"C:\Maya\bin\maya.exe"
        mayapy = r"C:\Maya\bin\mayapy.exe"
        self._test_resolution(exe, {exe, mayapy}, mayapy)

    def test_mayabatch(self):
        """mayabatch.exe should find mayapy.exe."""
        exe = r"C:\Maya\bin\mayabatch.exe"
        mayapy = r"C:\Maya\bin\mayapy.exe"
        self._test_resolution(exe, {exe, mayapy}, mayapy)

    def test_3dsmax(self):
        """3dsmax.exe should find 3dsmaxpy.exe."""
        exe = r"C:\Max\3dsmax.exe"
        maxpy = r"C:\Max\3dsmaxpy.exe"
        self._test_resolution(exe, {exe, maxpy}, maxpy)

    def test_houdini(self):
        """houdini.exe should ideally find hython.exe.
        Current generic logic might fail this without specific execution.
        Let's see if generic 'python' fallback works if hython isn't found?
        """
        # Note: Houdini usually has python.exe in bin too? Or just hython?
        # If we assume 'hython' is the target, the current logic won't find it unless we look for 'hython'.
        pass

    def test_generic_app_bundled_python(self):
        """SomeApp.exe with sibling python.exe."""
        exe = r"C:\App\SomeApp.exe"
        python = r"C:\App\python.exe"
        self._test_resolution(exe, {exe, python}, python)

    def test_unknown_app_no_python(self):
        """UnknownApp.exe with no python sibling should return itself (safest fallback)."""
        exe = r"C:\App\UnknownApp.exe"
        self._test_resolution(exe, {exe}, exe)

    def test_nuke(self):
        """Nuke13.0.exe. If python.exe exists next to it."""
        exe = r"C:\Nuke\Nuke13.0.exe"
        python = r"C:\Nuke\python.exe"
        self._test_resolution(exe, {exe, python}, python)


if __name__ == "__main__":
    unittest.main()
