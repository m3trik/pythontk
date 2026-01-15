# !/usr/bin/python
# coding=utf-8
import sys
import unittest
import os
import shutil
import time

# Ensure source is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import pythontk as ptk
except ImportError:
    # If standard import fails, try direct import from source
    import pythontk.core_utils.app_launcher as app_launcher

    AppLauncher = app_launcher.AppLauncher
else:
    AppLauncher = ptk.AppLauncher


class TestAppLauncher(unittest.TestCase):
    def test_find_python(self):
        """Test finding the python executable relative to PATH."""
        # 'python' should generally be in path
        path = AppLauncher.find_app("python")
        print(f"AppLauncher found python at: {path}")
        self.assertTrue(path, "Could not find python executable via AppLauncher")

    def test_launch_python_version(self):
        """Test launching python --version."""
        print("Launching python --version...")
        process = AppLauncher.launch(
            "python", args=["--version"], detached=False
        )  # not detached so we can wait
        self.assertIsNotNone(process, "Failed to launch python process")
        if process:
            process.wait()
            self.assertEqual(
                process.returncode, 0, "python --version returned non-zero exit code"
            )

    def test_wait_for_ready(self):
        """Test launching an app and waiting for its UI (Windows specific logic)."""
        if sys.platform == "win32":
            print("Launching test python UI app and waiting for readiness...")

            # Use our own fixture app that is guaranteed to be a standard process
            fixture_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "fixtures", "test_ui_app.py")
            )

            # Launch python pointing to the fixture
            # We don't use AppLauncher.launch("python", ...) directly here because we need to ensure
            # we run the SAME python that is running this test to avoid environment mismatches.
            python_exe = sys.executable
            cmd = [python_exe, fixture_path]

            # Use launch via the helper, but pass absolute path to python as identifier
            # We must use detached=False so that the PID we get is definitely the process running the window,
            # though even with detached=True it should work for standard python.exe.
            process = AppLauncher.launch(
                python_exe, args=[fixture_path], detached=False
            )
            self.assertIsNotNone(process)

            # Wait for it to be ready
            is_ready = AppLauncher.wait_for_ready(process, timeout=10)

            # Clean up
            import subprocess

            if process.poll() is None:
                # Use taskkill to force kill tree if needed, or terminate
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(process.pid)])

            if not is_ready:
                print(
                    "WARNING: Test app did not report ready. This might be due to test environment restriction (hidden windows)."
                )
            self.assertTrue(
                is_ready,
                "Test app did not report ready (visible window) within timeout",
            )
        else:
            print("Skipping wait_for_ready test on non-windows platform")

    def test_get_window_titles(self):
        """Test getting window titles for a PID (Windows only)."""
        if sys.platform != "win32":
            print("Skipping get_window_titles test on non-windows platform")
            return

        fixture_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "fixtures", "test_ui_app.py")
        )

        python_exe = sys.executable
        process = AppLauncher.launch(python_exe, args=[fixture_path], detached=False)
        self.assertIsNotNone(process)

        # Wait briefly for the window to appear
        AppLauncher.wait_for_ready(process, timeout=10)
        time.sleep(0.5)

        titles = AppLauncher.get_window_titles(process.pid)
        self.assertIsInstance(titles, list)
        self.assertTrue(
            any("TestAppWindow" in t for t in titles),
            f"Expected window title not found. Titles: {titles}",
        )

        # Cleanup
        import subprocess

        if process.poll() is None:
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(process.pid)])

    def test_find_system_apps(self):
        """Test finding OS specific apps."""
        if sys.platform == "win32":
            # Notepad is usually in path or registry
            path = AppLauncher.find_app("notepad")
            print(f"AppLauncher found notepad at: {path}")
            self.assertTrue(path, "Could not find notepad on Windows")

            # Chrome often is not in path but in registry
            # This is not guaranteed to be installed, so we check if it finds it OR returns None gracefully
            path = AppLauncher.find_app("chrome")
            if path:
                print(f"AppLauncher found chrome at: {path}")
            else:
                print("AppLauncher did not find chrome (might not be installed)")

        elif sys.platform.startswith("linux"):
            path = AppLauncher.find_app("ls")
            print(f"AppLauncher found ls at: {path}")
            self.assertTrue(path, "Could not find ls on Linux")


if __name__ == "__main__":
    unittest.main()
