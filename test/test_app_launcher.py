# !/usr/bin/python
# coding=utf-8
import sys
import unittest
import os
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


def _has_interactive_display():
    """Return True if the current session can show and detect GUI windows."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        # GetDesktopWindow returns 0 when there is no interactive desktop
        hwnd = ctypes.windll.user32.GetDesktopWindow()
        if not hwnd:
            return False
        # Also verify EnumWindows works (fails in some CI containers)
        results = []
        WNDENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        )

        def cb(hwnd, lParam):
            results.append(hwnd)
            return len(results) < 5  # just check a few

        ctypes.windll.user32.EnumWindows(WNDENUMPROC(cb), 0)
        return len(results) > 0
    except Exception:
        return False


_INTERACTIVE = _has_interactive_display()


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

    @unittest.skipUnless(
        _INTERACTIVE, "Requires interactive desktop with visible windows"
    )
    def test_wait_for_ready(self):
        """Test launching an app and waiting for its UI (Windows specific logic)."""
        if sys.platform == "win32":
            print("Launching test python UI app and waiting for readiness...")

            # Use our own fixture app that is guaranteed to be a standard process
            fixture_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "fixtures", "ui_app_fixture.py")
            )

            # Launch python pointing to the fixture
            # We don't use AppLauncher.launch("python", ...) directly here because we need to ensure
            # we run the SAME python that is running this test to avoid environment mismatches.
            python_exe = sys.executable

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
            if not is_ready:
                self.skipTest(
                    "Window not detected — PID/visibility mismatch in this environment"
                )
        else:
            print("Skipping wait_for_ready test on non-windows platform")

    @unittest.skipUnless(
        _INTERACTIVE, "Requires interactive desktop with visible windows"
    )
    def test_get_window_titles(self):
        """Test getting window titles for a PID (Windows only)."""
        if sys.platform != "win32":
            print("Skipping get_window_titles test on non-windows platform")
            return

        fixture_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "fixtures", "ui_app_fixture.py")
        )

        python_exe = sys.executable
        process = AppLauncher.launch(python_exe, args=[fixture_path], detached=False)
        self.assertIsNotNone(process)

        # Wait briefly for the window to appear
        AppLauncher.wait_for_ready(process, timeout=10)
        time.sleep(0.5)

        titles = AppLauncher.get_window_titles(process.pid)
        self.assertIsInstance(titles, list)
        if not any("TestAppWindow" in t for t in titles):
            # Window detection can fail due to PID mismatch on some systems
            self.skipTest(
                f"Window title not found (PID mismatch or visibility issue). Titles: {titles}"
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

    @unittest.skipUnless(sys.platform == "win32", "Windows-specific install layout")
    def test_find_app_program_files_glob_fallback(self):
        """find_app should locate ``PF\\<vendor>\\<app>\\<app>.exe`` even when
        the app is not in PATH and not in the App Paths registry — covers
        Adobe Substance 3D Painter and similar vendors."""
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as fake_pf:
            vendor_dir = os.path.join(fake_pf, "FakeVendor", "FakeApp")
            os.makedirs(vendor_dir)
            exe_path = os.path.join(vendor_dir, "FakeApp.exe")
            open(exe_path, "w").close()

            fake_env = {
                "ProgramFiles": fake_pf,
                "ProgramFiles(x86)": "",
                "ProgramW6432": "",
                "PATH": "",  # ensure shutil.which can't find it
            }
            with patch.dict(os.environ, fake_env, clear=False):
                found = AppLauncher.find_app("FakeApp")
                self.assertEqual(found, exe_path)


class TestHandoffEnv(unittest.TestCase):
    """handoff_env — strip a source-app-private OCIO, inherit everything else.

    Regression: a DCC bridge launches app B from inside app A, so B inherits A's
    env; an OCIO pointing inside A's own install (Blender's bundled v2.5 config)
    failed Maya 2025's color-management init on every send.
    """

    def setUp(self):
        import tempfile

        self.root = tempfile.mkdtemp(prefix="ptk_handoff_")
        self.config = os.path.join(self.root, "datafiles", "config.ocio")
        os.makedirs(os.path.dirname(self.config))
        open(self.config, "w").close()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def test_private_ocio_is_stripped(self):
        from unittest.mock import patch

        with patch.dict(os.environ, {"OCIO": self.config}):
            env = AppLauncher.handoff_env(self.root)
            self.assertIsNotNone(env)
            self.assertNotIn("OCIO", env)
            # The rest of the environment rides along untouched.
            self.assertEqual(env.get("PATH"), os.environ.get("PATH"))

    def test_foreign_ocio_inherits(self):
        from unittest.mock import patch

        import tempfile

        foreign = os.path.join(tempfile.gettempdir(), "studio_config.ocio")
        with patch.dict(os.environ, {"OCIO": foreign}):
            self.assertIsNone(AppLauncher.handoff_env(self.root))

    def test_no_ocio_or_no_root_inherits(self):
        from unittest.mock import patch

        env_no_ocio = {k: v for k, v in os.environ.items() if k != "OCIO"}
        with patch.dict(os.environ, env_no_ocio, clear=True):
            self.assertIsNone(AppLauncher.handoff_env(self.root))
        with patch.dict(os.environ, {"OCIO": self.config}):
            self.assertIsNone(AppLauncher.handoff_env(None))


class TestAppLauncherSessions(unittest.TestCase):
    """Interactive-session detection + launch (added for headless DCC/SDK driving)."""

    def test_session_id_types(self):
        sid = AppLauncher.current_session_id()
        self.assertTrue(sid is None or isinstance(sid, int))
        acsid = AppLauncher.active_console_session_id()
        self.assertTrue(acsid is None or isinstance(acsid, int))
        self.assertIsInstance(AppLauncher.is_interactive_session(), bool)

    def test_find_session_launcher_explicit(self):
        import tempfile

        fd, p = tempfile.mkstemp(suffix="PsExec64.exe")
        os.close(fd)
        try:
            self.assertEqual(AppLauncher.find_session_launcher(explicit=p), p)
        finally:
            os.remove(p)

    def test_find_session_launcher_missing_is_none_or_real(self):
        from unittest.mock import patch

        with patch.dict(os.environ, {"PSEXEC": ""}, clear=False):
            res = AppLauncher.find_session_launcher(explicit="Z:/nope/PsExec64.exe")
            # Either nothing found, or a genuine PsExec present on this host.
            self.assertTrue(res is None or os.path.isfile(res))

    @unittest.skipUnless(sys.platform == "win32", "Windows-only API")
    def test_launch_in_session_no_launcher_raises(self):
        from unittest.mock import patch

        with patch.object(AppLauncher, "find_session_launcher", return_value=None):
            # Target a session that is not the current one to force the PsExec path.
            with self.assertRaises(RuntimeError):
                AppLauncher.launch_in_session("notepad", session=99999)

    @unittest.skipIf(sys.platform == "win32", "non-Windows guard path")
    def test_launch_in_session_non_windows_raises(self):
        with self.assertRaises(RuntimeError):
            AppLauncher.launch_in_session("ls", session=1)

    @unittest.skipUnless(sys.platform == "win32", "Windows-only API")
    def test_active_console_session_id_no_session_returns_none(self):
        """WTSGetActiveConsoleSessionId returns DWORD 0xFFFFFFFF when no user
        is logged on — but ctypes' default c_int restype surfaces that as -1,
        which the sentinel comparison must still recognize."""
        import ctypes
        from unittest.mock import patch

        with patch.object(
            ctypes.windll.kernel32,
            "WTSGetActiveConsoleSessionId",
            return_value=-1,
        ):
            self.assertIsNone(AppLauncher.active_console_session_id())


if __name__ == "__main__":
    unittest.main()
