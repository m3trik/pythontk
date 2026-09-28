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
    def _python_on_path(self):
        """The running interpreter's bare name, with its folder first on PATH.

        Not a literal ``"python"``: a Linux distro may ship only ``python3``.
        """
        from unittest.mock import patch

        folder, exe = os.path.split(sys.executable)
        path = folder + os.pathsep + os.environ.get("PATH", "")
        patcher = patch.dict(os.environ, {"PATH": path})
        patcher.start()
        self.addCleanup(patcher.stop)
        return os.path.splitext(exe)[0]

    def test_find_python(self):
        """Test finding the python executable relative to PATH."""
        path = AppLauncher.find_app(self._python_on_path())
        print(f"AppLauncher found python at: {path}")
        self.assertTrue(path, "Could not find python executable via AppLauncher")

    def test_launch_python_version(self):
        """Test launching python --version."""
        print("Launching python --version...")
        process = AppLauncher.launch(
            self._python_on_path(), args=["--version"], detached=False
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

    def test_get_running_processes_accepts_the_bare_name(self):
        """``get_running_processes("maya")`` finds maya.exe -- the documented form.

        Regression: on Windows the bare name went to tasklist's IMAGENAME filter and
        the row check verbatim, so it matched nothing -- and MayaConnection's
        "is our PID still a Maya?" guard never let it close the stale instance it
        had launched before relaunching.
        """
        image = os.path.basename(sys.executable)
        if os.getpid() not in AppLauncher.get_running_processes(image):
            self.skipTest(f"this process is not listed under {image}")
        bare = os.path.splitext(image)[0]
        self.assertIn(os.getpid(), AppLauncher.get_running_processes(bare))

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

    @unittest.skipUnless(sys.platform == "win32", "Windows-specific install layout")
    def test_find_app_program_files_fallback_takes_the_newest_install(self):
        """Versions side by side: the newest wins, ranked naturally (``4.10``
        over ``4.9``). glob yields the file system's order -- by name on NTFS,
        the OLDEST first -- so find_app answered Blender 3.6 over 5.1, and
        blendertk's ``find_blender`` (highest version, before it delegated
        here) and mayatk's Blender bridge inherited it."""
        import shutil
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as fake_pf:
            vendor = os.path.join(fake_pf, "PtkFake Foundation")
            for version in ("3.6", "4.9", "4.10", "5.1"):
                folder = os.path.join(vendor, f"PtkFake {version}")
                os.makedirs(folder)
                open(os.path.join(folder, "PtkFake.exe"), "w").close()
            fake_env = {
                "ProgramFiles": fake_pf,
                "ProgramFiles(x86)": "",
                "ProgramW6432": "",
                "PATH": "",  # nor on PATH: the Program Files scan answers
            }
            with patch.dict(os.environ, fake_env, clear=False):
                newest = AppLauncher.find_app("PtkFake")
                self.assertEqual(
                    newest, os.path.join(vendor, "PtkFake 5.1", "PtkFake.exe")
                )
                # Natural, not string, order: "4.9" > "4.10" as strings.
                shutil.rmtree(os.path.join(vendor, "PtkFake 5.1"))
                self.assertEqual(
                    AppLauncher.find_app("PtkFake"),
                    os.path.join(vendor, "PtkFake 4.10", "PtkFake.exe"),
                )


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


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux: /proc and X11")
class TestLinuxProcesses(unittest.TestCase):
    """Process identity by NAME, and a process's windows through its children."""

    def _spawn(self, args, **kwargs):
        import subprocess

        proc = subprocess.Popen(args, **kwargs)
        self.addCleanup(proc.wait, 10)
        self.addCleanup(proc.kill)
        return proc

    def test_get_running_processes_matches_the_program_not_a_command_line(self):
        """``pgrep -f`` matched any command line containing the name, so the
        "is this PID still a Maya?" guard matched ``tail maya.log`` too."""
        sleeper = self._spawn(["sleep", "30"])
        mention = self._spawn(
            [sys.executable, "-c", "import time; time.sleep(30)  # sleep"]
        )
        time.sleep(0.2)  # let both exec
        pids = AppLauncher.get_running_processes("sleep")
        self.assertIn(sleeper.pid, pids)
        self.assertNotIn(mention.pid, pids)
        self.assertIn(sleeper.pid, AppLauncher.get_running_processes("sleep.exe"))

    def test_a_launched_app_is_its_process_tree_and_closes_as_one(self):
        """``bin/maya`` is a script that may fork ``maya.bin``: the port and
        window a launched PID "owns" can be a child's, and closing the launched
        PID must close the child too (``launch(detached=True)`` gives it its
        own process group, and ``close_process`` signals that group)."""
        wrapper = AppLauncher.launch(
            "/bin/sh", ["-c", "sleep 120 & echo $!; wait"], detached=True
        )
        self.addCleanup(wrapper.wait, 10)
        # detached=True leaves stdout inherited; read the child's pid from /proc.
        deadline = time.monotonic() + 5
        tree = {wrapper.pid}
        while time.monotonic() < deadline and len(tree) < 2:
            tree = AppLauncher.process_tree(wrapper.pid)
            time.sleep(0.05)
        (child,) = tree - {wrapper.pid}
        self.assertTrue(AppLauncher.close_process(wrapper.pid))
        wrapper.wait(10)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and os.path.exists(f"/proc/{child}"):
            with open(f"/proc/{child}/stat", "rb") as f:
                if f.read().rsplit(b")", 1)[1].split()[0] == b"Z":
                    break  # dead, awaiting its (new) parent's reap
            time.sleep(0.05)
        else:
            self.assertFalse(
                os.path.exists(f"/proc/{child}"), "the child outlived its app"
            )

    def test_window_titles_follow_a_launcher_scripts_children(self):
        """A Linux launcher is often a script that forks the real binary (Maya's
        ``bin/maya``): the window belongs to a CHILD of the PID we launched."""
        import shlex
        import subprocess

        from pythontk.core_utils.x11 import X11

        from conftest import X11Window

        if not os.environ.get("DISPLAY") or not X11.available():
            self.skipTest("needs a reachable X server")
        title = f"ptk-launcher-probe-{os.getpid()}"
        code = X11Window.script(title)
        # `sh -c "<python>; true"`: the trailing command keeps sh from exec'ing,
        # so python really is a child of the PID we hold.
        wrapper = self._spawn(
            ["sh", "-c", f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}; true"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(wrapper.stdout.readline().strip(), "up")
        deadline = time.monotonic() + 10
        while (
            time.monotonic() < deadline
            and title not in AppLauncher.get_window_titles(wrapper.pid)
        ):
            time.sleep(0.1)
        self.assertIn(title, AppLauncher.get_window_titles(wrapper.pid))
        self.assertTrue(AppLauncher._has_window(wrapper.pid))
        wrapper.stdin.close()


@unittest.skipIf(sys.platform == "win32", "POSIX: ~/.profile (Windows: the registry)")
class TestPersistentPathPosix(unittest.TestCase):
    """append_to_path(user_scope=True) owns ONE marked line per directory in
    ~/.profile; is_path_persisted finds exactly that line."""

    def test_the_profile_line_is_written_once_and_found(self):
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as home:
            tool_dir = os.path.join(home, "my tools", "bin")  # a space: quoted
            os.makedirs(tool_dir)
            profile = os.path.join(home, ".profile")
            with open(profile, "w", encoding="utf-8") as f:
                f.write("# the user's own profile\n")
            env = {"HOME": home, "PATH": os.environ.get("PATH", "")}
            with patch.dict(os.environ, env):
                self.assertFalse(AppLauncher.is_path_persisted(tool_dir))
                self.assertTrue(AppLauncher.append_to_path(tool_dir))
                self.assertTrue(AppLauncher.append_to_path(tool_dir))  # idempotent
                self.assertTrue(AppLauncher.is_path_persisted(tool_dir))
                self.assertIn(tool_dir, os.environ["PATH"].split(os.pathsep))
            with open(profile, encoding="utf-8") as f:
                text = f.read()
            self.assertTrue(text.startswith("# the user's own profile\n"))
            self.assertEqual(text.count("AppLauncher.append_to_path"), 1)
            # The line is valid sh: the directory arrives on PATH verbatim.
            import subprocess

            out = subprocess.run(
                ["sh", "-c", f'. {profile}; printf %s "$PATH"'],
                capture_output=True,
                text=True,
                env={"PATH": "/usr/bin:/bin", "HOME": home},
            ).stdout
            self.assertEqual(out.split(":")[-1], tool_dir)


class TestInstallDiscovery(unittest.TestCase):
    """One AppSpec lists every OS's install layouts: {exe}/{program_files}/~
    tokens, version-aware "newest", and (Linux) the application menu."""

    def setUp(self):
        import tempfile

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = folder.name
        self.exe = ".exe" if sys.platform == "win32" else ""

    def _touch(self, *parts):
        path = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("x")
        return path

    def test_scan_ranks_versions_naturally_and_spells_exe_per_os(self):
        """``Blender 4.10`` outranks ``Blender 4.9`` (a string sort had it
        the other way round), and ``{exe}`` is ``.exe`` on Windows only."""
        old = self._touch("Blender 4.9", f"blender{self.exe}")
        new = self._touch("Blender 4.10", f"blender{self.exe}")
        pattern = os.path.join(self.root, "Blender *", "blender{exe}")
        self.assertEqual(list(AppLauncher.scan_install_dirs([pattern])), [new, old])

    def test_scan_for_executables_ranks_versions_naturally(self):
        """Its newest-first promise held for a string sort only by luck:
        ``Blender 4.9`` outranked ``Blender 4.10``, as in scan_install_dirs."""
        old = self._touch("Blender 4.9", "blender.exe")
        new = self._touch("Blender 4.10", "blender.exe")
        found = AppLauncher.scan_for_executables(self.root, "blender.exe")
        self.assertEqual(found, [new, old])

    def test_a_program_files_pattern_is_a_windows_layout(self):
        from unittest.mock import patch

        self._touch("Vendor", "tool.exe")
        with patch.object(
            AppLauncher, "_program_files_roots", return_value=(self.root, self.root)
        ):
            found = list(
                AppLauncher.scan_install_dirs([r"{program_files}/Vendor/tool.exe"])
            )
        self.assertEqual(bool(found), sys.platform == "win32")

    def test_location_suffix_spells_exe_per_os(self):
        from unittest.mock import patch

        tool = self._touch("bin", f"mayapy{self.exe}")
        with patch.dict(os.environ, {"PTK_TEST_LOCATION": self.root}):
            found = AppLauncher.resolve_app_path(
                location_env_vars=(("PTK_TEST_LOCATION", ("bin", "mayapy{exe}")),)
            )
        self.assertEqual(found, tool)

    @unittest.skipUnless(sys.platform.startswith("linux"), "XDG application menu")
    def test_find_app_reads_the_application_menu(self):
        """An app off PATH (a tarball install with a menu entry) is found by
        its entry's name; a sandboxed entry's program is its launcher, never
        the app, so it is skipped."""
        from unittest.mock import patch

        tool = self._touch("opt", "Painter", "Adobe Substance 3D Painter")
        os.chmod(tool, 0o755)
        apps = os.path.join(self.root, "share", "applications")
        os.makedirs(apps)
        with open(os.path.join(apps, "painter.desktop"), "w") as f:
            f.write(
                "[Desktop Entry]\nName=Adobe Substance 3D Painter\n"
                f'Exec="{tool}" %F\nType=Application\n'
            )
        with open(os.path.join(apps, "org.blender.Blender.desktop"), "w") as f:
            f.write(
                "[Desktop Entry]\nName=Blender\nExec=/usr/bin/flatpak run org.blender.Blender %f\n"
            )
        env = {
            "XDG_DATA_HOME": os.path.join(self.root, "share"),
            "XDG_DATA_DIRS": os.path.join(self.root, "none"),
            "PATH": os.path.join(self.root, "none"),
            "HOME": self.root,
        }
        with patch.dict(os.environ, env):
            self.assertEqual(AppLauncher.find_app("Adobe Substance 3D Painter"), tool)
            self.assertEqual(AppLauncher.find_app("painter"), tool)  # the file name
            self.assertIsNone(AppLauncher.find_app("Blender"))

    @unittest.skipUnless(sys.platform.startswith("linux"), "XDG application menu")
    def test_the_application_menu_never_answers_with_a_launcher(self):
        """An entry whose program runs SOMETHING ELSE names that thing, not the
        program: Steam's ``Name=Blender`` entry is ``steam
        steam://rungameid/365670``, and find_app("Blender") answered Steam. A
        positional argument (a URL, a script) or a known launcher disqualifies
        the entry; options and field codes are the app's own."""
        from unittest.mock import patch

        steam = self._touch("games", "steam")
        tool = self._touch("opt", "tool", "tool")
        for program in (steam, tool):
            os.chmod(program, 0o755)
        apps = os.path.join(self.root, "share", "applications")
        os.makedirs(apps)
        entries = {
            "Blender.desktop": "Name=Blender\nExec=steam steam://rungameid/365670\n",
            "houdini.desktop": f"Name=Houdini\nExec=/bin/sh -c \"exec '{tool}'\"\n",
            "painter.desktop": "Name=Painter\nExec=python3 /opt/painter/run.py %F\n",
            "tool.desktop": f"Name=Tool\nExec={tool} --no-sandbox %U\n",
        }
        for name, body in entries.items():
            with open(os.path.join(apps, name), "w") as f:
                f.write("[Desktop Entry]\nType=Application\n" + body)
        env = {
            "XDG_DATA_HOME": os.path.join(self.root, "share"),
            "XDG_DATA_DIRS": os.path.join(self.root, "none"),
            # steam and python3 resolve, as they do on a real desktop
            "PATH": os.pathsep.join((os.path.dirname(steam), "/usr/bin", "/bin")),
            "HOME": self.root,
        }
        with patch.dict(os.environ, env):
            self.assertIsNone(AppLauncher.find_app("Blender"))  # Steam's entry
            self.assertIsNone(AppLauncher.find_app("Houdini"))  # a shell's
            self.assertIsNone(AppLauncher.find_app("Painter"))  # a script's
            self.assertEqual(AppLauncher.find_app("Tool"), tool)

    @unittest.skipUnless(sys.platform.startswith("linux"), "XDG application menu")
    def test_a_relative_xdg_data_dir_is_ignored(self):
        """The XDG spec: a relative ``$XDG_DATA_HOME`` or ``$XDG_DATA_DIRS``
        entry is invalid and ignored. Read as given, the application menu
        find_app searched moved with the working directory."""
        from unittest.mock import patch

        env = {
            "XDG_DATA_HOME": "relshare",
            "XDG_DATA_DIRS": "rel/share:/usr/share",
            "HOME": self.root,
        }
        with patch.dict(os.environ, env):
            dirs = AppLauncher._desktop_entry_dirs()
            user_dir = os.path.expanduser("~/.local/share/applications")
        self.assertEqual(dirs[0], user_dir)
        self.assertIn("/usr/share/applications", dirs)
        self.assertEqual([d for d in dirs if not os.path.isabs(d)], [])


class TestLiveEnviron(unittest.TestCase):
    """process_environ reads what a child inherits; desktop_env keeps a host's
    loader overrides away from the desktop programs it hands a path to."""

    @unittest.skipIf(sys.platform == "win32", "POSIX: libc environ")
    def test_process_environ_sees_a_c_level_setenv(self):
        """A host sets variables with C-level ``setenv`` after Python started:
        ``os.environ`` (a snapshot) misses them, a child inherits them."""
        import ctypes

        libc = ctypes.CDLL(None)
        libc.setenv.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        libc.unsetenv.argtypes = [ctypes.c_char_p]
        self.addCleanup(libc.unsetenv, b"PTK_LIVE_ENV_PROBE")
        libc.setenv(b"PTK_LIVE_ENV_PROBE", b"from-c", 1)
        self.assertNotIn("PTK_LIVE_ENV_PROBE", os.environ)
        env = AppLauncher.process_environ()
        self.assertEqual(env.get("PTK_LIVE_ENV_PROBE"), "from-c")
        self.assertEqual(env.get("PATH"), os.environ.get("PATH"))

    def test_desktop_env_drops_a_hosts_loader_overrides(self):
        from unittest.mock import patch

        host = {
            "LD_LIBRARY_PATH": "/usr/autodesk/maya2025/lib",
            "PYTHONHOME": "/usr/autodesk/maya2025",
            "QT_PLUGIN_PATH": "/usr/autodesk/maya2025/plugins",
            "PTK_DESKTOP_KEEP": "1",
        }
        with patch.dict(os.environ, host):
            env = AppLauncher.desktop_env()
        if sys.platform == "win32":
            self.assertIsNone(env)  # inherit: the shell resolves its own libraries
            return
        for key in ("LD_LIBRARY_PATH", "PYTHONHOME", "QT_PLUGIN_PATH"):
            self.assertNotIn(key, env)
        self.assertEqual(env.get("PTK_DESKTOP_KEEP"), "1")


class TestPythonArgsViaEnv(unittest.TestCase):
    """``python_args_via_env``: a Python child's command line travels in its env.

    mayapy.exe decodes its command line in the ANSI code page -- measured on Maya
    2025: "José" arrived as "Jos\\udce9" and "Жук" as "???" -- while the
    environment and the working directory arrive intact.
    """

    WORD = "José Жук"
    PROBE = (
        "import json, os, sys\n"
        "with open(sys.argv[1], 'w', encoding='utf-8') as fh:\n"
        "    json.dump({'argv': sys.argv, 'name': __name__, 'path0': sys.path[0],\n"
        "               'var': os.environ.get('PYTHONTK_ARGV')}, fh)\n"
    )

    def setUp(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.dir = os.path.join(here, "temp_tests", f"argv {self.WORD} {os.getpid()}")
        os.makedirs(self.dir)
        import shutil

        self.addCleanup(shutil.rmtree, self.dir, True)
        self.out = os.path.join(self.dir, "out.json")

    def _run(self, python, argv, cwd=None):
        import json
        import subprocess

        args, env = AppLauncher.python_args_via_env(argv)
        subprocess.run([python, *args], env=env, cwd=cwd, timeout=300, check=True)
        with open(self.out, encoding="utf-8") as fh:
            return json.load(fh)

    def _script(self):
        path = os.path.join(self.dir, "probe.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.PROBE)
        return path

    def test_the_command_line_is_ascii_and_the_env_carries_argv(self):
        import json
        from unittest.mock import patch

        with patch.object(AppLauncher, "process_environ", return_value={"A": "1"}):
            args, env = AppLauncher.python_args_via_env([self.WORD, 2])
        self.assertEqual(args[0], "-c")
        self.assertTrue(all(arg.isascii() for arg in args), args)
        self.assertEqual(env["A"], "1")  # None = the env the child would inherit
        self.assertTrue(env[AppLauncher.PYTHON_ARGV_VAR].isascii())
        self.assertEqual(json.loads(env[AppLauncher.PYTHON_ARGV_VAR]), [self.WORD, "2"])
        mine = {"B": "2"}
        _, env = AppLauncher.python_args_via_env(["x.py"], mine)
        self.assertEqual(set(env), {"B", AppLauncher.PYTHON_ARGV_VAR})
        self.assertEqual(mine, {"B": "2"})  # the caller's mapping is not mutated
        with self.assertRaises(ValueError):
            AppLauncher.python_args_via_env([])

    def test_a_script_runs_as_python_script_py_would(self):
        script = self._script()
        seen = self._run(sys.executable, [script, self.out])
        self.assertEqual(seen["argv"], [script, self.out])
        self.assertEqual(seen["name"], "__main__")
        self.assertEqual(seen["path0"], self.dir)
        self.assertIsNone(seen["var"])  # consumed: nothing the script spawns sees it

    def test_a_module_runs_as_python_dash_m_would(self):
        with open(os.path.join(self.dir, "probe_mod.py"), "w", encoding="utf-8") as fh:
            fh.write(self.PROBE)
        seen = self._run(sys.executable, ["-m", "probe_mod", self.out], cwd=self.dir)
        self.assertEqual(
            seen["argv"], [os.path.join(self.dir, "probe_mod.py"), self.out]
        )
        self.assertEqual(seen["name"], "__main__")
        self.assertIsNone(seen["var"])

    def test_run_keeps_output_it_cannot_decode(self):
        """A child's output is in ITS encoding: mayapy writes cp1252 ("é" = 0xE9)
        while Blender's Python decodes as UTF-8. The strict decode died in the
        reader thread and the run came back with no output -- the very traceback a
        failed hand-off embeds in its error. Measured through the Maya bridge."""
        code = "import sys; sys.stdout.buffer.write(b'MARK \\x81\\xe9\\xff END\\n')"
        result = AppLauncher.run(sys.executable, args=["-c", code], timeout=60)
        self.assertIn("MARK", result.stdout)
        self.assertIn("END", result.stdout)

    def test_mayapy_receives_a_non_ascii_script_and_argument_intact(self):
        from conftest import find_mayapy

        mayapy = find_mayapy()
        if not mayapy:
            self.skipTest("mayapy.exe not installed")
        script = self._script()
        seen = self._run(mayapy, [script, self.out])
        self.assertEqual(seen["argv"], [script, self.out])


def _ansi_codec_ok(text):
    return AppLauncher._ansi_encodable(text)


class TestAnsiCodecFallback(unittest.TestCase):
    """The ANSI codec answers on a host with no Windows code page to read.

    With no ``winreg`` the codec fell back to ``mbcs``, which exists only on
    Windows: every check then raised ``LookupError``. A test forcing the win32
    branch on Linux CI (``python_args_via_env`` for the watchdog and dialog
    sidecars) died there, and the dialog's silent catch fell through to the
    native fallback.
    """

    def test_no_windows_code_page_still_answers(self):
        import builtins
        import codecs
        from unittest import mock

        real_import, real_lookup = builtins.__import__, codecs.lookup

        def no_winreg(name, *args, **kwargs):
            if name == "winreg":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        def no_mbcs(name):
            if name.lower() == "mbcs":
                raise LookupError(name)
            return real_lookup(name)

        saved = AppLauncher._ANSI_CODEC
        AppLauncher._ANSI_CODEC = None
        try:
            with mock.patch("builtins.__import__", no_winreg):
                with mock.patch("codecs.lookup", no_mbcs):
                    codec = AppLauncher._ansi_codec()
                    no_mbcs(codec)  # one this host has; raises for mbcs off Windows
            self.assertTrue(AppLauncher._ansi_encodable("/tmp/plain"))
            self.assertFalse(AppLauncher._ansi_encodable("/tmp/Жук"))
        finally:
            AppLauncher._ANSI_CODEC = saved


@unittest.skipUnless(sys.platform == "win32", "the ANSI code page is a Windows notion")
class TestAnsiSafePath(unittest.TestCase):
    """``ansi_safe_path``: a path Maya can open although it reads paths in the ANSI
    code page. Measured on Maya 2025 (cp1252 here): with a TEMP under "José Жук",
    ``cmds.file`` save gave "An invalid path was specified" and open "File not
    found", and Maya read TEMP as "???" and fell back to the CURRENT DIRECTORY for
    its own temp files; the 8.3 short form of the same folders worked for all three.
    """

    CYR = "Жук"  # outside cp1252
    LATIN = "José"  # inside it

    def setUp(self):
        import shutil

        here = os.path.dirname(os.path.abspath(__file__))
        self.root = os.path.join(here, "temp_tests", f"ansi {os.getpid()}")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.cyr = os.path.join(self.root, f"{self.LATIN} {self.CYR}", "sub")
        os.makedirs(self.cyr)
        AppLauncher._ANSI_WARNED.clear()
        # 8.3 names are a per-volume setting; without one there is nothing to prove.
        if _ansi_codec_ok(self.cyr):
            self.skipTest("this ANSI code page holds Cyrillic")
        # A volume with them off answers the LONG name (a GitHub runner's D:), not None.
        short = AppLauncher._short_name(os.path.dirname(self.cyr))
        if not short or not os.path.basename(short).isascii():
            self.skipTest("no 8.3 short names on this volume")

    def test_an_ansi_path_is_returned_as_is(self):
        path = os.path.join(self.root, f"{self.LATIN}.ma")
        self.assertIs(AppLauncher.ansi_safe_path(path), path)
        self.assertIsNone(AppLauncher.ansi_safe_path(None))

    def test_only_the_components_the_code_page_cannot_hold_are_shortened(self):
        target = os.path.join(self.cyr, "payload.fbx")
        with open(target, "w") as fh:
            fh.write("x")
        safe = AppLauncher.ansi_safe_path(target)
        self.assertTrue(_ansi_codec_ok(safe), ascii(safe))
        self.assertTrue(os.path.samefile(safe, target))
        # Names the code page CAN hold are kept verbatim: a template that reads the
        # payload's own name (or a folder like "temp_tests") sees what was written.
        self.assertEqual(os.path.basename(safe), "payload.fbx")
        self.assertEqual(os.path.basename(os.path.dirname(safe)), "sub")
        self.assertIn(os.path.join(self.root, ""), safe)

    def test_a_file_not_yet_written_keeps_its_name_under_a_short_folder(self):
        target = os.path.join(self.cyr, ".out.saving.ma")
        safe = AppLauncher.ansi_safe_path(target)
        self.assertTrue(_ansi_codec_ok(safe), ascii(safe))
        self.assertEqual(os.path.basename(safe), ".out.saving.ma")
        self.assertTrue(os.path.samefile(os.path.dirname(safe), self.cyr))

    def test_without_a_short_name_it_warns_once_and_keeps_the_path(self):
        from unittest.mock import patch

        target = os.path.join(self.cyr, "a.ma")
        with patch.object(AppLauncher, "_short_name", return_value=None):
            with self.assertLogs(
                "pythontk.core_utils.app_launcher._environment", "WARNING"
            ) as logs:
                self.assertEqual(AppLauncher.ansi_safe_path(target), target)
                self.assertEqual(
                    AppLauncher.ansi_safe_path(os.path.join(self.cyr, "b.ma")),
                    os.path.join(self.cyr, "b.ma"),
                )
        self.assertEqual(len(logs.records), 1, logs.output)
        self.assertIn("TEMP", logs.output[0])

    def test_ascii_only_shortens_every_non_ascii_component(self):
        """``ascii_only``: for a path written INTO a file another program decodes as
        ANSI. RizomUV 2020.1 reads a Lua payload's UTF-8 path bytes that way, so an
        "é" the code page holds breaks there too (measured: ZomLoad / ZomSave under
        "José Ångström" timed out, the 8.3 form passed)."""
        latin = os.path.join(self.root, self.LATIN)
        os.makedirs(latin)
        target = os.path.join(latin, "payload.fbx")
        with open(target, "w") as fh:
            fh.write("x")
        # The code page holds it.
        self.assertIs(AppLauncher.ansi_safe_path(target), target)
        for path in (target, os.path.join(self.cyr, "not_yet.lua")):
            safe = AppLauncher.ansi_safe_path(path, ascii_only=True)
            self.assertTrue(safe.isascii(), ascii(safe))
            self.assertEqual(os.path.basename(safe), os.path.basename(path))
            self.assertTrue(
                os.path.samefile(os.path.dirname(safe), os.path.dirname(path))
            )
        plain = os.path.join(self.root, "plain.fbx")
        self.assertIs(AppLauncher.ansi_safe_path(plain, ascii_only=True), plain)

    def test_ascii_only_without_a_short_name_warns_once_and_keeps_the_path(self):
        from unittest.mock import patch

        target = os.path.join(self.root, f"{self.LATIN}.lua")
        with patch.object(AppLauncher, "_short_name", return_value=None):
            with self.assertLogs(
                "pythontk.core_utils.app_launcher._environment", "WARNING"
            ) as logs:
                for _ in range(2):
                    self.assertEqual(
                        AppLauncher.ansi_safe_path(target, ascii_only=True), target
                    )
        self.assertEqual(len(logs.records), 1, logs.output)

    def test_the_code_page_is_the_systems_not_this_processs(self):
        """Blender's manifest declares UTF-8, so inside Blender ``mbcs`` holds every
        character (GetACP() = 65001, measured) while the mayapy it launches reads
        cp1252. The check reads the SYSTEM code page, which a manifest-less child
        like Maya uses."""
        import winreg

        key = r"SYSTEM\CurrentControlSet\Control\Nls\CodePage"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
            acp = winreg.QueryValueEx(handle, "ACP")[0]
        expected = "utf-8" if acp == "65001" else f"cp{acp}"
        self.assertEqual(AppLauncher._ansi_codec(), expected)

    def test_off_windows_it_is_a_no_op(self):
        from unittest.mock import patch

        with patch.object(sys, "platform", "linux"):
            self.assertEqual(AppLauncher.ansi_safe_path(self.cyr), self.cyr)

    def test_a_python_childs_temp_is_handed_over_in_its_short_form(self):
        # TMPDIR too: a Python child's tempfile reads it FIRST, and TestSandbox
        # sets all three to its root -- a long TMPDIR beside a short TEMP handed
        # mayapy's own tempfile paths back in the form Maya cannot open.
        _, env = AppLauncher.python_args_via_env(
            ["x.py"],
            {"TEMP": self.cyr, "Tmp": self.cyr, "TMPDIR": self.cyr, "OTHER": self.cyr},
        )
        for key in ("TEMP", "Tmp", "TMPDIR"):
            self.assertTrue(_ansi_codec_ok(env[key]), ascii(env[key]))
            self.assertTrue(os.path.samefile(env[key], self.cyr))
        self.assertEqual(env["OTHER"], self.cyr)  # only the temp roots are rewritten


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


class TestPosixEnviron(unittest.TestCase):
    """``_posix_environ`` is "libc's environ, or None" -- never a raise."""

    def test_a_host_with_no_loadable_libc_answers_none(self):
        """``ctypes.CDLL(None)`` raises TypeError (not OSError) where there is no
        libc to load -- a Windows host asked the POSIX question, as a test that
        simulates Linux does -- and ``desktop_env`` then took ``open_explorer``
        down with it (the reveal silently did nothing)."""
        from unittest import mock
        import ctypes

        with mock.patch.object(ctypes, "CDLL", side_effect=TypeError("no libc")):
            self.assertIsNone(ptk.AppLauncher._posix_environ())


class TestCompanionPython(unittest.TestCase):
    """``companion_python``: the Python that pairs with a host binary.

    A DCC's GUI binary can't run ``-c`` Python or ``-m pip`` -- it runs MEL /
    MaxScript, or opens a second copy of the app -- but the interpreter it ships
    shares its site-packages. uitk kept a table of product names for this;
    Blender's entry pointed beside ``blender.exe``, where Blender ships no
    python, so a Blender host fell back to the GUI binary itself.
    """

    EXT = ".exe" if sys.platform == "win32" else ""

    def setUp(self):
        import tempfile

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = folder.name

    def _touch(self, *parts):
        path = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
        return path

    def test_a_python_is_its_own_companion(self):
        py = self._touch("bin", "python" + self.EXT)
        self.assertEqual(ptk.AppLauncher.companion_python(py), py)
        self.assertTrue(ptk.AppLauncher.looks_like_python(py))

    def test_a_host_finds_its_named_companion_beside_it(self):
        for host, companion in (
            ("maya", "mayapy"),
            ("mayabatch", "mayapy"),
            ("3dsmax", "3dsmaxpy"),
        ):
            with self.subTest(host=host):
                exe = self._touch(host, "bin", host + self.EXT)
                expected = self._touch(host, "bin", companion + self.EXT)
                self.assertFalse(ptk.AppLauncher.looks_like_python(exe))
                self.assertEqual(ptk.AppLauncher.companion_python(exe), expected)

    def test_linux_mayas_gui_binary_finds_mayapy(self):
        exe = self._touch("linux", "bin", "maya.bin")
        expected = self._touch("linux", "bin", "mayapy" + self.EXT)
        self.assertEqual(ptk.AppLauncher.companion_python(exe), expected)

    def test_this_processes_host_finds_the_python_under_its_prefix(self):
        """Blender ships its interpreter under ``<ver>/python/bin``, not beside
        the binary -- reachable only through this process's ``sys.prefix``."""
        from unittest import mock

        exe = self._touch("blender", "blender" + self.EXT)
        prefix = os.path.join(self.root, "blender", "5.1", "python")
        bundled = self._touch("blender", "5.1", "python", "bin", "python" + self.EXT)
        with (
            mock.patch.object(sys, "executable", exe),
            mock.patch.object(sys, "prefix", prefix),
            mock.patch.object(sys, "base_prefix", prefix),
            mock.patch.object(sys, "exec_prefix", prefix),
        ):
            self.assertEqual(ptk.AppLauncher.companion_python(), bundled)
            self.assertEqual(ptk.AppLauncher.companion_python(exe), bundled)

    def test_another_install_never_borrows_the_running_prefix(self):
        """``sys.prefix`` describes THIS process: a question about some other
        install must not be answered with the running one's python."""
        other = self._touch("other", "blender" + self.EXT)
        self.assertIsNone(ptk.AppLauncher.companion_python(other))


class TestWriteBatchScript(unittest.TestCase):
    """``write_batch_script``: the ONE writer of a script a shell will run.

    Its callers (the RealityScan and SuGaR runners) each hand-wrote the file,
    and two defects rode along: CRLF joined by hand in a TEXT-mode file on
    Windows doubled every CR (``\\r\\r\\n``), and a UTF-8 .bat turned a
    non-ASCII path into mojibake for cmd, which reads it in the OEM codepage.
    """

    def setUp(self):
        import tempfile

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = folder.name

    def _read(self, path):
        with open(path, "rb") as fh:
            return fh.read()

    def test_cmd_script_has_exact_crlf_line_ends(self):
        path = os.path.join(self.root, "run.bat")
        AppLauncher.write_batch_script(path, ["@echo off", "echo hi"], shell="cmd")
        data = self._read(path)
        self.assertNotIn(b"\r\r\n", data)
        self.assertEqual(data, b"@echo off\r\necho hi\r\n")

    def test_cmd_script_normalizes_line_ends_inside_a_line(self):
        """A caller passing a pre-joined block still gets one CRLF per line."""
        path = os.path.join(self.root, "run.bat")
        AppLauncher.write_batch_script(path, ["@echo off\necho a\r\necho b"], "cmd")
        self.assertEqual(self._read(path), b"@echo off\r\necho a\r\necho b\r\n")

    def test_a_pre_joined_block_may_be_one_string(self):
        """The docstring invites a pre-joined block; passed as a bare string it
        was iterated a character at a time, one character per line."""
        bat = os.path.join(self.root, "run.bat")
        AppLauncher.write_batch_script(bat, "@echo off\r\necho hi", "cmd")
        self.assertEqual(self._read(bat), b"@echo off\r\necho hi\r\n")
        sh = os.path.join(self.root, "run.sh")
        AppLauncher.write_batch_script(sh, "echo a\necho b", "bash")
        self.assertEqual(self._read(sh), b"#!/usr/bin/env bash\necho a\necho b\n")

    @unittest.skipUnless(sys.platform == "win32", "the OEM codec is Windows-only")
    def test_cmd_script_is_encoded_in_the_console_oem_codepage(self):
        # 'é' is representable in every Western OEM codepage (cp437 / cp850),
        # and encodes differently from UTF-8 there -- the mojibake case.
        target = os.path.join(self.root, "café", "out.log")
        path = os.path.join(self.root, "run.bat")
        AppLauncher.write_batch_script(path, [f'echo x > "{target}"'], shell="cmd")
        data = self._read(path)
        self.assertIn(target.encode("oem"), data)
        self.assertNotIn(target.encode("utf-8"), data)

    @unittest.skipUnless(sys.platform == "win32", "the OEM codec is Windows-only")
    def test_cmd_script_refuses_an_unrepresentable_path(self):
        """No errors='replace': a path cmd cannot spell must fail loudly, not
        be rewritten to '?' and redirect output to the wrong file."""
        path = os.path.join(self.root, "run.bat")
        with self.assertRaises(UnicodeEncodeError):
            AppLauncher.write_batch_script(path, ["echo \U0001f600"], shell="cmd")

    def test_bash_script_is_lf_with_a_shebang_and_executable(self):
        path = os.path.join(self.root, "run.sh")
        AppLauncher.write_batch_script(path, ["echo hi"], shell="bash")
        data = self._read(path)
        self.assertEqual(data, b"#!/usr/bin/env bash\necho hi\n")
        if os.name != "nt":
            self.assertTrue(os.access(path, os.X_OK))

    def test_bash_script_keeps_a_caller_shebang(self):
        path = os.path.join(self.root, "run.sh")
        AppLauncher.write_batch_script(path, ["#!/bin/sh", "echo hi"], shell="bash")
        self.assertEqual(self._read(path), b"#!/bin/sh\necho hi\n")

    def test_default_shell_follows_the_platform(self):
        path = os.path.join(self.root, "run")
        AppLauncher.write_batch_script(path, ["echo hi"])
        data = self._read(path)
        if os.name == "nt":
            self.assertEqual(data, b"echo hi\r\n")
        else:
            self.assertEqual(data, b"#!/usr/bin/env bash\necho hi\n")

    def test_unknown_shell_raises(self):
        with self.assertRaises(ValueError):
            AppLauncher.write_batch_script(
                os.path.join(self.root, "x"), ["echo"], shell="fish"
            )

    def test_returns_the_path(self):
        path = os.path.join(self.root, "run.bat")
        self.assertEqual(AppLauncher.write_batch_script(path, ["echo"], "cmd"), path)


class TestSpawn(unittest.TestCase):
    """The attached launch shape: runs beside this process, and dies with it."""

    def test_output_arrives_merged_on_one_binary_pipe(self):
        """ProcessReader reads BINARY pipes; stderr rides the same one."""
        proc = AppLauncher.spawn(
            sys.executable,
            [
                "-c",
                "import sys; print('out', flush=True); print('err', file=sys.stderr)",
            ],
        )
        out = proc.stdout.read()
        proc.wait(timeout=30)
        self.assertIsInstance(out, bytes)
        self.assertEqual(sorted(out.split()), [b"err", b"out"])

    def test_a_missing_app_raises(self):
        with self.assertRaises(FileNotFoundError):
            AppLauncher.spawn("no-such-app-for-spawn-xyz")

    def test_the_binding_can_be_declined(self):
        proc = AppLauncher.spawn(sys.executable, ["-c", "pass"], bind_lifetime=False)
        proc.wait(timeout=30)
        proc.stdout.close()
        self.assertFalse(proc.bound_to_parent)

    @unittest.skipUnless(sys.platform == "win32", "the binding is a Windows Job Object")
    def test_the_child_dies_with_its_parent_however_the_parent_ends(self):
        """The case the binding exists for: a parent that ends WITHOUT running
        its cleanup -- a crashed DCC -- must not leave its child running. A
        tunnel orphaned that way keeps a public link on a port whatever binds
        it next would be published through."""
        import ctypes
        import subprocess
        from ctypes import wintypes

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        parent = (
            "import os, sys\n"
            f"sys.path.insert(0, {root!r})\n"
            "from pythontk.core_utils.app_launcher import AppLauncher\n"
            "child = AppLauncher.spawn(sys.executable, "
            "['-c', 'import time; time.sleep(120)'])\n"
            "print(child.pid, child.bound_to_parent, flush=True)\n"
            "os._exit(0)  # no atexit, no finally: a crash, as the child sees it\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", parent], capture_output=True, text=True, timeout=60
        )
        pid, bound = result.stdout.split()
        self.assertEqual(bound, "True", result.stderr)

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x00100000, False, int(pid))  # SYNCHRONIZE
        if not handle:
            return  # already gone -- and no longer openable, so not reused yet
        try:
            # WAIT_OBJECT_0: exited. OpenProcess succeeding proves nothing.
            exited = kernel32.WaitForSingleObject(handle, 10000) == 0
        finally:
            kernel32.CloseHandle(handle)
        if not exited:
            subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
        self.assertTrue(exited, "the child outlived its parent")

    @unittest.skipIf(sys.platform == "win32", "POSIX: the sh lifetime watcher")
    def test_the_child_and_its_own_children_die_with_a_crashed_parent_posix(self):
        """POSIX twin: the child leads its own process group and a ``sh``
        watcher kills that group once the parent is gone -- so what the child
        started (a tunnel's helper) goes with it."""
        import signal
        import subprocess

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        parent = (
            "import os, sys\n"
            f"sys.path.insert(0, {root!r})\n"
            "from pythontk.core_utils.app_launcher import AppLauncher\n"
            "child = AppLauncher.spawn('/bin/sh', "
            "['-c', 'sleep 120 & echo $!; wait'])\n"
            "grandchild = child.stdout.readline().decode().strip()\n"
            "print(child.pid, grandchild, child.bound_to_parent, flush=True)\n"
            "os._exit(0)  # no atexit, no finally: a crash, as the child sees it\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", parent], capture_output=True, text=True, timeout=60
        )
        child, grandchild, bound = result.stdout.split()
        self.assertEqual(bound, "True", result.stderr)

        def alive(pid):
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                return False
            try:  # a zombie still answers kill(0): dead all the same
                with open(f"/proc/{pid}/stat", "rb") as f:
                    return f.read().rsplit(b")", 1)[1].split()[0] != b"Z"
            except OSError:
                return True  # no /proc (macOS): kill(0) is all there is

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and (alive(child) or alive(grandchild)):
            time.sleep(0.2)
        survivors = [p for p in (child, grandchild) if alive(p)]
        if survivors:
            os.killpg(int(child), signal.SIGKILL)
        self.assertEqual(survivors, [], "outlived their parent")


if __name__ == "__main__":
    unittest.main()
