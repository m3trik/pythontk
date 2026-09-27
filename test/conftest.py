#!/usr/bin/python
# coding=utf-8
"""
Pytest configuration and shared fixtures for pythontk tests.

This module provides:
- Shared test utilities and base classes
- Common fixtures for all test modules
- Path management for test resources
"""

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

# Enable OpenCV's EXR codec before any test module imports cv2 (OpenCV caches
# this flag at first codec init). Mirrors the production setting in ImgUtils.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# Process-level isolation -- no real browser launch, one throwaway temp root --
# before the first temp allocation. Import-time, not a fixture; and ALSO in
# ``run_tests.py``, which discovers with ``unittest`` and never loads this file.
from pythontk.core_utils.test_sandbox import TestSandbox  # noqa: E402

TestSandbox.activate()


# =============================================================================
# Session gates
# =============================================================================

#: Env var carrying how many tests pytest collected, for the collector drift
#: guard in test_test_sandbox.py. An env var rather than a module global
#: because pytest imports conftest under its own name, so a test doing
#: ``import conftest`` gets a DIFFERENT module object and would read None.
#: Absent under the unittest runner, which never loads this file.
COLLECTED_COUNT_ENV = "PYTHONTK_PYTEST_COLLECTED"


def pytest_collection_modifyitems(session, config, items):
    """Record the collected count, but only for a whole-directory run.

    A subset run (``pytest test/test_x.py``) collects a handful of tests while
    unittest discovery still walks everything, so recording it would make the
    drift guard fail on every targeted run. Absent means "no comparable
    number", which the guard skips on.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    args = [a.split("::", 1)[0] for a in config.args]
    whole_dir = bool(args) and all(
        os.path.abspath(a).rstrip("\\/") == here for a in args
    )
    os.environ.pop(COLLECTED_COUNT_ENV, None)
    if whole_dir:
        os.environ[COLLECTED_COUNT_ENV] = str(len(items))


def pytest_sessionfinish(session, exitstatus):
    """Fail the run when the sandbox refused a browser launch and the code
    under test swallowed it.

    ``run_tests.py`` has always done this, but CI runs bare ``pytest test/``
    (``.github/workflows/tests.yml``) and never loads that runner -- so on the
    path that actually gates a PR, five downstream suites armed the guard and
    nothing read the record. A refused launch that a broad ``except`` ate
    leaves its test green; in production the same call opened a tab.

    It sets ``session.exitstatus`` rather than printing. A print gates
    nothing, which is the mistake this replaces.
    """
    leaked = list(TestSandbox.launches)
    if not leaked:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    message = (
        f"{len(leaked)} browser launch(es) blocked by the sandbox and swallowed "
        "by the code under test: " + ", ".join(leaked)
    )
    if reporter is not None:
        reporter.write_sep("=", "BROWSER LEAK", red=True)
        reporter.write_line(message)
    else:  # -p no:terminal, or a very early failure
        print("[ERROR] " + message)
    # 0 and 5 (no tests collected) are the statuses a leak must override;
    # a real test failure already fails and keeps its own, more specific code.
    if exitstatus in (0, 5):
        session.exitstatus = 1


# =============================================================================
# Test Utilities & Base Classes
# =============================================================================


class TestPaths:
    """Centralized test path management."""

    BASE_DIR = Path(__file__).parent
    TEST_FILES_DIR = BASE_DIR / "test_files"
    IMGTK_TEST_DIR = TEST_FILES_DIR / "imgtk_test"

    @classmethod
    def get(cls, *parts: str) -> str:
        """Get absolute path to a test file."""
        return str(cls.TEST_FILES_DIR.joinpath(*parts))

    @classmethod
    def get_imgtk(cls, filename: str) -> str:
        """Get path to an image test file."""
        return str(cls.IMGTK_TEST_DIR / filename)


class X11Window:
    """A real top-level X11 window owned by a child process -- the fixture for
    window lookups by PID, with no GUI toolkit needed.

    Maps a window, sets ``_NET_WM_PID`` and ``WM_NAME`` as Qt/GTK apps (Maya,
    Blender, browsers) do -- Tk 9 sets no ``_NET_WM_PID`` -- prints ``up``, and
    holds the window until its stdin closes. Run :meth:`script` with
    ``python -c``.
    """

    _SCRIPT = """
import ctypes, os, sys
x = ctypes.cdll.LoadLibrary("libX11.so.6")
vp, ul, i = ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int
for name, args, res in (
    ("XOpenDisplay", [ctypes.c_char_p], vp),
    ("XDefaultRootWindow", [vp], ul),
    ("XCreateSimpleWindow", [vp, ul, i, i, ctypes.c_uint, ctypes.c_uint,
                             ctypes.c_uint, ul, ul], ul),
    ("XInternAtom", [vp, ctypes.c_char_p, i], ul),
    ("XChangeProperty", [vp, ul, ul, ul, i, i, vp, i], i),
    ("XStoreName", [vp, ul, ctypes.c_char_p], i),
    ("XMapWindow", [vp, ul], i),
    ("XSync", [vp, i], i),
):
    fn = getattr(x, name)
    fn.argtypes, fn.restype = args, res
d = x.XOpenDisplay(None)
w = x.XCreateSimpleWindow(d, x.XDefaultRootWindow(d), 10, 10, 120, 80, 0, 0, 0)
pid = ctypes.c_long(os.getpid())  # format-32 data travels as C longs
x.XChangeProperty(d, w, x.XInternAtom(d, b"_NET_WM_PID", 0), 6, 32, 0,
                  ctypes.byref(pid), 1)
x.XStoreName(d, w, TITLE.encode())
x.XMapWindow(d, w)
x.XSync(d, 0)
print("up", flush=True)
sys.stdin.readline()
"""

    @classmethod
    def script(cls, title: str) -> str:
        return f"TITLE = {title!r}\n" + cls._SCRIPT


class BaseTestCase(unittest.TestCase):
    """Base test case with common utilities and assertions."""

    @staticmethod
    def replace_mem_address(obj: object) -> str:
        """Normalize memory addresses in string representations for comparison.

        Parameters:
            obj: Object to convert and normalize.

        Returns:
            String with memory addresses replaced by '0x00000000000'.

        Example:
            >>> replace_mem_address("<Widget at 0x1ebe2677e80>")
            "<Widget at 0x00000000000>"
        """
        return re.sub(r"0x[a-fA-F\d]+", "0x00000000000", str(obj))

    def assertImageMode(self, image, expected_mode: str, msg: str = None):
        """Assert that a PIL Image has the expected mode."""
        self.assertEqual(image.mode, expected_mode, msg)

    def assertImageSize(self, image, expected_size: tuple, msg: str = None):
        """Assert that a PIL Image has the expected size."""
        self.assertEqual(image.size, expected_size, msg)

    def assertPathExists(self, path: str, msg: str = None):
        """Assert that a file or directory exists."""
        self.assertTrue(os.path.exists(path), msg or f"Path does not exist: {path}")

    #: Executes a file the way ``m3trik/scripts/generate_api_registry.py`` does
    #: (off disk, registered before exec), with ``import pythontk`` failing, and
    #: reports the top-level modules the exec pulled in beyond the stdlib (the
    #: list needs ``sys.stdlib_module_names``, 3.10+).
    _STANDALONE_LOAD = """
import importlib.util, json, sys
sys.modules["pythontk"] = None
before = set(sys.modules)
spec = importlib.util.spec_from_file_location("_standalone", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
try:
    spec.loader.exec_module(module)
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}", "foreign": []}))
    raise SystemExit(0)
stdlib = getattr(sys, "stdlib_module_names", None)
roots = {name.partition(".")[0] for name in set(sys.modules) - before}
foreign = sorted(roots - {spec.name} - set(stdlib)) if stdlib else []
print(json.dumps({"error": None, "foreign": foreign}))
"""

    def assertLoadsStandalone(self, path: str):
        """Assert that *path* imports with the stdlib alone, pythontk absent.

        For a module ``generate_api_registry.py`` loads off disk, on a CI box
        with nothing installed: executed in a fresh interpreter where
        ``import pythontk`` fails.
        """
        proc = subprocess.run(
            [sys.executable, "-c", self._STANDALONE_LOAD, str(path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertIsNone(result["error"], f"{path} does not load standalone")
        self.assertEqual(result["foreign"], [], f"{path} imports beyond the stdlib")
