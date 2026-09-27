# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.x11 (the ctypes Xlib client).

Two halves: a recording libX11 stand-in drives the connection, typing, locking
and caching logic on ANY OS (pythontk's CI runs Windows too), and a real X
server -- Linux with a DISPLAY, e.g. under Xvfb -- checks the queries end to end.
"""

import contextlib
import ctypes
import ctypes.util
import os
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import pythontk
from pythontk.core_utils.x11 import X11

from conftest import X11Window


class _XFunc:
    """One libX11 function: records the ctypes types each call ran under."""

    def __init__(self, name, calls, result=0, on_call=None):
        self.name, self.calls, self.result, self.on_call = name, calls, result, on_call
        self.argtypes = None
        self.restype = None  # ctypes' default: a C int

    def __call__(self, *args):
        self.calls.append((self.name, self.argtypes, self.restype))
        if self.on_call:
            self.on_call(*args)
        return self.result


class _RecordingX11:
    """A libX11 stand-in: every function records the types it was called with.

    *on_first_setup* runs on the first access of ``XOpenDisplay`` -- the client
    typing the library, the window a second thread must not see into.
    """

    def __init__(self, display=0x7F0012345678, on_first_setup=None, on_open=None):
        self.calls = []
        self._on_first_setup = on_first_setup
        self._funcs = {
            "XOpenDisplay": _XFunc("XOpenDisplay", self.calls, display, on_open),
            "XKeysymToKeycode": _XFunc("XKeysymToKeycode", self.calls, 9),
        }

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "XOpenDisplay":
            hook, self._on_first_setup = self._on_first_setup, None
            if hook:
                hook()
        return self._funcs.setdefault(name, _XFunc(name, self.calls))

    def calls_to(self, name):
        return [c for c in self.calls if c[0] == name]


@contextlib.contextmanager
def _fresh_x11(loader):
    """X11 on any OS, from a never-connected state, through *loader*."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(ctypes.cdll, "LoadLibrary", loader))
        for name, value in (("_lib", None), ("_display", None), ("_atoms", {})):
            stack.enter_context(patch.object(X11, name, value))
        yield


class TestX11Client(unittest.TestCase):
    """Connection, typing, locking and caching -- no X server needed."""

    def test_two_threads_only_ever_call_a_typed_library(self):
        """Several threads poll at once (a CancelScope's Esc-hold source and the
        monitor's worker). A second thread arriving while the first is typing
        the library must never call it untyped: an untyped XOpenDisplay returns
        the Display* as a C int, cached truncated for the process's lifetime."""
        second_opened = threading.Event()
        second_result = []
        second = threading.Thread(
            target=lambda: second_result.append(X11.key_down(0xFF1B))
        )

        def start_second_thread():
            second.start()
            second_opened.wait(0.5)  # the lock holds it off: this times out

        x11 = _RecordingX11(
            on_first_setup=start_second_thread,
            on_open=lambda *_: second_opened.set(),
        )
        with _fresh_x11(lambda name: x11):
            self.assertFalse(X11.key_down(0xFF1B))
            second.join(5)
        self.assertEqual(second_result, [False])  # the second thread ran to the end
        opens = x11.calls_to("XOpenDisplay")
        self.assertEqual(len(opens), 1, "one Display per process")
        self.assertEqual(opens[0][2], ctypes.c_void_p)
        for _, argtypes, _ in x11.calls_to("XKeysymToKeycode"):
            self.assertEqual(argtypes, [ctypes.c_void_p, ctypes.c_ulong])

    def test_a_failed_open_or_load_is_not_retried(self):
        """Callers poll several times a second, and XOpenDisplay against an
        unreachable server can block: no X server (pure Wayland, SSH) and no
        libX11 are each discovered once."""
        x11 = _RecordingX11(display=None)  # XOpenDisplay returned NULL
        with _fresh_x11(lambda name: x11):
            for _ in range(3):
                self.assertIsNone(X11.key_down(0xFF1B))
                self.assertFalse(X11.available())
        self.assertEqual(len(x11.calls_to("XOpenDisplay")), 1)

        loader = MagicMock(side_effect=OSError("libX11.so.6: cannot open"))
        with _fresh_x11(loader):
            for _ in range(3):
                self.assertIsNone(X11.window_titles([os.getpid()]))
        self.assertEqual(loader.call_count, 1)

    def test_key_down_reads_the_keycodes_bit(self):
        """Keycode 9 is byte 1, bit 1 of XQueryKeymap's 32-byte vector."""
        x11 = _RecordingX11()
        for pressed in (True, False):
            with self.subTest(pressed=pressed):

                def keymap(display, keys, pressed=pressed):
                    keys[1] = b"\x02" if pressed else b"\x00"

                x11._funcs["XQueryKeymap"] = _XFunc(
                    "XQueryKeymap", x11.calls, 1, keymap
                )
                with _fresh_x11(lambda name: x11):
                    self.assertIs(X11.key_down(0xFF1B), pressed)

    def test_a_session_installs_and_restores_the_error_handler(self):
        """Xlib's DEFAULT error handler exits the process (a window vanishing
        mid-query raises BadWindow): ours is in place while we talk to the
        server, and whatever was installed before is back afterwards."""
        previous = object()
        x11 = _RecordingX11()
        installed = []
        handler = x11.XSetErrorHandler
        handler.on_call = installed.append
        handler.result = previous
        with _fresh_x11(lambda name: x11):
            X11.key_down(0xFF1B)
        self.assertEqual(len(installed), 2)
        self.assertIs(installed[0], X11._error_handler)
        self.assertIs(installed[1], previous)


def _x_server_open():
    """True on Linux when libX11 loads and the DISPLAY server answers."""
    if not sys.platform.startswith("linux") or not os.environ.get("DISPLAY"):
        return False
    if not ctypes.util.find_library("X11"):
        return False
    return X11.available()


@unittest.skipUnless(_x_server_open(), "needs a reachable X server (Linux + DISPLAY)")
class TestX11Server(unittest.TestCase):
    """Against a real X server (a bare Xvfb in CI: no window manager, no
    compositor)."""

    def _child_env(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(pythontk.__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
        return env

    def test_a_worker_thread_call_does_not_crash_the_process(self):
        """The Esc probe runs on the monitor's worker thread, where libX11's
        Display* lives above 4 GB: untyped, the truncated pointer was a SIGSEGV.
        A child process, because that failure takes the process down."""
        code = (
            "import threading\n"
            "from pythontk.core_utils.x11 import X11\n"
            "out = []\n"
            "t = threading.Thread(target=lambda: out.append(X11.key_down(0xFF1B)))\n"
            "t.start(); t.join()\n"
            "print('result', out[0])\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            env=self._child_env(),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("result False", proc.stdout)  # nobody is holding Esc

    def test_window_titles_finds_a_real_windows_title_by_its_pid(self):
        """A client's top-level window is found through ``_NET_WM_PID`` -- on
        a bare server with no window manager, among the root's viewable
        children. Other PIDs see nothing of it."""
        title = f"ptk-x11-probe-{os.getpid()}"
        child = subprocess.Popen(
            [sys.executable, "-c", X11Window.script(title)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=self._child_env(),
        )
        try:
            self.assertEqual(child.stdout.readline().strip(), "up")
            deadline = time.monotonic() + 10
            titles = []
            while time.monotonic() < deadline:
                titles = X11.window_titles([child.pid]) or []
                if title in titles:
                    break
                time.sleep(0.1)
            self.assertIn(title, titles)
            self.assertNotIn(title, X11.window_titles([os.getpid()]))
        finally:
            child.stdin.close()
            child.wait(10)

    def test_pointer_is_a_screen_position(self):
        pos = X11.pointer()
        self.assertIsInstance(pos, tuple)
        self.assertEqual(len(pos), 2)
        self.assertTrue(all(isinstance(v, int) for v in pos))

    def test_compositor_and_focus_answer_without_raising(self):
        """A bare server has neither, and says so rather than guessing."""
        self.assertIn(X11.has_compositor(), (True, False))
        pid = X11.active_window_pid()
        self.assertTrue(pid is None or isinstance(pid, int))


if __name__ == "__main__":
    unittest.main()
