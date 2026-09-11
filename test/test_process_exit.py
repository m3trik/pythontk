# !/usr/bin/python
# coding=utf-8
"""Tests for :class:`pythontk.ProcessExit` -- the DLL-detach-free process exit.

Every assertion runs in a CHILD process: the whole point of the primitive is
that the calling process does not survive it, so the only way to observe a
result is the child's exit status and the output it managed to flush.

These run under a PLAIN interpreter, where ``DLL_PROCESS_DETACH`` has nothing
hostile to run, so they pin the CONTRACT (the code survives, buffered output
survives, no atexit hook runs) rather than the crash avoidance -- that half is
host-specific and is verified under ``mayapy``.  See the module docstring of
``pythontk/core_utils/process_exit.py`` for the measurement.
"""

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk.core_utils.process_exit import ProcessExit


def _child(body: str, timeout: int = 60):
    """Run *body* in a child interpreter; return (returncode, stdout)."""
    proc = subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    return proc.returncode, proc.stdout


_PRELUDE = (
    "import sys, os\n"
    f"sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})\n"
    "from pythontk.core_utils.process_exit import ProcessExit\n"
)


class TestHardExitCode(unittest.TestCase):
    """The requested status reaches the parent unchanged."""

    def test_nonzero_code_survives(self):
        rc, _ = _child(_PRELUDE + "ProcessExit.hard_exit(7)")
        self.assertEqual(rc, 7)

    def test_zero_code_survives(self):
        rc, _ = _child(_PRELUDE + "ProcessExit.hard_exit(0)")
        self.assertEqual(rc, 0)

    def test_failure_code_survives(self):
        # 1 is what every runner returns for "the suite failed" — the value
        # this primitive exists to deliver intact.
        rc, _ = _child(_PRELUDE + "ProcessExit.hard_exit(1)")
        self.assertEqual(rc, 1)


class TestHardExitFlushes(unittest.TestCase):
    """Buffered output is flushed before the process dies.

    ``TerminateProcess`` drops whatever is still in the interpreter's buffers,
    so the flush is part of the contract, not a convenience.  A child whose
    stdout is a PIPE is block-buffered, which is exactly the case that loses
    the text without it.
    """

    def test_buffered_stdout_is_flushed(self):
        rc, out = _child(
            _PRELUDE + "print('RESULT_LINE', end='')\nProcessExit.hard_exit(3)"
        )
        self.assertEqual(rc, 3)
        self.assertIn("RESULT_LINE", out)

    def test_buffered_stderr_does_not_block_exit(self):
        rc, _ = _child(
            _PRELUDE + "sys.stderr.write('E' * 4096)\nProcessExit.hard_exit(4)"
        )
        self.assertEqual(rc, 4)


class TestHardExitSkipsTeardown(unittest.TestCase):
    """Interpreter teardown is skipped, which is the reason it exists.

    A DCC's static destructors run at ``DLL_PROCESS_DETACH``; the portable,
    testable shadow of that is ``atexit``, which must NOT run.
    """

    def test_atexit_hooks_do_not_run(self):
        rc, out = _child(
            _PRELUDE
            + "import atexit\n"
            + "atexit.register(lambda: print('ATEXIT_RAN'))\n"
            + "print('BEFORE', end='')\n"
            + "ProcessExit.hard_exit(5)"
        )
        self.assertEqual(rc, 5)
        self.assertIn("BEFORE", out)
        self.assertNotIn("ATEXIT_RAN", out)

    def test_never_returns(self):
        # A caller that keeps running after hard_exit would silently do the
        # work twice; the sentinel must be unreachable.
        rc, out = _child(
            _PRELUDE + "ProcessExit.hard_exit(6)\nprint('UNREACHABLE', end='')"
        )
        self.assertEqual(rc, 6)
        self.assertNotIn("UNREACHABLE", out)


class TestHardExitIsTotal(unittest.TestCase):
    """It exits even when its own fast path is broken.

    The Windows ``TerminateProcess`` route is an OPTIMISATION over ``os._exit``,
    never a precondition for exiting.  A caller told "never returns" that
    instead receives an exception from its last statement reports a finished,
    green run as a crash -- so anything the fast path can raise has to degrade
    to the portable exit.

    Both cases PIN the Windows branch instead of discovering it: ``hard_exit``
    only consults ``_terminate_self`` under ``os.name == "nt"``, so on a POSIX
    CI runner these would exit 9 and 8 without ever entering the code they
    exist to cover -- green where they prove the least.  ``_WIN`` swaps the
    module's ``os`` for a stand-in reporting ``"nt"``, keeping the real
    ``_exit`` so the child still dies with the right status.
    """

    # Forces the branch on every platform (repo rule: a test that forks on a
    # machine capability pins the branch, it does not discover it).
    _WIN = (
        "import types, os as _real_os\n"
        "import pythontk.core_utils.process_exit as _m\n"
        "_m.os = types.SimpleNamespace(name='nt', _exit=_real_os._exit)\n"
    )

    def test_a_raising_terminate_still_exits(self):
        rc, out = _child(
            _PRELUDE
            + self._WIN
            + "ProcessExit._terminate_self = staticmethod(\n"
            + "    lambda code: (_ for _ in ()).throw(OSError('boom')))\n"
            + "print('BEFORE', end='')\n"
            + "ProcessExit.hard_exit(9)"
        )
        self.assertEqual(rc, 9)
        self.assertIn("BEFORE", out)

    def test_a_refused_terminate_still_exits(self):
        # The real fallback branch: the kill is declined, _terminate_self
        # returns, and hard_exit must still reach os._exit.
        rc, _ = _child(
            _PRELUDE
            + self._WIN
            + "ProcessExit._terminate_self = staticmethod(lambda code: None)\n"
            + "ProcessExit.hard_exit(8)"
        )
        self.assertEqual(rc, 8)

    def test_the_windows_branch_is_actually_entered(self):
        """Guard for the two above: prove ``_WIN`` reaches ``_terminate_self``.

        Without this, a change to how ``hard_exit`` selects the platform would
        make both of them pass for the wrong reason and say nothing.
        """
        # flush=True is load-bearing: hard_exit flushes BEFORE calling
        # _terminate_self, and the os._exit that follows discards whatever the
        # stand-in buffered after that point.
        rc, out = _child(
            _PRELUDE
            + self._WIN
            + "ProcessExit._terminate_self = staticmethod(\n"
            + "    lambda code: print('TERMINATE_CALLED', end='', flush=True))\n"
            + "ProcessExit.hard_exit(4)"
        )
        self.assertEqual(rc, 4)
        self.assertIn("TERMINATE_CALLED", out)

    def test_a_broken_stream_does_not_block_the_exit(self):
        rc, _ = _child(
            _PRELUDE
            + "class Boom:\n"
            + "    def flush(self): raise ValueError('closed')\n"
            + "    def write(self, s): raise ValueError('closed')\n"
            + "sys.stdout = Boom()\n"
            + "sys.stderr = Boom()\n"
            + "ProcessExit.hard_exit(2)"
        )
        self.assertEqual(rc, 2)


class TestHardExitSourceIsConsoleSafe(unittest.TestCase):
    """The module carries no glyph a Windows console cannot encode.

    Its diagnostics go to ``stderr``, which a Windows console encodes in the
    ACTIVE CODE PAGE -- and a non-cp1252 one raises ``UnicodeEncodeError`` on a
    stray em-dash.  On the failed-kill branch that exception would propagate
    and skip the very ``os._exit`` the branch exists to reach.  ``mayatk``'s
    suite driver carries a whole TeeStream to swallow this class of failure;
    here the cheaper answer is to not emit one.
    """

    def test_module_source_is_ascii(self):
        import pythontk.core_utils.process_exit as mod

        with open(mod.__file__, encoding="utf-8") as f:
            src = f.read()
        offenders = sorted({c for c in src if ord(c) > 127})
        self.assertEqual(offenders, [], f"non-ASCII in process_exit.py: {offenders!r}")


class TestHardExitSurface(unittest.TestCase):
    """The class is reachable the way callers are told to reach it."""

    def test_exported_on_the_package_root(self):
        import pythontk as ptk

        self.assertIs(ptk.ProcessExit, ProcessExit)

    def test_is_a_static_call(self):
        # Callers use it as `ptk.ProcessExit.hard_exit(code)` with no instance;
        # binding it to an instance would be a different contract.
        self.assertTrue(callable(ProcessExit.hard_exit))


if __name__ == "__main__":
    unittest.main()
