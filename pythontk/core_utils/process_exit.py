# !/usr/bin/python
# coding=utf-8
"""Exit a DCC-hosted interpreter without running ``DLL_PROCESS_DETACH``.

A DCC standalone interpreter (``mayapy`` et al.) does not shut down cleanly.
Maya's own static destructors fault during detach -- measured 2026-09-10 on
Maya 2025.3, where the teardown runs
``TDNregister::~TDNregister -> TdependGraph::deregisterNodeType ->
TnClothShape::sCleanupClass -> TattributeList::reset -> TsharedObject::unref ->
free()`` and takes an access violation in the heap.  There is no repo code on
that stack and nothing downstream can fix it; the only lever is to not be there.

``os._exit`` is NOT that lever on Windows.  It routes to the CRT's ``_exit``,
which calls ``ExitProcess``, and ``ExitProcess`` still runs the detach callback
of every loaded DLL -- the exact frames above.  Measured on Maya 2025 with a
scene loaded, exiting with status 7 either way:

===================  ===========  ==================  ==================
exit path            exit status  new crash minidump  recovery ``.ma``
===================  ===========  ==================  ==================
``os._exit(7)``      7            yes                 yes
``TerminateProcess`` 7            no                  no
===================  ===========  ==================  ==================

The status survived both ways in that configuration, so the damage is not
(here) a wrong answer: it is that every exit files a crash report.  A day of
ordinary test running left 28 minidumps and 7 ``untitled[Recovered-...].ma``
files in the system temp dir, which is indistinguishable from a real crash --
one backlog entry was opened on exactly that misreading.  Under Qt the status
does not always survive: ``uitk``'s runner measured green runs reported as
failed, the status replaced by ``0xC0000005``.

This lives in the shipped package, at the bottom of the stack, for the same
reason :class:`pythontk.TestSandbox` does -- the effect is process-wide and
every downstream entry point that exits from inside a DCC host needs the
identical treatment (``uitk``, ``mayatk``, ``blendertk``, ``tentacle``), and a
second copy is a copy that drifts.  It is the write side of the note in
``pythontk/core_utils/script_run.py``, which judges a DCC run by its artifact
because "the exit code is advisory only": a script that exits through here
makes its status trustworthy again.

Usage is one call, last, after everything that must survive is on disk::

    import pythontk as ptk
    ptk.ProcessExit.hard_exit(0 if ok else 1)

It never returns.  ``atexit`` hooks, ``__del__`` finalizers and C++ static
destructors do NOT run, so release anything that matters BEFORE calling it.
On POSIX there are no detach callbacks to skip and it is simply ``os._exit``.
"""

import os
import sys
from typing import NoReturn

__all__ = ["ProcessExit"]


class _ProcessExitInternal:
    """Platform mechanics behind :class:`ProcessExit`."""

    @staticmethod
    def _flush_streams() -> None:
        """Push buffered output out before the process stops existing.

        ``TerminateProcess`` drops whatever is still in the interpreter's
        buffers.  A child whose stdout is a pipe -- every test chunk a runner
        spawns -- is block-buffered, so without this the last block of the
        report is lost exactly when it is being read by a parent.
        """
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001 - a closed/None stream must not block the exit
                pass

    @staticmethod
    def _terminate_self(code: int) -> None:
        """Windows: stop the process without running detach callbacks.

        Returns only if ``TerminateProcess`` reported failure, so the caller
        can fall back.
        """
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # The HANDLE types are NOT decoration.  ctypes defaults a restype to
        # c_int, which truncates GetCurrentProcess()'s 64-bit pseudo-handle
        # (-1) to 0x00000000FFFFFFFF -- an invalid handle, so the kill failed
        # DETERMINISTICALLY and every run fell through to the detach path this
        # function exists to skip.  That bug hid for two fix cycles because the
        # fallback was silent.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        kernel32.TerminateProcess.restype = ctypes.c_int
        kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        kernel32.WaitForSingleObject.restype = ctypes.c_uint

        handle = kernel32.GetCurrentProcess()
        if not kernel32.TerminateProcess(handle, ctypes.c_uint(code)):
            err = ctypes.get_last_error()
            # ASCII only, and guarded.  A Windows console encodes stderr in the
            # active code page, and a non-cp1252 one raises UnicodeEncodeError
            # on a single stray glyph -- which would propagate out of here and
            # skip the os._exit fallback this branch exists to reach, turning a
            # recoverable failed kill into an exception in the exit path.
            try:
                print(
                    "ProcessExit.hard_exit: TerminateProcess failed "
                    f"(WinError {err}); falling back to os._exit - "
                    "DLL_PROCESS_DETACH will run and may file a crash report.",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:  # noqa: BLE001 - diagnostics never outrank the exit
                pass
            return

        # TerminateProcess is ASYNCHRONOUS: it can return to this thread while
        # the kill is still in flight.  Falling through to os._exit here would
        # re-enter the detach callbacks, so park instead -- with ONE wait that
        # never returns, not a sleep loop.  A loop re-enters Python bytecode
        # and ctypes marshalling once per iteration inside a dying process, and
        # that execution surface is itself where an access violation replaced a
        # green run's status with 0xC0000005 (observed live, uitk 2026-07-25).
        infinite = 0xFFFFFFFF
        while True:
            # Looped only against a spurious return (e.g. WAIT_FAILED); falling
            # through would defeat the whole function.
            kernel32.WaitForSingleObject(handle, infinite)


class ProcessExit(_ProcessExitInternal):
    """Process-wide exit that skips interpreter and DLL teardown."""

    @staticmethod
    def hard_exit(code: int = 0) -> NoReturn:
        """Stop this process immediately with status *code*.  Never returns.

        Flushes ``stdout`` / ``stderr``, then terminates without running
        ``atexit`` hooks, object finalizers, or -- on Windows -- any DLL's
        ``DLL_PROCESS_DETACH`` handler.  Release anything that must outlive the
        process (files, locks, sandboxes) BEFORE calling.

        Parameters:
            code: Exit status handed to the parent. Conventionally 0 for
                success and 1 for failure; Windows accepts any 32-bit value,
                POSIX keeps only the low 8 bits.

        Returns:
            Never returns.
        """
        code = int(code)
        ProcessExit._flush_streams()
        if os.name == "nt":
            # Total by contract: the Windows path is an OPTIMISATION over
            # os._exit, never a precondition for exiting.  It returns on a
            # refused kill, and anything it can raise (a WinDLL that will not
            # load, a ctypes surprise on an unusual build) must degrade to the
            # portable exit rather than propagate -- a caller that is told this
            # never returns would otherwise get an exception from its last
            # statement, after the work it was reporting on had already
            # finished.
            try:
                ProcessExit._terminate_self(code)
            except Exception:  # noqa: BLE001 - see above; never block the exit
                pass
        os._exit(code)
