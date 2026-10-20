# !/usr/bin/python
# coding=utf-8
"""Hidden runs: an app with no headless mode, run where no one sees it, and never left behind.

For a GUI app a script drives (RizomUV ``-cfi``, Marmoset Toolbag ``-run``):
:meth:`AppLauncher.run` with ``hidden=True``. Windows only. The child is created
on a private desktop of this window station -- the user's own monitors, so the
app opens, and saves, its windows exactly where it always does (measured: a Qt
window's saved geometry byte-identical after a hidden run; RizomUV's frame and
Toolbag's window prefs unchanged, its bakes pixel-identical), while nothing
reaches the user's desktop. It is created inside a kill-on-close job of its own,
from its first instruction (``PROC_THREAD_ATTRIBUTE_JOB_LIST``; suspended, then
assigned, before Windows 10 1607), so nothing it starts escapes: the app's exit,
a timeout, an error, an interrupt or a cancel ends the whole tree, and so does
this process ending, however it ends (the job's one handle is this process's).
A dialog the app opens there can't be answered, so the run is cancellable while
it waits: the ambient :class:`~pythontk.CancelScope` is polled at every step.

Not elsewhere (``hidden`` is a no-op there, as ``hide_window`` is;
:meth:`AppLauncher.can_hide` says which): a virtual X
display is not the user's -- measured, Qt refuses to restore a geometry saved
on a screen more than 25% wider or narrower, the app falls back to its defaults
and saves them, and its next normal launch opens small in a corner. Not for
:meth:`AppLauncher.launch` either: an app that outlives this process on a
desktop no one can see could never be closed.

One job of :class:`AppLauncher`, composed in ``_app_launcher.py``.
"""

import logging
import os
import subprocess
import threading

logger = logging.getLogger(__name__)

#: The private desktop every hidden child of this process starts on.
_DESKTOP_NAME = "pythontk-hidden"

_lock = threading.Lock()
#: The desktop's handle, held for this process's life: a desktop lasts while a
#: handle to it is open, and every hidden child must find it there.
_desktop_handle = None

if os.name == "nt":
    import ctypes
    from ctypes import wintypes as _wt

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.OpenDesktopW.restype = _wt.HANDLE
    _user32.OpenDesktopW.argtypes = (_wt.LPCWSTR, _wt.DWORD, _wt.BOOL, _wt.DWORD)
    _user32.CreateDesktopW.restype = _wt.HANDLE
    _user32.CreateDesktopW.argtypes = (
        _wt.LPCWSTR,
        _wt.LPCWSTR,
        ctypes.c_void_p,
        _wt.DWORD,
        _wt.DWORD,
        ctypes.c_void_p,
    )
    # Private, as ``_processes`` keeps its own: these argtypes reach no one else.
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.InitializeProcThreadAttributeList.argtypes = (
        ctypes.c_void_p,
        _wt.DWORD,
        _wt.DWORD,
        ctypes.POINTER(ctypes.c_size_t),
    )
    _kernel32.UpdateProcThreadAttribute.argtypes = (
        ctypes.c_void_p,
        _wt.DWORD,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    _kernel32.DeleteProcThreadAttributeList.argtypes = (ctypes.c_void_p,)
    _kernel32.CreateProcessW.argtypes = (
        _wt.LPCWSTR,
        _wt.LPWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        _wt.BOOL,
        _wt.DWORD,
        ctypes.c_void_p,
        _wt.LPCWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    _kernel32.AssignProcessToJobObject.argtypes = (_wt.HANDLE, _wt.HANDLE)
    _kernel32.TerminateProcess.argtypes = (_wt.HANDLE, _wt.UINT)
    _kernel32.ResumeThread.argtypes = (_wt.HANDLE,)
    _kernel32.CloseHandle.argtypes = (_wt.HANDLE,)

    _GENERIC_ALL = 0x10000000
    _JOB_LIST = 0x2000D  # PROC_THREAD_ATTRIBUTE_JOB_LIST
    _HANDLE_LIST = 0x20002  # PROC_THREAD_ATTRIBUTE_HANDLE_LIST
    _STARTF_USESTDHANDLES = 0x100
    _CREATE_SUSPENDED = 0x4
    _CREATE_UNICODE_ENVIRONMENT = 0x400
    _EXTENDED_STARTUPINFO_PRESENT = 0x80000

    class _StartupInfo(ctypes.Structure):
        _fields_ = [
            ("cb", _wt.DWORD),
            ("lpReserved", _wt.LPWSTR),
            ("lpDesktop", _wt.LPWSTR),
            ("lpTitle", _wt.LPWSTR),
            ("dwX", _wt.DWORD),
            ("dwY", _wt.DWORD),
            ("dwXSize", _wt.DWORD),
            ("dwYSize", _wt.DWORD),
            ("dwXCountChars", _wt.DWORD),
            ("dwYCountChars", _wt.DWORD),
            ("dwFillAttribute", _wt.DWORD),
            ("dwFlags", _wt.DWORD),
            ("wShowWindow", _wt.WORD),
            ("cbReserved2", _wt.WORD),
            ("lpReserved2", ctypes.c_void_p),
            ("hStdInput", _wt.HANDLE),
            ("hStdOutput", _wt.HANDLE),
            ("hStdError", _wt.HANDLE),
        ]

    class _StartupInfoEx(ctypes.Structure):
        _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", ctypes.c_void_p)]

    class _ProcessInformation(ctypes.Structure):
        _fields_ = [
            ("hProcess", _wt.HANDLE),
            ("hThread", _wt.HANDLE),
            ("dwProcessId", _wt.DWORD),
            ("dwThreadId", _wt.DWORD),
        ]

    def _desktop() -> str:
        """The private desktop's name, opened (or created) on first use."""
        global _desktop_handle
        with _lock:
            if _desktop_handle is None:
                handle = _user32.OpenDesktopW(
                    _DESKTOP_NAME, 0, False, _GENERIC_ALL
                ) or _user32.CreateDesktopW(
                    _DESKTOP_NAME, None, None, 0, _GENERIC_ALL, None
                )
                if not handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                _desktop_handle = handle
        return _DESKTOP_NAME

    def _attributes(job, inherit):
        """``(attribute list, what it points at)`` that create a process INSIDE
        *job* and let it inherit only *inherit* -- its own std handles, never
        another pipe of this process, whose reader would then wait on an app no
        one can see for its end of file. ``(None, None)`` where Windows has no
        job-list attribute (before 10 1607)."""
        count = 2 if inherit else 1
        size = ctypes.c_size_t()
        _kernel32.InitializeProcThreadAttributeList(None, count, 0, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not _kernel32.InitializeProcThreadAttributeList(
            buffer, count, 0, ctypes.byref(size)
        ):
            return None, None
        holder = _wt.HANDLE(job)
        handles = (_wt.HANDLE * len(inherit))(*inherit) if inherit else None
        ok = _kernel32.UpdateProcThreadAttribute(
            buffer, 0, _JOB_LIST, ctypes.byref(holder), ctypes.sizeof(holder), None, None
        )
        if ok and handles is not None:
            ok = _kernel32.UpdateProcThreadAttribute(
                buffer, 0, _HANDLE_LIST, handles, ctypes.sizeof(handles), None, None
            )
        if not ok:
            _kernel32.DeleteProcThreadAttributeList(buffer)
            return None, None
        return buffer, (holder, handles)

    class _HiddenPopen(subprocess.Popen):
        """``subprocess.Popen`` whose child starts on the private desktop, inside a
        job of its own. Pipes, ``communicate`` and the exit code are Popen's own.

        The job -- the child and everything it started -- ends once the child is
        seen to exit (``poll``, ``wait``), on ``kill`` / ``terminate``, on leaving
        a ``with`` block, and with this process.
        """

        _job = None

        def _execute_child(
            self,
            args,
            executable,
            preexec_fn,
            close_fds,
            pass_fds,
            cwd,
            env,
            startupinfo,
            creationflags,
            shell,
            p2cread,
            p2cwrite,
            c2pread,
            c2pwrite,
            errread,
            errwrite,
            *_posix_only,  # their count differs between Python versions
        ):
            from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

            if shell:
                raise ValueError("A hidden run starts the app itself, not a shell.")
            if not isinstance(args, str):
                args = subprocess.list2cmdline(args)
            info = _StartupInfoEx()
            si = info.StartupInfo
            if startupinfo is not None:
                si.dwFlags, si.wShowWindow = startupinfo.dwFlags, startupinfo.wShowWindow
            use_std = -1 not in (p2cread, c2pwrite, errwrite)
            if use_std:
                si.dwFlags |= _STARTF_USESTDHANDLES
                si.hStdInput, si.hStdOutput, si.hStdError = (
                    int(p2cread),
                    int(c2pwrite),
                    int(errwrite),
                )
            block = None
            if env is not None:
                creationflags |= _CREATE_UNICODE_ENVIRONMENT
                # Sorted, case-insensitively, as CreateProcess documents the block.
                pairs = sorted(env.items(), key=lambda item: item[0].upper())
                text = "".join(f"{k}={v}\0" for k, v in pairs) + "\0"
                # Sized in UTF-16 units (an emoji is two), plus the closing NUL
                # an empty block still needs.
                units = len(text.encode("utf-16-le", "surrogatepass")) // 2
                block = ctypes.create_unicode_buffer(text, units + 1)
            pi = _ProcessInformation()
            job = attributes = None
            try:  # the child's ends of the pipes close whatever happens
                si.lpDesktop = _desktop()
                job = AppLauncher._new_job()
                inherit = (
                    sorted({int(h) for h in (p2cread, c2pwrite, errwrite)})
                    if use_std
                    else []
                )
                attributes, _kept = _attributes(job, inherit)
                if attributes is not None:
                    creationflags |= _EXTENDED_STARTUPINFO_PRESENT
                    info.lpAttributeList = ctypes.cast(attributes, ctypes.c_void_p)
                    si.cb = ctypes.sizeof(info)
                else:
                    creationflags |= _CREATE_SUSPENDED
                    si.cb = ctypes.sizeof(si)
                ok = _kernel32.CreateProcessW(
                    executable,
                    ctypes.create_unicode_buffer(args),
                    None,
                    None,
                    bool(use_std),
                    creationflags,
                    block,
                    None if cwd is None else os.fspath(cwd),
                    ctypes.byref(info),
                    ctypes.byref(pi),
                )
                if not ok:
                    raise ctypes.WinError(ctypes.get_last_error())
                if attributes is None:  # created suspended: join the job, then run
                    if not _kernel32.AssignProcessToJobObject(job, pi.hProcess):
                        error = ctypes.get_last_error()
                        _kernel32.TerminateProcess(pi.hProcess, 1)  # never run one we cannot end
                        raise ctypes.WinError(error)
                    _kernel32.ResumeThread(pi.hThread)
            except BaseException:
                # The job's last handle among them: whatever is in it ends too.
                for handle in (pi.hThread, pi.hProcess, job):
                    if handle:
                        _kernel32.CloseHandle(handle)
                raise
            finally:
                if attributes is not None:
                    _kernel32.DeleteProcThreadAttributeList(attributes)
                self._close_pipe_fds(p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite)
            _kernel32.CloseHandle(pi.hThread)
            self._job = job
            self._child_created = True
            self._handle = subprocess.Handle(pi.hProcess)
            self.pid = pi.dwProcessId

        def _end_tree(self) -> None:
            from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

            with _lock:
                job, self._job = self._job, None
            if job:
                AppLauncher._end_job(job)

        def poll(self):
            code = super().poll()
            if code is not None:
                self._end_tree()
            return code

        def wait(self, timeout=None):
            code = super().wait(timeout)
            self._end_tree()
            return code

        def kill(self):
            """End the child and everything it started."""
            self._end_tree()
            super().kill()

        terminate = kill

        def __exit__(self, *exc_info):
            try:
                return super().__exit__(*exc_info)
            finally:
                self._end_tree()


class _HiddenMixin:
    """Hidden runs; a private part of :class:`AppLauncher`."""

    @staticmethod
    def can_hide() -> bool:
        """Whether ``run(hidden=True)`` hides the app on this platform.

        Windows only: a virtual display elsewhere would move the app's window on
        its next normal launch, so there ``hidden`` is a no-op. A caller that
        tells its user what a run did (a timed-out hidden run may have hidden a
        dialog) asks this rather than spelling the platform rule again.

        Returns:
            True on Windows.
        """
        return os.name == "nt"

    @staticmethod
    def _popen(hidden: bool):
        """The ``Popen`` class a run starts its child with."""
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        if not hidden:
            return subprocess.Popen
        if not AppLauncher.can_hide():
            logger.debug("No hidden run on this platform: the app shows as usual.")
            return subprocess.Popen
        return _HiddenPopen

    #: Seconds between looks at whether a hidden run's app has exited.
    _HIDDEN_EXIT_POLL = 0.25

    @staticmethod
    def _run_hidden(cmd, timeout=None, capture_output=False, **kwargs):
        """``subprocess.run`` over the hidden ``Popen`` (the stdlib's takes no
        class): the same result and ``TimeoutExpired``, and the child's tree
        ended on every way out.

        It waits in short steps, not for end of file: a helper the app started
        inherits its output pipe and can hold it open after the app exits.
        Seeing the exit ends the job (``poll``), the helper with it, and the
        pipe closes -- where a plain run waits on it until the timeout.

        Each step is also a checkpoint of the ambient
        :class:`~pythontk.CancelScope`: an app waiting on a dialog no one can
        see would otherwise hold its caller for the whole timeout, so a cancel
        (an Esc hold under a ``@Cancelable`` slot) ends the tree and raises
        :class:`~pythontk.OperationCancelled`. An app already gone is not
        cancelled: its run completed.
        """
        import time

        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher
        from pythontk.core_utils.cancel_scope import CancelScope

        if capture_output:
            kwargs["stdout"] = kwargs["stderr"] = subprocess.PIPE
        deadline = None if timeout is None else time.monotonic() + timeout
        with _HiddenPopen(cmd, **kwargs) as process:
            try:
                while True:
                    step = AppLauncher._HIDDEN_EXIT_POLL
                    if deadline is not None:
                        step = max(0.0, min(step, deadline - time.monotonic()))
                    try:
                        stdout, stderr = process.communicate(timeout=step)
                        break
                    except subprocess.TimeoutExpired:
                        # The app gone: its job ends (poll), and its output is
                        # the next step's to collect.
                        if process.poll() is None and not CancelScope.proceed():
                            raise AppLauncher._cancelled(process.args) from None
                        if deadline is not None and time.monotonic() >= deadline:
                            raise subprocess.TimeoutExpired(
                                process.args, timeout
                            ) from None
            except subprocess.TimeoutExpired as error:
                process.kill()
                error.stdout, error.stderr = process.communicate()
                raise
            except BaseException:
                process.kill()
                raise
            return subprocess.CompletedProcess(
                process.args, process.poll(), stdout, stderr
            )
