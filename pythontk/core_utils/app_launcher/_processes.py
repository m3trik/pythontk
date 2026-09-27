# !/usr/bin/python
# coding=utf-8
"""Running processes: find by name, close, walk a tree, bind a lifetime.

The OS-process half of :class:`AppLauncher` (``get_running_processes``,
``close_process``, ``process_tree``) and the lifetime binding
:meth:`AppLauncher.spawn` puts every helper child under: a kill-on-close Job
Object on Windows, a ``sh`` watcher in its own session on POSIX. Linux reads
``/proc`` directly (no ``psutil``).

One job of :class:`AppLauncher`, composed in ``_app_launcher.py``. A cross-call
goes through the facade (imported on use: that module imports this one for its
base), so a caller's ``patch.object(AppLauncher, ...)`` reaches it.
"""

import os
import platform
import subprocess
import logging
import threading

logger = logging.getLogger(__name__)


class _ProcessesMixin:
    """Process table and child lifetime; a private part of :class:`AppLauncher`."""

    #: The kill-on-close Job Object every :meth:`AppLauncher.spawn` child joins
    #: on Windows. Created on first use and deliberately never closed: its
    #: handle closes when THIS process ends -- a crash or a kill included -- and
    #: closing the last handle to such a job terminates everything in it. That
    #: is the whole mechanism; there is no watchdog or reaper to keep running.
    _lifetime_job = None
    _lifetime_lock = threading.Lock()

    @staticmethod
    def _proc_entries(with_names=True):
        """``(pid, ppid, names)`` for every process in ``/proc`` (Linux).

        *names* holds the executable's basename (``/proc/<pid>/exe``, readable
        for this user's processes), argv[0]'s basename and the kernel's
        ``comm`` -- a script launcher shows as its interpreter in ``exe`` but
        by its own name in argv[0]. Empty without *with_names*.
        """
        try:
            entries = os.listdir("/proc")
        except OSError:
            return
        for entry in entries:
            if not entry.isdigit():
                continue
            base = f"/proc/{entry}"
            try:
                with open(f"{base}/stat", "rb") as f:
                    stat = f.read()
                # "pid (comm) state ppid ..." -- comm may hold spaces or ')'.
                comm = stat[stat.index(b"(") + 1 : stat.rindex(b")")]
                ppid = int(stat[stat.rindex(b")") + 1 :].split()[1])
            except (OSError, ValueError, IndexError):
                continue  # gone since listdir, or not a process
            names = set()
            if with_names:
                names.add(comm.decode(errors="replace"))
                try:
                    names.add(os.path.basename(os.readlink(f"{base}/exe")))
                except OSError:
                    pass
                try:
                    with open(f"{base}/cmdline", "rb") as f:
                        argv0 = f.read().split(b"\0", 1)[0]
                    if argv0:
                        names.add(os.path.basename(argv0.decode(errors="replace")))
                except OSError:
                    pass
            yield int(entry), ppid, names

    #: The POSIX lifetime watcher: while this process ($1) lives, wait; once
    #: the child ($2) is gone, leave; if this process goes first (a crash
    #: included), TERM then KILL the child's target ($3: its process group as
    #: "-PGID" when it leads one, else its PID). POSIX sh -- dash needs the
    #: "--" before a negative group id.
    _LIFETIME_WATCHER = (
        'while kill -0 "$1" 2>/dev/null; do '
        'kill -0 "$2" 2>/dev/null || exit 0; sleep 1; done; '
        'kill -s TERM -- "$3" 2>/dev/null; sleep 3; '
        'kill -s KILL -- "$3" 2>/dev/null; exit 0'
    )

    @staticmethod
    def _watch_lifetime(pid: int) -> bool:
        """Start the POSIX watcher that kills *pid* once this process is gone.

        The watcher runs in its own session, so whatever takes this process
        down (a crash, a kill of its process group) leaves it running to clean
        up; it is ``sh``, not python, since inside a DCC ``sys.executable`` is
        the host binary. A child that leads its own process group (``spawn``
        starts it so on POSIX) is killed with every process it started.
        """
        try:
            target = f"-{pid}" if os.getpgid(pid) == pid else str(pid)
            subprocess.Popen(
                [
                    "/bin/sh",
                    "-c",
                    _ProcessesMixin._LIFETIME_WATCHER,
                    "ptk-lifetime",
                    str(os.getpid()),
                    str(pid),
                    target,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
            return True
        except (OSError, ValueError) as e:
            logger.debug(f"lifetime watcher for {pid} not started: {e}")
            return False

    @staticmethod
    def _descendants(pid):
        """Every live descendant PID of *pid* (Linux ``/proc``; empty elsewhere)."""
        children = {}
        for child, ppid, _ in _ProcessesMixin._proc_entries(with_names=False):
            children.setdefault(ppid, []).append(child)
        found, stack = set(), [pid]
        while stack:
            for child in children.get(stack.pop(), ()):
                if child not in found:
                    found.add(child)
                    stack.append(child)
        return found

    @staticmethod
    def _bind_lifetime(pid: int) -> bool:
        """Make process *pid* die with this one; False where that cannot be arranged.

        Windows: a kill-on-close Job Object (see :attr:`_lifetime_job`). POSIX:
        a ``/bin/sh`` watcher in its own session (:meth:`_watch_lifetime`).
        Not ``PR_SET_PDEATHSIG``: it needs a ``preexec_fn`` (unsafe in a
        threaded host) and fires when the spawning THREAD ends, not the process.
        """
        if os.name != "nt":
            return _ProcessesMixin._watch_lifetime(pid)
        import ctypes
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class _BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimits),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        extended_limit_information = 9  # JobObjectExtendedLimitInformation
        kill_on_job_close = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        set_quota_and_terminate = 0x0100 | 0x0001  # PROCESS_SET_QUOTA | _TERMINATE

        # A private handle on kernel32, so the argtypes declared here cannot
        # leak into another caller's use of the shared `ctypes.windll.kernel32`.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.POINTER(_ExtendedLimits),
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

        with _ProcessesMixin._lifetime_lock:
            job = _ProcessesMixin._lifetime_job
            if job is None:
                job = kernel32.CreateJobObjectW(None, None)
                if not job:
                    logger.debug(f"CreateJobObject failed: {ctypes.get_last_error()}")
                    return False
                limits = _ExtendedLimits()
                limits.BasicLimitInformation.LimitFlags = kill_on_job_close
                if not kernel32.SetInformationJobObject(
                    job,
                    extended_limit_information,
                    ctypes.byref(limits),
                    ctypes.sizeof(limits),
                ):
                    logger.debug(
                        f"SetInformationJobObject failed: {ctypes.get_last_error()}"
                    )
                    kernel32.CloseHandle(job)
                    return False
                _ProcessesMixin._lifetime_job = job
        process = kernel32.OpenProcess(set_quota_and_terminate, False, pid)
        if not process:
            return False  # already gone, or not ours to manage
        try:
            if kernel32.AssignProcessToJobObject(job, process):
                return True
            logger.debug(
                f"AssignProcessToJobObject({pid}) failed: {ctypes.get_last_error()}"
            )
            return False
        finally:
            kernel32.CloseHandle(process)

    @staticmethod
    def get_running_processes(process_name):
        """
        Returns a list of PIDs of running processes matching the given name.
        Uses tasklist (Windows), ``/proc`` (Linux) or ``pgrep -x`` (other Unix)
        to avoid external dependencies like psutil.

        The match is on the executable's NAME, never a substring of a command
        line: ``maya`` must not match ``tail maya.log``. A bare name means that
        program in any extension -- ``maya.exe`` on Windows, ``maya.bin`` (Maya's
        Linux GUI binary) on Linux -- and a ``.exe`` suffix is ignored off Windows.

        :param process_name: The name of the process (e.g. 'notepad.exe', 'maya').
        :return: List of integer PIDs.
        """
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        pids = []
        system = platform.system().lower()

        try:
            if system == "windows":
                # tasklist matches the whole image name: "maya" is "maya.exe".
                if not process_name.lower().endswith(".exe"):
                    process_name += ".exe"
                # 'tasklist /FO CSV /NH' returns "Image Name","PID","Session Name","Session#","Mem Usage"
                # Filter by name using /FI to rely on system filter.
                # List args (no shell=True): immune to quoting/injection issues
                # if process_name ever contains shell metacharacters.
                cmd = [
                    "tasklist",
                    "/FO",
                    "CSV",
                    "/NH",
                    "/FI",
                    f"IMAGENAME eq {process_name}",
                ]
                output = subprocess.check_output(cmd).decode(errors="ignore")

                import csv
                import io

                reader = csv.reader(io.StringIO(output))
                for row in reader:
                    if len(row) >= 2:
                        try:
                            # Verify the name matches because tasklist might return empty or "INFO: No tasks..."
                            # row[0] is image name, row[1] is PID
                            if row[0].lower() == process_name.lower():
                                pids.append(int(row[1]))
                        except ValueError:
                            pass

            elif system == "linux":
                wanted = process_name
                if wanted.lower().endswith(".exe"):
                    wanted = wanted[:-4]
                for pid, _ppid, names in AppLauncher._proc_entries():
                    if any(
                        n == wanted or os.path.splitext(n)[0] == wanted for n in names
                    ):
                        pids.append(pid)

            else:
                # Other Unix (macOS): exact process-name match.
                cmd = ["pgrep", "-x", process_name]
                output = subprocess.check_output(cmd).decode(errors="ignore")
                for line in output.splitlines():
                    if line.strip().isdigit():
                        pids.append(int(line.strip()))

        except subprocess.CalledProcessError:
            # pgrep returns non-zero if no process found
            pass
        except Exception as e:
            logger.debug(f"Error finding processes: {e}")

        return pids

    @staticmethod
    def close_process(pid, force=False):
        """
        Terminates the process with the given PID.

        On POSIX a process that leads its own process group -- as
        :meth:`launch` (``detached=True``) and :meth:`spawn` start them -- is
        closed with its whole group: a Linux launcher script's child (Maya's
        ``maya.bin`` behind ``bin/maya``) is the application the caller means.

        :param pid: Process ID to terminate.
        :param force: If True, force kill (SIGKILL/TerminateProcess).
        :return: True if successful, False otherwise.
        """
        import signal

        try:
            if platform.system().lower() == "windows":
                # Use taskkill
                args = ["taskkill", "/PID", str(pid)]
                if force:
                    args.append("/F")
                # Hide output
                subprocess.check_call(
                    args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return True
            else:
                sig = signal.SIGKILL if force else signal.SIGTERM
                if os.getpgid(pid) == pid:
                    os.killpg(pid, sig)
                else:
                    os.kill(pid, sig)
                return True
        except Exception as e:
            logger.debug(f"Failed to close process {pid}: {e}")
            return False

    @staticmethod
    def process_tree(pid):
        """*pid* and every live descendant: ``{pid, ...}``.

        Linux reads ``/proc``; elsewhere it is ``{pid}``. A Linux launcher is
        often a script that forks the real binary (Maya's ``bin/maya`` runs
        ``maya.bin``), so what the launched PID "owns" -- a window, a listening
        port -- may belong to a child of it.
        """
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        return {pid} | AppLauncher._descendants(pid)
