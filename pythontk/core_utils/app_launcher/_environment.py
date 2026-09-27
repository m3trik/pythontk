# !/usr/bin/python
# coding=utf-8
"""The environment a child inherits, and the user's persisted PATH.

``process_environ`` reads the LIVE environment block (a host app that embeds
Python sets variables at the C level that the ``os.environ`` snapshot never
sees); ``handoff_env`` and ``desktop_env`` derive a child's env from it for a
hand-off to another app and for a desktop helper; ``python_args_via_env`` moves
a Python child's command line into its env (mayapy reads its own in the ANSI
code page), and ``ansi_safe_path`` spells a path so a program that reads paths
in that code page (Maya) can open it. ``append_to_path`` /
``is_path_persisted`` own the user-scope PATH entry (the registry on Windows,
one marked ``~/.profile`` line elsewhere).

One job of :class:`AppLauncher`, composed in ``_app_launcher.py``. A cross-call
goes through the facade (imported on use: that module imports this one for its
base), so a caller's ``patch.object(AppLauncher, ...)`` reaches it.
"""

import os
import platform
import logging

logger = logging.getLogger(__name__)


class _EnvironmentMixin:
    """Process environment and PATH; a private part of :class:`AppLauncher`."""

    #: Loader and interpreter overrides a host app sets for ITSELF (Maya's and
    #: Blender's own Qt, libpython and plug-in trees). A desktop program handed a
    #: path -- xdg-open and the file manager it starts -- loads them in place of
    #: its own libraries and fails or crashes.
    _HOST_PRIVATE_VARS = (
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONPATH",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QML2_IMPORT_PATH",
    )

    @staticmethod
    def _posix_environ():
        """libc's ``environ`` -- the block a child inherits -- or None.

        Sees a variable the host set with C-level ``setenv`` after Python
        started, which ``os.environ`` (a snapshot) cannot. ``os.environ``
        writes go through ``putenv``, so they are in it too. None where the
        symbol is not exported (macOS keeps it behind ``_NSGetEnviron``).
        """
        import ctypes
        import sys

        try:
            block = ctypes.POINTER(ctypes.c_char_p).in_dll(ctypes.CDLL(None), "environ")
        except (OSError, ValueError, AttributeError, TypeError):
            # TypeError: ``CDLL(None)`` where no libc loads (a non-POSIX host).
            return None
        env = {}
        encoding = sys.getfilesystemencoding()
        i = 0
        while block[i]:
            entry = block[i].decode(encoding, "surrogateescape")
            key, sep, value = entry.partition("=")
            if key and sep:
                env[key] = value
            i += 1
        return env or None

    @staticmethod
    def _profile_path_line(path):
        """The one ``~/.profile`` line :meth:`AppLauncher.append_to_path` owns
        for *path* -- marked, so it can be found again exactly."""
        import shlex

        return (
            f'export PATH="$PATH":{shlex.quote(path)}'
            "  # added by pythontk AppLauncher.append_to_path"
        )

    @staticmethod
    def process_environ():
        """The LIVE process environment -- what a child would actually inherit.

        ``os.environ`` is a snapshot taken at interpreter init (plus Python-side
        changes). A HOST app that embeds Python and sets a variable at the C
        level afterwards is invisible to the snapshot, yet every child process
        inherits the real value: Blender 5.x sets ``OCIO`` to its bundled
        v2.5 config during color-management init, so ``os.environ`` showed no
        ``OCIO`` while a child ``cmd /c echo %%OCIO%%`` printed Blender's config
        (measured live -- the mechanism behind Maya's color-management init
        failing on every bridge send). Windows: decode the live block via
        ``GetEnvironmentStringsW``. Linux: walk libc's ``environ``. Elsewhere,
        or when neither is readable, a copy of ``os.environ``.
        """
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        if platform.system().lower() != "windows":
            return AppLauncher._posix_environ() or dict(os.environ)
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetEnvironmentStringsW.restype = ctypes.c_void_p
        ptr = kernel32.GetEnvironmentStringsW()
        if not ptr:
            return dict(os.environ)
        try:
            env = {}
            offset = ptr
            while True:
                entry = ctypes.wstring_at(offset)
                if not entry:
                    break
                # Advance by the UTF-16 byte length + terminator, not len(entry):
                # a non-BMP character is ONE Python char but TWO UTF-16 code
                # units, and undercounting desyncs the rest of the block walk.
                offset += len(entry.encode("utf-16-le")) + 2
                key, sep, value = entry.partition("=")
                # Skip the hidden per-drive CWD entries ("=C:=C:\\..."), whose
                # key is empty -- children rebuild those themselves.
                if key and sep:
                    env[key] = value
            return env
        finally:
            kernel32.FreeEnvironmentStringsW(ctypes.c_void_p(ptr))

    #: The variable :meth:`python_args_via_env` hands a Python child its command
    #: line in: a JSON list (ASCII whatever it holds), the argv that would follow
    #: the interpreter -- ``[script, *args]`` or ``["-m", module, *args]``. A host
    #: whose startup hook takes code rather than a path (maya.exe's ``-command``)
    #: reads the script from item 0 of it.
    PYTHON_ARGV_VAR = "PYTHONTK_ARGV"

    #: What :meth:`python_args_via_env` runs with ``-c``: exactly ``python <argv>``,
    #: the argv read from :attr:`PYTHON_ARGV_VAR`. The variable is removed before
    #: the target runs, so nothing the target spawns inherits it. A script gets its
    #: own directory as ``sys.path[0]`` (what ``python script.py`` gives it) unless
    #: the interpreter was told to add none (``-I`` / ``-P``).
    _PYTHON_ARGV_SHIM = "\n".join(
        (
            "import json, os, runpy, sys",
            f"argv = json.loads(os.environ.pop({PYTHON_ARGV_VAR!r}))",
            "if argv[0] == '-m':",
            "    sys.argv[:] = argv[1:]",
            "    runpy.run_module(argv[1], run_name='__main__', alter_sys=True)",
            "else:",
            "    sys.argv[:] = argv",
            "    if sys.path[:1] == ['']:",
            "        sys.path[0] = os.path.dirname(os.path.abspath(argv[0]))",
            "    runpy.run_path(argv[0], run_name='__main__')",
        )
    )

    @staticmethod
    def python_args_via_env(argv, env=None):
        """The arguments and environment that make a Python interpreter run
        ``python <argv>`` with nothing of *argv* on its command line.

        mayapy.exe (and maya.exe) decode their command line in the ANSI code
        page: measured on Maya 2025, "José" arrived as "Jos\\udce9" and "Жук" as
        "???", while the environment and the working directory arrived intact.
        So ``mayapy <script>`` for a user whose %TEMP% holds such a letter ran a
        file that does not exist. Here the command line is only ``-c`` and a
        fixed ASCII shim; *argv* travels in :attr:`PYTHON_ARGV_VAR`. Any Python
        reads either route, so callers need not know which interpreter it is.

        Parameters:
            argv: What would follow the interpreter: ``[script, *args]`` or
                ``["-m", module, *args]``. Items are converted with ``str``.
            env: The child's environment to add to (not modified); ``None`` =
                the environment the child would inherit (:meth:`process_environ`).

        Returns:
            tuple: ``(args, env)`` -- *args* to follow the interpreter and any
            interpreter flags (``[python, *flags, *args]``), and a new env dict
            carrying *argv*.

        Raises:
            ValueError: When *argv* is empty (there is nothing to run).
        """
        import json
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        argv = [str(arg) for arg in argv]
        if not argv:
            raise ValueError("argv is empty: name a script or '-m <module>' to run.")
        child_env = dict(AppLauncher.process_environ() if env is None else env)
        child_env[AppLauncher.PYTHON_ARGV_VAR] = json.dumps(argv)
        # The child's temp root in a form it can open too (see ansi_safe_path): with
        # TEMP under "Жук", Maya fell back to its CURRENT DIRECTORY for temp files.
        # TMPDIR too: a Python child's tempfile reads it first (TestSandbox sets it).
        for key, value in child_env.items():
            if key.upper() in ("TEMP", "TMP", "TMPDIR") and value:
                child_env[key] = AppLauncher.ansi_safe_path(value)
        return ["-c", AppLauncher._PYTHON_ARGV_SHIM], child_env

    #: Offending folders :meth:`ansi_safe_path` has already warned about (once each).
    _ANSI_WARNED = set()
    #: The SYSTEM ANSI code page's codec, read once (see :meth:`_ansi_codec`).
    _ANSI_CODEC = None

    @staticmethod
    def _ansi_codec():
        """The codec of the SYSTEM ANSI code page: the one a program without a
        UTF-8 manifest (Maya) reads paths in.

        Not this process's code page: Blender's manifest declares UTF-8, so inside
        Blender ``mbcs`` and ``GetACP()`` are 65001 (measured, Blender 5.1) while
        the mayapy it launches reads cp1252 -- a check against ``mbcs`` there
        passes every path Maya cannot open.
        """
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        if AppLauncher._ANSI_CODEC is None:
            import codecs

            codec = "mbcs"
            try:
                import winreg

                key = r"SYSTEM\CurrentControlSet\Control\Nls\CodePage"
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                    acp = str(winreg.QueryValueEx(handle, "ACP")[0]).strip()
                name = "utf-8" if acp == "65001" else f"cp{acp}"
                codecs.lookup(name)
                codec = name
            except (ImportError, OSError, LookupError):
                pass
            AppLauncher._ANSI_CODEC = codec
        return AppLauncher._ANSI_CODEC

    @staticmethod
    def _ansi_encodable(text):
        """Whether *text* survives the system's ANSI code page (no best-fit)."""
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        try:
            text.encode(AppLauncher._ansi_codec())
        except UnicodeEncodeError:
            return False
        return True

    @staticmethod
    def _short_name(path):
        """The 8.3 form of the EXISTING *path* (``GetShortPathNameW``), else None."""
        import ctypes
        from ctypes import wintypes

        get = ctypes.windll.kernel32.GetShortPathNameW
        get.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get.restype = wintypes.DWORD
        size = get(path, None, 0)
        if not size:
            return None
        buffer = ctypes.create_unicode_buffer(size)
        return buffer.value if get(path, buffer, size) else None

    @staticmethod
    def ansi_safe_path(path, ascii_only=False):
        """*path* spelled so a program that reads paths in the ANSI code page can open it.

        Maya opens and saves files through the ANSI code page. Measured on Maya
        2025 (cp1252 here), with folders under "José Жук": ``cmds.file`` save gave
        "An invalid path was specified", open gave "File not found", and a TEMP
        there read as "???" so Maya put its own temp files in the current
        directory. The 8.3 short form of the same folders worked for all three.
        So every component the code page cannot hold is swapped for its short name
        (``GetShortPathNameW``); every other component is kept as written, so a
        template that reads a payload's own name still sees it. A component not
        yet on disk (a file the child will write) is kept as written.

        With no short name to use (8.3 names switched off on that volume, or the
        offending component not yet created), the path is returned unchanged and
        one warning names the remedy. It never falls back to a shared folder: a
        payload is private to the user. A no-op off Windows and for a path the
        code page already holds.

        *ascii_only* is for a path written INTO a file that another program
        decodes as ANSI whatever the code page could hold: RizomUV 2020.1 reads the
        UTF-8 bytes of a Lua script's paths that way, so an "é" (inside cp1252)
        breaks there as surely as a Cyrillic letter -- measured, ``ZomLoad`` /
        ``ZomSave`` under a "José Ångström" folder timed out and the 8.3 form
        passed. Every non-ASCII component is then swapped.

        Parameters:
            path: A file or folder path (``str`` or path-like), or ``None``.
            ascii_only: Swap every non-ASCII component, not only those the ANSI
                code page cannot hold.

        Returns:
            The path as given when nothing needs changing, else its ANSI-safe
            ``str`` form (absolute).
        """
        import re
        import sys
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        if sys.platform != "win32" or not path:
            return path
        text = os.fspath(path)
        fits = str.isascii if ascii_only else AppLauncher._ansi_encodable
        if fits(text):
            return path
        drive, tail = os.path.splitdrive(os.path.abspath(text))
        written = safe = drive + os.sep
        for part in (p for p in re.split(r"[\\/]+", tail) if p):
            written = os.path.join(written, part)
            if not fits(part):
                short = AppLauncher._short_name(written)
                part = os.path.basename(short) if short else ""
                if not part or not fits(part):
                    if written not in AppLauncher._ANSI_WARNED:
                        AppLauncher._ANSI_WARNED.add(written)
                        where = (
                            "plain ASCII"
                            if ascii_only
                            else "this system's ANSI code page"
                        )
                        logger.warning(
                            f"{written} holds a character outside {where} and "
                            "has no 8.3 short name, so Maya (and any program that "
                            "reads paths in the ANSI code page, RizomUV's scripts "
                            "included) cannot open files under it. Point TEMP and TMP at a "
                            "folder whose path is plain ASCII (e.g. C:\\Temp), or "
                            "move the file to one."
                        )
                    return text
            safe = os.path.join(safe, part)
        return safe

    @staticmethod
    def handoff_env(source_root):
        """Child env for launching a DIFFERENT app: this process's env, minus its
        app-private ``OCIO`` config.

        A DCC hand-off launches app B from inside app A, so B inherits A's whole
        environment -- including an ``OCIO`` var pointing INSIDE A's own install
        tree (e.g. Blender's bundled ``ocio_profile_version: 2.5`` config), which
        B's OpenColorIO runtime may be unable to load: Maya 2025 (OCIO 2.3) then
        fails color-management init on every bridge launch. A config *outside*
        the source install (a studio ACES pipeline config) is deliberate
        cross-app state and passes through untouched.

        Reads (and, when stripping, copies) the LIVE process environment via
        :meth:`process_environ`, never ``os.environ`` -- Blender sets ``OCIO``
        at the C level AFTER Python init, so the snapshot can neither see nor
        strip the very variable this hook exists for.

        :param source_root: The RUNNING app's installation root. ``OCIO`` is
                            stripped only when its path lies under this tree.
        :return: A copied env dict without ``OCIO``, or ``None`` (= inherit
                 unchanged) when there is nothing to strip.
        """
        from pathlib import Path
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        environ = AppLauncher.process_environ()
        ocio = environ.get("OCIO")
        if not (ocio and source_root):
            return None
        try:
            private = Path(ocio).resolve().is_relative_to(Path(source_root).resolve())
        except (OSError, ValueError):
            return None
        if not private:
            return None
        env = dict(environ)
        del env["OCIO"]
        logger.info(
            f"Launch env: dropped OCIO ({ocio}) -- private to the running app's "
            f"install ({source_root}); the target app gets its own default."
        )
        return env

    @staticmethod
    def desktop_env():
        """Child env for a desktop helper (``xdg-open``, a file manager) on POSIX.

        The LIVE environment (:meth:`process_environ`) minus the loader and
        interpreter overrides a host app sets for itself
        (``LD_LIBRARY_PATH``, ``PYTHONHOME``, ``QT_PLUGIN_PATH``, ...): run from
        inside Maya or Blender, ``xdg-open`` hands them to the file manager it
        starts, which then loads the host's Qt or libpython in place of its own.

        :return: The stripped env dict, or ``None`` (= inherit unchanged) on
                 Windows, whose shell resolves its own libraries.
        """
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        if platform.system().lower() == "windows":
            return None
        env = AppLauncher.process_environ()
        for key in AppLauncher._HOST_PRIVATE_VARS:
            env.pop(key, None)
        return env

    @staticmethod
    def append_to_path(path, user_scope=True):
        """
        Appends a directory to the system PATH.

        :param path: The directory path to append.
        :param user_scope: If True, updates User environment variables (persistent):
                           the registry on Windows, one marked ``export`` line in
                           ``~/.profile`` elsewhere (read by login shells and the
                           display manager -- desktop-launched apps see it from
                           the next login).
                           If False, only updates current process environment (temporary).
        :return: True if successful.
        """
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        if not path or not os.path.isdir(path):
            return False

        # Always update current process
        current_path = os.environ.get("PATH", "")
        entries = [os.path.normcase(p) for p in current_path.split(os.pathsep)]
        if os.path.normcase(path) not in entries:
            os.environ["PATH"] = f"{current_path}{os.pathsep}{path}"

        if not user_scope:
            return True

        if platform.system().lower() == "windows":
            import winreg

            try:
                key_path = r"Environment"
                root_key = winreg.HKEY_CURRENT_USER

                with winreg.OpenKey(
                    root_key, key_path, 0, winreg.KEY_ALL_ACCESS
                ) as key:
                    try:
                        old_path, _ = winreg.QueryValueEx(key, "Path")
                    except OSError:
                        old_path = ""

                    if path.lower() in old_path.lower().split(os.pathsep):
                        return True  # Already there

                    new_path = f"{old_path}{os.pathsep}{path}" if old_path else path
                    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)

                    return True
            except Exception as e:
                logger.error(f"Failed to update registry PATH: {e}")
                return False

        line = AppLauncher._profile_path_line(path)
        profile = os.path.expanduser("~/.profile")
        try:
            if line in AppLauncher._read_lines(profile):
                return True  # Already there
            with open(profile, "a", encoding="utf-8") as f:
                f.write(f"\n{line}\n")
            return True
        except OSError as e:
            logger.error(f"Failed to update {profile}: {e}")
            return False

    @staticmethod
    def is_path_persisted(path):
        """
        Checks if the path is permanently stored in the system configuration (e.g. Windows Registry).
        useful to avoid prompting the user repeatedly if they haven't restarted their shell.
        """
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        if not path:
            return False

        path_norm = os.path.normpath(path).lower()

        if platform.system().lower() == "windows":
            import winreg

            try:
                # Check User Environment
                key_path = r"Environment"
                try:
                    with winreg.OpenKey(
                        winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ
                    ) as key:
                        val, _ = winreg.QueryValueEx(key, "Path")
                        if path_norm in [
                            os.path.normpath(p).lower()
                            for p in val.split(os.pathsep)
                            if p
                        ]:
                            return True
                except OSError:
                    pass

                # Check System Environment (ReadOnly usually)
                key_path_lm = (
                    r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
                )
                try:
                    with winreg.OpenKey(
                        winreg.HKEY_LOCAL_MACHINE, key_path_lm, 0, winreg.KEY_READ
                    ) as key:
                        val, _ = winreg.QueryValueEx(key, "Path")
                        if path_norm in [
                            os.path.normpath(p).lower()
                            for p in val.split(os.pathsep)
                            if p
                        ]:
                            return True
                except OSError:
                    pass
            except Exception:
                pass
            return False

        profile = os.path.expanduser("~/.profile")
        return AppLauncher._profile_path_line(path) in AppLauncher._read_lines(profile)
