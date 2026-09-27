# !/usr/bin/python
# coding=utf-8
"""Where an app is installed: its executable, and the Python it pairs with.

``find_app`` (an absolute path, ``PATH``, then Windows App Paths or the XDG
``.desktop`` entries), ``resolve_app_path`` (``$ENV`` -> ``find_app`` ->
install-dir scan, first hit wins), ``scan_install_dirs`` /
``scan_for_executables`` (the install-layout globs), and ``companion_python`` /
``looks_like_python`` (the interpreter a host binary ships beside it).

One job of :class:`AppLauncher`, composed in ``_app_launcher.py``. A cross-call
goes through the facade (imported on use: that module imports this one for its
base), so a caller's ``patch.object(AppLauncher, ...)`` reaches it.
"""

import os
import sys
import glob
import shutil
import platform
from typing import Optional


class _DiscoveryMixin:
    """Install and interpreter discovery; a private part of :class:`AppLauncher`."""

    #: Programs that start ANOTHER app, so a menu entry running one names the
    #: app it starts, never the program: a sandbox (flatpak, snap, ``env``
    #: before a snap), a game store (Steam's ``Name=Blender`` entry runs
    #: ``steam steam://rungameid/365670``), a shell, interpreter or runtime
    #: running a script, an opener. ``fnmatch`` patterns on the file name.
    _LAUNCHER_PROGRAMS = (
        "flatpak",
        "snap",
        "env",
        "steam",
        "lutris",
        "heroic",
        "gtk-launch",
        "xdg-open",
        "sh",
        "bash",
        "python*",
        "wine*",
        "java",
    )

    #: Freedesktop ``Exec=`` field codes: the files, URLs and icon the menu
    #: fills in at launch, never an argument of the app's own.
    _DESKTOP_FIELD_CODES = frozenset("%" + code for code in "fFuUdDnNickvm")

    @staticmethod
    def _desktop_entry_dirs():
        """XDG application dirs, user first (``$XDG_DATA_HOME``, then
        ``$XDG_DATA_DIRS``), each also as its flatpak export. A relative path
        in either is invalid by the XDG spec and ignored."""
        from pythontk.core_utils.user_config import UserConfig

        home = UserConfig.xdg_home("data")
        data_dirs = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
        roots = [home] + [d for d in data_dirs.split(":") if os.path.isabs(d)]
        roots += [
            os.path.join(home, "flatpak", "exports", "share"),
            "/var/lib/flatpak/exports/share",
        ]
        return [os.path.join(r, "applications") for r in dict.fromkeys(roots)]

    @staticmethod
    def _find_desktop_entry_app(app_identifier):
        """An installed app's executable from its ``.desktop`` entry, or None.

        Matches the entry's file name or ``Name=`` (case-insensitively) and
        takes ``Exec=``'s program -- only when the entry is that program's
        own: the program is no launcher (:attr:`_LAUNCHER_PROGRAMS`) and
        nothing but options and field codes (``%f``, ``%U``) follows it. A
        positional argument is what the program runs -- a URL, a script, a
        sandboxed app's id -- so the program is not the app.
        """
        import fnmatch
        import shlex
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        wanted = app_identifier.lower()
        for folder in AppLauncher._desktop_entry_dirs():
            try:
                names = sorted(os.listdir(folder))
            except OSError:
                continue
            for fn in names:
                if not fn.endswith(".desktop"):
                    continue
                fields = {}
                section = None
                for line in AppLauncher._read_lines(os.path.join(folder, fn)):
                    line = line.strip()
                    if line.startswith("["):
                        section = line
                    elif section == "[Desktop Entry]" and "=" in line:
                        key, _, value = line.partition("=")
                        fields.setdefault(key.strip(), value.strip())
                stem = fn[: -len(".desktop")].lower()
                if wanted not in (stem, fields.get("Name", "").lower()):
                    continue
                try:
                    argv = shlex.split(fields.get("Exec", ""))
                except ValueError:
                    continue
                if not argv or any(
                    fnmatch.fnmatchcase(os.path.basename(argv[0]), launcher)
                    for launcher in AppLauncher._LAUNCHER_PROGRAMS
                ):
                    continue
                if any(
                    not arg.startswith("-")
                    and arg not in AppLauncher._DESKTOP_FIELD_CODES
                    for arg in argv[1:]
                ):
                    continue
                program = argv[0] if os.path.isabs(argv[0]) else shutil.which(argv[0])
                if program and os.path.isfile(program):
                    return program
        return None

    @staticmethod
    def scan_for_executables(root_paths, executable_name, depth=3):
        """
        Scans directories for a specific executable.

        :param root_paths: List of root directories to search.
        :param executable_name: Name of the executable (e.g. 'maya.exe').
        :param depth: Max folder depth to search.
        :return: List of absolute paths found, newest first -- a natural sort,
                 as :meth:`scan_install_dirs` ranks (``4.10`` over ``4.9``).
        """
        from pythontk.str_utils._str_utils import StrUtils

        found = []
        if isinstance(root_paths, str):
            root_paths = [root_paths]

        for root in root_paths:
            if not os.path.exists(root):
                continue

            for dirpath, dirnames, filenames in os.walk(root):
                # Calculate current depth relative to root
                try:
                    rel_path = os.path.relpath(dirpath, root)
                    current_depth = (
                        0 if rel_path == "." else len(rel_path.split(os.sep))
                    )

                    if current_depth >= depth:
                        # Don't descend further, clear dirs to stop walk
                        dirnames[:] = []
                        continue
                except ValueError:
                    continue

                for f in filenames:
                    if f.lower() == executable_name.lower():
                        found.append(os.path.join(dirpath, f))

        return sorted(
            found,
            key=lambda p: StrUtils.natural_sort_key(p, ignore_case=True),
            reverse=True,
        )

    @staticmethod
    def _program_files_roots():
        """Return the 64-bit and 32-bit ``Program Files`` roots from the environment."""
        return (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        )

    @staticmethod
    def scan_install_dirs(scan_globs):
        """Yield existing files matching *scan_globs*; pattern order is a priority.

        Every match for the first pattern is yielded (newest install dir
        first, via reverse sort) before any match for the second pattern,
        and so on — mirroring how :meth:`resolve_app_path` ranks its stages
        (first hit wins). The previous pooled sort made the *filename* an
        accidental tiebreaker: RizomUV's ``rizomuv_RS.exe`` outranked the
        intended ``Rizomuv_VS.exe`` inside the same install dir purely by
        ASCII order, inverting the caller's tuple priority.

        A ``{program_files}`` token in a pattern expands to both Program Files roots,
        so callers write one portable pattern instead of hardcoding the 64/32-bit
        dirs; such a pattern is a Windows install layout and is skipped
        elsewhere. ``{exe}`` is ``.exe`` on Windows and empty elsewhere, and a
        leading ``~`` is the user's home -- so one spec lists every OS's
        layouts::

            AppLauncher.scan_install_dirs([
                r"{program_files}\\Autodesk\\Maya*\\bin\\maya.exe",
                "/usr/autodesk/maya*/bin/maya",
                "~/blender-*/blender{exe}",
            ])

        "Newest" is a natural sort: ``Blender 4.10`` outranks ``Blender 4.9``.

        :param scan_globs: An ordered iterable of glob patterns (each may use the tokens).
        :return: A generator of absolute file paths — highest-priority pattern
                 first, newest first within each pattern, duplicates skipped.
        """
        from pythontk.str_utils._str_utils import StrUtils
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        windows = platform.system().lower() == "windows"
        exe = ".exe" if windows else ""
        seen = set()
        for pattern in scan_globs:
            if "{program_files}" in pattern:
                if not windows:
                    continue
                roots = AppLauncher._program_files_roots()
            else:
                roots = (None,)
            candidates = []
            for root in roots:
                pat = pattern.replace("{exe}", exe)
                if root is not None:
                    pat = pat.replace("{program_files}", root)
                candidates.extend(glob.glob(os.path.expanduser(pat)))
            for c in sorted(
                set(candidates),
                key=lambda p: StrUtils.natural_sort_key(p, ignore_case=True),
                reverse=True,
            ):
                if c not in seen and os.path.isfile(c):
                    seen.add(c)
                    yield c

    @staticmethod
    def resolve_app_path(
        *, env_vars=(), location_env_vars=(), app_names=(), scan_globs=()
    ):
        """Resolve a target application executable; the first hit wins.

        Consolidates the ``$ENV -> find_app -> install-dir scan`` discovery that
        app hand-off callers (Maya / Blender / RizomUV / Painter / Toolbag) would
        otherwise each re-implement. Resolution order:

        1. *env_vars* -- each name whose ``os.environ`` value points at an existing
           file (e.g. ``BLENDER_EXE`` / ``MAYA_EXE``).
        2. *location_env_vars* -- each ``(env_var, suffix)`` where
           ``<env_value>/<suffix>`` exists (e.g.
           ``("MAYA_LOCATION", ("bin", "maya{exe}"))``). *suffix* may be a string
           or a path-segment sequence; ``{exe}`` is ``.exe`` on Windows only.
        3. *app_names* -- each name via :meth:`find_app` (PATH / Windows App Paths).
        4. *scan_globs* -- :meth:`scan_install_dirs` over the patterns (newest wins).

        :return: The absolute path, or ``None`` when nothing resolves.
        """
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        for var in env_vars:
            p = os.environ.get(var)
            if p and os.path.isfile(p):
                return p

        exe = ".exe" if platform.system().lower() == "windows" else ""
        for var, suffix in location_env_vars:
            loc = os.environ.get(var)
            if not loc:
                continue
            parts = [suffix] if isinstance(suffix, str) else list(suffix)
            p = os.path.join(loc, *(part.replace("{exe}", exe) for part in parts))
            if os.path.isfile(p):
                return p

        for name in app_names:
            found = AppLauncher.find_app(name)
            if found:
                return found

        for found in AppLauncher.scan_install_dirs(scan_globs):
            return found
        return None

    @staticmethod
    def looks_like_python(path: str) -> bool:
        """Whether *path* names a Python interpreter rather than a host app.

        By name only (``python``, ``python3.11``, ``mayapy``, ``3dsmaxpy``,
        ``hython``): a host's GUI binary (``maya``, ``blender``) runs ``-c`` as
        its own language, or opens a second copy of the app.
        """
        name = os.path.splitext(os.path.basename(path or ""))[0].lower()
        return "python" in name or name.endswith("py") or name == "hython"

    @staticmethod
    def companion_python(executable: Optional[str] = None) -> Optional[str]:
        """The Python interpreter that pairs with *executable* (default: this process).

        *executable* itself when it is a Python; else a companion beside it --
        ``{name}py`` (``maya`` / ``mayabatch`` -> ``mayapy``, ``3dsmax`` ->
        ``3dsmaxpy``), then a sibling ``python`` / ``python3`` / ``hython``; and,
        only when *executable* is this process's own, the interpreter bundled
        under ``sys.prefix`` (Blender: ``<ver>/python/bin``). ``sys.prefix``
        describes THIS process, so a question about another install never
        borrows it. A companion shares the host's site-packages, so ``-m pip``
        through it installs where the host imports from.

        Parameters:
            executable: A host or interpreter binary; ``sys.executable`` when omitted.

        Returns:
            The interpreter's path, or ``None`` when *executable* is a host with
            none discoverable -- never the host binary itself.
        """
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        own = sys.executable or ""
        executable = executable or own
        if executable and AppLauncher.looks_like_python(executable):
            return executable

        ext = ".exe" if sys.platform == "win32" else ""
        candidates = []
        if executable:
            folder = os.path.dirname(executable)
            base = os.path.basename(executable).lower()
            base = base.split(".")[0].replace("batch", "")  # maya.bin, mayabatch
            for name in (base + "py", "python", "python3", "hython"):
                candidates.append(os.path.join(folder, name + ext))
        if own and os.path.normcase(executable) == os.path.normcase(own):
            prefixes = dict.fromkeys(
                p for p in (sys.prefix, sys.base_prefix, sys.exec_prefix) if p
            )
            for prefix in prefixes:
                candidates.append(os.path.join(prefix, "python" + ext))
                candidates.append(os.path.join(prefix, "bin", "python" + ext))
                candidates.append(os.path.join(prefix, "bin", "python3" + ext))
                candidates.extend(
                    sorted(glob.glob(os.path.join(prefix, "bin", "python3.*")))
                )
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    @staticmethod
    def find_app(app_identifier):
        """
        Attempts to locate the executable for the given application identifier.

        :param app_identifier: The name or path of the application.
        :return: The absolute path to the executable, or None if not found.
        """
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        # 1. Check if it's a valid absolute path
        if os.path.isabs(app_identifier) and os.path.exists(app_identifier):
            return app_identifier

        # 2. Check if it's in the PATH
        path_in_env = shutil.which(app_identifier)
        if path_in_env:
            return path_in_env

        system = platform.system().lower()

        # 3. Windows Registry (App Paths)
        if system == "windows":
            return AppLauncher._find_windows_app(app_identifier)

        # 4. Linux: the desktop's application menu (XDG .desktop entries)
        if system == "linux":
            return AppLauncher._find_desktop_entry_app(app_identifier)

        return None

    @staticmethod
    def _find_windows_app(app_name):
        """
        Looks up the application in the Windows Registry App Paths, then falls
        back to scanning the Program Files install roots up to two levels deep
        (covers vendors that don't register App Paths — Adobe Substance 3D
        Painter, Houdini, Reaper, etc.), the newest install first at each
        depth.
        """
        import winreg
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        if not app_name.endswith(".exe"):
            app_name += ".exe"

        reg_paths = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths",  # 32-bit apps on 64-bit OS
        ]

        for reg_path in reg_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                    try:
                        with winreg.OpenKey(key, app_name) as app_key:
                            value, _ = winreg.QueryValueEx(
                                app_key, None
                            )  # Default value contains path
                            if value and os.path.exists(value):
                                return value
                    except FileNotFoundError:
                        continue  # App not in this registry path key
            except OSError:
                continue

        # Also check CURRENT_USER
        for reg_path in reg_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path) as key:
                    try:
                        with winreg.OpenKey(key, app_name) as app_key:
                            value, _ = winreg.QueryValueEx(app_key, None)
                            if value and os.path.exists(value):
                                return value
                    except FileNotFoundError:
                        continue
            except OSError:
                continue

        # Program Files glob fallback. Ranked by scan_install_dirs: each depth
        # newest first (``Blender 5.1`` over ``4.10`` over ``4.9``). glob alone
        # yields the file system's order -- by name on NTFS, the OLDEST first.
        program_files_roots = [
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("ProgramW6432"),
        ]
        for root in dict.fromkeys(r for r in program_files_roots if r):
            for match in AppLauncher.scan_install_dirs(
                (
                    os.path.join(root, app_name),
                    os.path.join(root, "*", app_name),
                    os.path.join(root, "*", "*", app_name),
                )
            ):
                return match

        return None
