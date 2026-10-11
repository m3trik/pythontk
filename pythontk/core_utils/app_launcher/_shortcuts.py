# !/usr/bin/python
# coding=utf-8
"""Desktop shortcuts: a launcher on the desktop or in the start menu.

``create_shortcut`` writes a Windows ``.lnk`` or a freedesktop ``.desktop``
entry that runs a program with its arguments. On Windows the link is built by
the shell's own ``IShellLinkW`` over ctypes, never ``WScript.Shell``: that
scripting object goes through the ANSI code page, so a name or path holding a
character outside it came back as "?" and the save failed (measured with a
"★"). The same interface stamps the AppUserModelID a program sets for itself,
which is what lets the taskbar group the program's windows under the shortcut
and pin it.

One job of :class:`AppLauncher`, composed in ``_app_launcher.py``. A cross-call
goes through the facade (imported on use: that module imports this one for its
base), so a caller's ``patch.object(AppLauncher, ...)`` reaches it.
"""

import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from typing import Optional, Sequence


class _ShortcutsMixin:
    """Desktop and start-menu shortcuts; a private part of :class:`AppLauncher`."""

    #: The named places a shortcut can go; any existing directory works too.
    SHORTCUT_LOCATIONS = ("desktop", "start_menu")

    @staticmethod
    def create_shortcut(
        name: str,
        target: str,
        args: Optional[Sequence[str]] = None,
        location: str = "desktop",
        icon: Optional[str] = None,
        working_dir: Optional[str] = None,
        comment: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> str:
        """Write a shortcut that runs *target* with *args*.

        Writing one again with the same *name* and *location* replaces it, so a
        shortcut whose target moved is refreshed by creating it again.

        Parameters:
            name: The label the user sees. Characters a file name cannot hold
                are replaced in the file name only.
            target: The program to run, as an absolute path.
            args: Its arguments, each quoted as the platform requires; one
                string is one argument, as for :meth:`launch`.
            location: ``"desktop"``, ``"start_menu"`` (Windows' Start menu
                Programs folder; the application menu on Linux), or an existing
                directory to write into.
            icon: An image for the shortcut: ``.ico`` (or an ``.exe`` /
                ``.dll``) on Windows, ``.png`` / ``.svg`` on Linux. ``None``
                shows the target's own icon.
            working_dir: The directory the program starts in; the user's home
                folder when omitted, so the program never starts in a system
                folder (``python -m`` imports from its working directory).
            comment: A one-line description (the tooltip on Windows,
                ``Comment=`` on Linux).
            app_id: Windows only: the AppUserModelID the program sets for
                itself (uitk's ``Bootstrap.set_app_id``). Stamped on the
                shortcut, it makes the taskbar group the program's windows
                under this shortcut, with its icon, and pin it as a launcher of
                the program rather than of the bare interpreter.

        Returns:
            The shortcut file's path.

        Raises:
            NotImplementedError: Not on Windows or Linux.
            FileNotFoundError: *location* is a directory that does not exist.
            OSError: The shell refused to write the shortcut.
        """
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher

        if isinstance(args, str):
            args = [args]
        args = [str(arg) for arg in (args or ())]
        working_dir = working_dir or os.path.expanduser("~")
        if sys.platform == "win32":
            folder = AppLauncher._shortcut_folder(location)
            path = os.path.join(folder, _ShortcutFiles.file_name(name) + ".lnk")
            _ShellLink.write(
                path,
                target,
                subprocess.list2cmdline(args),
                icon=icon,
                working_dir=working_dir,
                comment=comment,
                app_id=app_id,
            )
            return path
        if sys.platform.startswith("linux"):
            folder = AppLauncher._shortcut_folder(location)
            path = os.path.join(folder, _ShortcutFiles.slug(name) + ".desktop")
            _DesktopEntry.write(
                path,
                name,
                [target, *args],
                icon=icon,
                working_dir=working_dir,
                comment=comment,
            )
            return path
        raise NotImplementedError(
            f"Shortcuts are written on Windows and Linux, not on {sys.platform!r}."
        )

    @staticmethod
    def _shortcut_folder(location: str) -> str:
        """The directory a shortcut at *location* goes in, created when named.

        An explicit directory must already exist: a typo there should fail,
        not leave a shortcut in a folder nobody looks in.
        """
        if location not in _ShortcutsMixin.SHORTCUT_LOCATIONS:
            if not os.path.isdir(location):
                raise FileNotFoundError(
                    f"No such directory for a shortcut: {location!r} (or one of "
                    f"{_ShortcutsMixin.SHORTCUT_LOCATIONS})."
                )
            return location
        if sys.platform == "win32":
            folder = _ShellLink.known_folder(location)
        else:
            folder = _DesktopEntry.folder(location)
        os.makedirs(folder, exist_ok=True)
        return folder


class _ShortcutFiles:
    """The file name a shortcut's label becomes."""

    @staticmethod
    def file_name(name: str) -> str:
        """*name* with what a Windows file name cannot hold replaced by ``_``."""
        clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
        return clean or "shortcut"

    @staticmethod
    def slug(name: str) -> str:
        """A desktop-file id for *name*: words joined by ``-``, no spaces."""
        clean = re.sub(r"[^\w.]+", "-", name).strip("-.")
        return clean or "shortcut"


class _ShellLink:
    """A Windows ``.lnk`` written through the shell's ``IShellLinkW`` over ctypes.

    Unicode end to end, unlike ``WScript.Shell`` (see the module docstring).
    Each COM call goes through its interface's vtable slot; the slot numbers
    below follow ``shobjidl.h`` / ``objidl.h`` / ``propsys.h``, IUnknown's
    three methods first.
    """

    _CLSID_SHELL_LINK = "00021401-0000-0000-C000-000000000046"
    _IID_SHELL_LINK_W = "000214F9-0000-0000-C000-000000000046"
    _IID_PERSIST_FILE = "0000010B-0000-0000-C000-000000000046"
    _IID_PROPERTY_STORE = "886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"
    _PKEY_APP_USER_MODEL_ID = ("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3", 5)
    _KNOWN_FOLDERS = {
        "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",  # FOLDERID_Desktop
        "start_menu": "A77F5D77-2E2B-44C3-A6A2-ABA601054A51",  # FOLDERID_Programs
    }

    _QUERY_INTERFACE, _RELEASE = 0, 2
    _SET_DESCRIPTION, _SET_WORKING_DIRECTORY, _SET_ARGUMENTS = 7, 9, 11
    _SET_ICON_LOCATION, _SET_PATH = 17, 20
    _PERSIST_SAVE = 6
    _STORE_SET_VALUE, _STORE_COMMIT = 6, 7
    _VT_LPWSTR = 31

    @staticmethod
    def _guid(text: str):
        import ctypes

        class GUID(ctypes.Structure):
            _fields_ = [
                ("data1", ctypes.c_ulong),
                ("data2", ctypes.c_ushort),
                ("data3", ctypes.c_ushort),
                ("data4", ctypes.c_ubyte * 8),
            ]

        return GUID.from_buffer_copy(uuid.UUID(text).bytes_le)

    @staticmethod
    def _method(pointer, slot: int, *argtypes, restype=None):
        """Interface method *slot* of the COM object at *pointer*, bound to it."""
        import ctypes

        vtable = ctypes.cast(
            pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
        ).contents
        prototype = ctypes.WINFUNCTYPE(
            restype or ctypes.HRESULT, ctypes.c_void_p, *argtypes
        )
        function = prototype(vtable[slot])
        return lambda *args: function(pointer, *args)

    @classmethod
    def _release(cls, pointer) -> None:
        import ctypes

        if pointer:
            cls._method(pointer, cls._RELEASE, restype=ctypes.c_ulong)()

    @classmethod
    def _query(cls, pointer, iid: str):
        """QueryInterface *pointer* for *iid*; the caller releases the result."""
        import ctypes

        out = ctypes.c_void_p()
        guid = cls._guid(iid)
        cls._method(pointer, cls._QUERY_INTERFACE, ctypes.c_void_p, ctypes.c_void_p)(
            ctypes.byref(guid), ctypes.byref(out)
        )
        return out

    @classmethod
    def known_folder(cls, location: str) -> str:
        """The shell's path for *location* (``SHGetKnownFolderPath``): the
        real Desktop even when it is redirected (OneDrive, a roaming profile)."""
        import ctypes
        from ctypes import wintypes

        prototype = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_void_p),
        )
        get_path = prototype(("SHGetKnownFolderPath", ctypes.windll.shell32))
        folder_id = cls._guid(cls._KNOWN_FOLDERS[location])
        buffer = ctypes.c_void_p()
        get_path(ctypes.byref(folder_id), 0, None, ctypes.byref(buffer))
        try:
            return ctypes.wstring_at(buffer)
        finally:
            ctypes.windll.ole32.CoTaskMemFree(buffer)

    @classmethod
    def write(
        cls,
        path: str,
        target: str,
        arguments: str,
        icon: Optional[str] = None,
        working_dir: Optional[str] = None,
        comment: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> None:
        """Save a shell link at *path* (replacing one there)."""
        import ctypes

        ole32 = ctypes.windll.ole32
        # S_OK / S_FALSE are ours to balance. RPC_E_CHANGED_MODE means a host
        # (Qt, a DCC) already runs COM in another apartment model on this
        # thread, which serves an in-process shell link just as well.
        owns_com = ole32.CoInitializeEx(None, 0x2) in (0, 1)  # APARTMENTTHREADED
        link = ctypes.c_void_p()
        try:
            create = ctypes.WINFUNCTYPE(
                ctypes.HRESULT,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.c_void_p,
                ctypes.c_void_p,
            )(("CoCreateInstance", ole32))
            clsid = cls._guid(cls._CLSID_SHELL_LINK)
            iid = cls._guid(cls._IID_SHELL_LINK_W)
            create(ctypes.byref(clsid), None, 1, ctypes.byref(iid), ctypes.byref(link))

            text = ctypes.c_wchar_p
            cls._method(link, cls._SET_PATH, text)(target)
            cls._method(link, cls._SET_ARGUMENTS, text)(arguments)
            if working_dir:
                cls._method(link, cls._SET_WORKING_DIRECTORY, text)(working_dir)
            if comment:
                cls._method(link, cls._SET_DESCRIPTION, text)(comment)
            if icon:
                cls._method(link, cls._SET_ICON_LOCATION, text, ctypes.c_int)(icon, 0)
            if app_id:
                cls._stamp_app_id(link, app_id)

            persist = cls._query(link, cls._IID_PERSIST_FILE)
            try:
                cls._method(persist, cls._PERSIST_SAVE, text, ctypes.c_int)(
                    os.path.abspath(path), 1
                )
            finally:
                cls._release(persist)
        finally:
            cls._release(link)
            if owns_com:
                ole32.CoUninitialize()

    @classmethod
    def _stamp_app_id(cls, link, app_id: str) -> None:
        """Set ``System.AppUserModel.ID`` in the link's property store."""
        import ctypes

        class PropertyKey(ctypes.Structure):
            _fields_ = [("fmtid", ctypes.c_ubyte * 16), ("pid", ctypes.c_ulong)]

        class PropVariant(ctypes.Structure):
            # vt, three reserved words, then the union (here: an LPWSTR); the
            # pad keeps the struct at PROPVARIANT's full size on 64-bit.
            _fields_ = [
                ("vt", ctypes.c_ushort),
                ("reserved", ctypes.c_ushort * 3),
                ("value", ctypes.c_wchar_p),
                ("pad", ctypes.c_void_p),
            ]

        fmtid, pid = cls._PKEY_APP_USER_MODEL_ID
        key = PropertyKey(
            (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(fmtid).bytes_le), pid
        )
        value = PropVariant(vt=cls._VT_LPWSTR, value=app_id)
        store = cls._query(link, cls._IID_PROPERTY_STORE)
        try:
            cls._method(store, cls._STORE_SET_VALUE, ctypes.c_void_p, ctypes.c_void_p)(
                ctypes.byref(key), ctypes.byref(value)
            )
            cls._method(store, cls._STORE_COMMIT)()
        finally:
            cls._release(store)


class _DesktopEntry:
    """A freedesktop ``.desktop`` launcher (the Desktop Entry Specification)."""

    #: Characters that make an ``Exec`` argument need quoting.
    _RESERVED = set(" \t\n\"'\\><~|&;$*?#()`")

    @staticmethod
    def folder(location: str) -> str:
        """The user's desktop (``xdg-user-dir DESKTOP``), or their application
        menu directory (``$XDG_DATA_HOME/applications``)."""
        from pythontk.core_utils.app_launcher._app_launcher import AppLauncher
        from pythontk.core_utils.user_config import UserConfig

        if location == "start_menu":
            return os.path.join(UserConfig.xdg_home("data"), "applications")
        if shutil.which("xdg-user-dir"):
            try:
                result = AppLauncher.run("xdg-user-dir", ["DESKTOP"], timeout=10)
                folder = (result.stdout or "").strip()
                if result.returncode == 0 and os.path.isabs(folder):
                    return folder
            except Exception:  # noqa: BLE001 -- fall back to the spec's default
                pass
        return os.path.join(os.path.expanduser("~"), "Desktop")

    @staticmethod
    def _string(value: str) -> str:
        """A ``string``/``localestring`` value: the spec's escapes applied."""
        return (
            value.replace("\\", "\\\\")
            .replace("\n", "\\n")
            .replace("\t", "\\t")
            .replace("\r", "\\r")
        )

    @classmethod
    def exec_arg(cls, arg: str) -> str:
        """One ``Exec`` argument, quoted by the spec's two rules.

        The quoting rule wraps an argument holding a reserved character (or an
        empty one, which would otherwise vanish) in double quotes,
        backslash-escaping ``"`` ````` ``$`` ``\\`` inside them; the general
        string escape is applied on top, so a literal backslash in a quoted
        argument is written as four and a line break as ``\\n``. ``%`` doubles
        (field codes).
        """
        arg = arg.replace("%", "%%")
        if not arg or any(char in cls._RESERVED for char in arg):
            arg = '"' + re.sub(r'(["`$\\])', r"\\\1", arg) + '"'
        return cls._string(arg)

    @classmethod
    def write(
        cls,
        path: str,
        name: str,
        command: Sequence[str],
        icon: Optional[str] = None,
        working_dir: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> None:
        """Save a launcher at *path* running *command* (program first)."""
        lines = [
            "[Desktop Entry]",
            "Type=Application",
            "Version=1.0",
            f"Name={cls._string(name)}",
        ]
        if comment:
            lines.append(f"Comment={cls._string(comment)}")
        lines.append("Exec=" + " ".join(cls.exec_arg(arg) for arg in command))
        if working_dir:
            lines.append(f"Path={cls._string(working_dir)}")
        if icon:
            lines.append(f"Icon={cls._string(icon)}")
        lines += ["Terminal=false", "StartupNotify=true"]
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
        # A launcher on the desktop runs only when executable (GNOME, KDE).
        mode = os.stat(path).st_mode
        os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
