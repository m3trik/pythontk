# !/usr/bin/python
# coding=utf-8
"""The platform trash behind :meth:`FileUtils.move_to_trash` (``_TrashInternal``).

Three backends, one contract: the file ends up where the platform's own trash
keeps it, restorable -- or, where its volume keeps no trash that would hold
it, it is not touched at all and the caller hears ``None``. The Windows half
exists because the obvious call is not safe: ``SHFileOperationW`` with
``FOF_ALLOWUNDO`` deletes a file PERMANENTLY, without a word, wherever the
Recycle Bin will not take it -- a network share, a removable drive, a SUBST
drive, a bin set to "remove files immediately" or too small for the file.
Each is checked first, and the bin's own record of the item is looked for
afterwards, so a case nothing here foresaw is at least reported.
"""

import datetime
import logging
import os
import re
import sys
import time
from typing import List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


class _TrashInternal:
    """Private: :meth:`FileUtils.move_to_trash`'s body, one backend per platform."""

    # -- Windows: SHFileOperationW ----------------------------------------------
    _DRIVE_FIXED = 3
    _FO_DELETE = 0x0003
    _FOF_SILENT = 0x0004
    _FOF_NOCONFIRMATION = 0x0010
    _FOF_ALLOWUNDO = 0x0040
    _FOF_NOERRORUI = 0x0400
    #: ``SHFileOperationW`` cannot take a longer path (MAX_PATH, terminator
    #: included); what it does with one is not worth finding out on a user's map.
    _MAX_PATH = 260
    _EXPLORER = r"Software\Microsoft\Windows\CurrentVersion"
    #: Seconds of clock skew allowed between this process and the file system
    #: stamping the bin's ``$I`` record.
    _RECORD_SLACK = 2.0
    #: Waits between looks for that record, which lands just after the call.
    _RECORD_WAITS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)

    @classmethod
    def run(cls, path: str) -> Optional[str]:
        """:meth:`FileUtils.move_to_trash`; see there."""
        path = os.path.abspath(os.fspath(path))
        if not os.path.lexists(path):
            raise FileNotFoundError(2, "No such file", path)
        if os.path.isdir(path) and not os.path.islink(path):
            raise IsADirectoryError(21, "A folder, not a file", path)
        if sys.platform.startswith("win"):
            return cls._windows(path)
        if sys.platform == "darwin":
            return cls._macos(path)
        return cls._freedesktop(path)

    @classmethod
    def can_trash(cls, path: str) -> bool:
        """:meth:`FileUtils.can_trash`; see there."""
        path = os.path.abspath(os.fspath(path))
        if not os.path.lexists(path) or (
            os.path.isdir(path) and not os.path.islink(path)
        ):
            return False
        if sys.platform.startswith("win"):
            return cls._windows_bin_keeps(cls._windows_real_path(path))
        if sys.platform == "darwin":
            device = os.lstat(path).st_dev
            return any(
                os.path.isdir(trash)
                and cls._device_of(trash) == device
                and os.access(trash, os.W_OK)
                for trash in cls._macos_candidates(path)
            )
        return any(cls._writable(trash) for trash in cls._freedesktop_candidates(path))

    @classmethod
    def _writable(cls, folder: str) -> bool:
        """Whether *folder* is, or could be made, a folder this user writes:
        its nearest existing ancestor must be a writable folder."""
        probe = os.path.abspath(folder)
        while not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                return False
            probe = parent
        return os.path.isdir(probe) and os.access(probe, os.W_OK)

    @classmethod
    def _windows(cls, path: str) -> Optional[str]:
        """The Recycle Bin, when the file's volume has one that keeps it."""
        import ctypes
        from ctypes import wintypes

        path = cls._windows_real_path(path)
        if not cls._windows_bin_keeps(path):
            return None

        layout = {
            "_fields_": [
                ("hwnd", wintypes.HWND),
                ("wFunc", wintypes.UINT),
                ("pFrom", wintypes.LPCWSTR),
                ("pTo", wintypes.LPCWSTR),
                ("fFlags", ctypes.c_uint16),
                ("fAnyOperationsAborted", wintypes.BOOL),
                ("hNameMappings", ctypes.c_void_p),
                ("lpszProgressTitle", wintypes.LPCWSTR),
            ]
        }
        if ctypes.sizeof(ctypes.c_void_p) == 4:
            layout["_pack_"] = 1  # byte-packed on 32-bit Windows only (shellapi.h)
        op_struct = type("SHFILEOPSTRUCTW", (ctypes.Structure,), layout)
        started = time.time()
        # pFrom is a DOUBLE-null-terminated list: ctypes adds the second null.
        source = ctypes.c_wchar_p(path + "\0")
        op = op_struct(
            hwnd=None,
            wFunc=cls._FO_DELETE,
            pFrom=source,
            pTo=None,
            fFlags=cls._FOF_ALLOWUNDO
            | cls._FOF_NOCONFIRMATION
            | cls._FOF_SILENT
            | cls._FOF_NOERRORUI,
        )
        result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        if result or op.fAnyOperationsAborted:
            # The Win32 code as ``winerror``: Python picks the subclass (32, a
            # sharing violation, reads as PermissionError, 2 as FileNotFound).
            raise OSError(
                None,
                f"the Recycle Bin did not take it (SHFileOperation 0x{result:X})",
                path,
                result or 1223,  # ERROR_CANCELLED
            )
        if os.path.lexists(path):
            raise OSError(None, "the Recycle Bin left it where it was", path, 1223)
        # The bin writes its record a moment after the call returns (measured:
        # 6 of 8 quick calls in a row had none yet), so the look is retried.
        for delay in cls._RECORD_WAITS:
            item = cls._windows_bin_item(path, started - cls._RECORD_SLACK)
            if item is not None:
                return item
            time.sleep(delay)
        item = cls._windows_bin_item(path, started - cls._RECORD_SLACK)
        if item is None:
            logger.warning(
                "Windows reported %s moved to the Recycle Bin, but no Recycle Bin "
                "record names it: it may have been deleted permanently.",
                path,
            )
            return ""
        return item

    @staticmethod
    def _windows_real_path(path: str) -> str:
        """*path* on the volume a SUBST drive stands for -- the bin of a SUBST
        drive deletes permanently, the bin of its host volume does not. Any
        other path unchanged."""
        import ctypes

        for _hop in range(8):  # a SUBST of a SUBST; never a loop
            drive, rest = os.path.splitdrive(path)
            if len(drive) != 2 or drive[1] != ":":
                return path
            target = ctypes.create_unicode_buffer(1024)
            if not ctypes.windll.kernel32.QueryDosDeviceW(drive, target, 1024):
                return path
            mapped = target.value
            if not mapped.startswith("\\??\\"):
                return path  # a real volume, or a mapped share
            rest = rest.lstrip("\\/")
            if mapped.startswith("\\??\\UNC\\"):  # a SUBST onto a share: no bin
                return "\\\\" + mapped[8:].rstrip("\\") + "\\" + rest
            path = os.path.join(mapped[4:].rstrip("\\") + "\\", rest)
        return path

    @classmethod
    def _windows_bin_keeps(cls, path: str) -> bool:
        """Whether the Recycle Bin of *path*'s volume would KEEP it, rather than
        delete it at once -- every case ``FOF_ALLOWUNDO`` deletes in silently."""
        import ctypes

        if len(path) >= cls._MAX_PATH:
            return False
        kernel32 = ctypes.windll.kernel32
        root = ctypes.create_unicode_buffer(cls._MAX_PATH + 1)
        if not kernel32.GetVolumePathNameW(path, root, len(root)):
            return False
        # A drive's own root: a share (\\server\share\) or a volume mounted
        # into a folder has no bin a file can go to.
        if not re.fullmatch(r"[A-Za-z]:\\", root.value):
            return False
        if kernel32.GetDriveTypeW(root.value) != cls._DRIVE_FIXED:
            return False  # removable, network, optical, RAM disk
        import winreg

        policy = cls._EXPLORER + r"\Policies\Explorer"
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            if cls._registry_int(hive, policy, "NoRecycleFiles") == 1:
                return False
        volume = ctypes.create_unicode_buffer(64)
        if kernel32.GetVolumeNameForVolumeMountPointW(root.value, volume, 64):
            guid = re.search(r"\{[0-9A-Fa-f-]+\}", volume.value)
            if guid:
                key = (
                    cls._EXPLORER + r"\Explorer\BitBucket\Volume" + "\\" + guid.group()
                )
                hive = winreg.HKEY_CURRENT_USER
                if cls._registry_int(hive, key, "NukeOnDelete") == 1:
                    return False
                capacity = cls._registry_int(hive, key, "MaxCapacity")  # MB
                try:
                    size = os.lstat(path).st_size
                except OSError:
                    return False
                if capacity is not None and size > capacity * 1024 * 1024:
                    return False
        return True

    @staticmethod
    def _registry_int(hive, subkey: str, name: str) -> Optional[int]:
        """The integer value *name* under *subkey*, or ``None`` when absent."""
        import winreg

        try:
            with winreg.OpenKey(hive, subkey) as key:
                return int(winreg.QueryValueEx(key, name)[0])
        except (OSError, TypeError, ValueError):
            return None

    @classmethod
    def _windows_bin_item(cls, path: str, since: float) -> Optional[str]:
        """The ``$R`` file the Recycle Bin keeps *path* as, found by its ``$I``
        record (the original path) among those written since *since*; ``None``
        when no record names it."""
        drive = os.path.splitdrive(path)[0]
        bin_root = os.path.join(drive + "\\", "$Recycle.Bin")
        wanted = os.path.normcase(path)
        try:
            owners = [e.path for e in os.scandir(bin_root) if e.is_dir()]
        except OSError:
            return None
        for owner in owners:  # one per user SID; only this user's list
            try:
                entries = list(os.scandir(owner))
            except OSError:
                continue
            for entry in entries:
                if not entry.name.upper().startswith("$I"):
                    continue
                try:
                    if entry.stat().st_mtime < since:
                        continue
                    original = cls._read_bin_record(entry.path)
                except OSError:
                    continue
                if original and os.path.normcase(original) == wanted:
                    return os.path.join(owner, "$R" + entry.name[2:])
        return None

    @staticmethod
    def _read_bin_record(record: str) -> str:
        """The original path a Recycle Bin ``$I`` record holds: version 1
        (Vista to 8, a fixed 260-character field) or 2 (Windows 10 on, a
        counted string); ``""`` for anything else."""
        with open(record, "rb") as fh:
            data = fh.read(4096)
        version = int.from_bytes(data[:8], "little")
        if version == 1:
            raw = data[24 : 24 + 520]
        elif version == 2 and len(data) >= 28:
            count = int.from_bytes(data[24:28], "little")
            raw = data[28 : 28 + 2 * count]
        else:
            return ""
        return raw.decode("utf-16-le", errors="replace").split("\0", 1)[0]

    # -- Linux and other freedesktop hosts -----------------------------------------

    @classmethod
    def _freedesktop(cls, path: str) -> Optional[str]:
        """The freedesktop.org trash (the Trash specification 1.0): the home
        trash when the file shares its device, else its volume's own
        ``.Trash/$uid`` or ``.Trash-$uid`` -- a rename, never a copy."""
        for trash in cls._freedesktop_candidates(path):
            moved = cls._into_freedesktop_trash(path, trash)
            if moved is not None:
                return moved
        return None

    @classmethod
    def _freedesktop_candidates(cls, path: str) -> List[str]:
        """The trash folders the spec allows for *path*, in order: the home
        trash when the file shares its device, else the volume's own."""
        device = os.lstat(path).st_dev
        data_home = os.environ.get("XDG_DATA_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "share"
        )
        home_trash = os.path.join(data_home, "Trash")
        if cls._device_of(home_trash) == device:
            return [home_trash]
        top = cls._mount_point(path)
        uid = getattr(os, "getuid", lambda: None)()
        if top is None or uid is None:
            return []
        shared = os.path.join(top, ".Trash")
        candidates: List[str] = []
        # A shared .Trash counts only as the spec says: a real folder, sticky.
        if (
            os.path.isdir(shared)
            and not os.path.islink(shared)
            and os.stat(shared).st_mode & 0o1000
        ):
            candidates.append(os.path.join(shared, str(uid)))
        candidates.append(os.path.join(top, f".Trash-{uid}"))
        return candidates

    @staticmethod
    def _device_of(folder: str) -> Optional[int]:
        """The device *folder* is (or would be, once made) on."""
        probe = os.path.abspath(folder)
        while True:
            try:
                return os.stat(probe).st_dev
            except OSError:
                parent = os.path.dirname(probe)
                if parent == probe:
                    return None
                probe = parent

    @staticmethod
    def _mount_point(path: str) -> Optional[str]:
        """The mount point *path*'s file system hangs from."""
        folder = os.path.dirname(os.path.abspath(path))
        while not os.path.ismount(folder):
            parent = os.path.dirname(folder)
            if parent == folder:
                return None
            folder = parent
        return folder

    @classmethod
    def _into_freedesktop_trash(cls, path: str, trash: str) -> Optional[str]:
        """Move *path* into the trash folder *trash*: its ``.trashinfo`` claimed
        first, under a free name, then the rename. ``None`` -- the file
        untouched -- when the trash cannot be made or written (a read-only
        volume, a folder this user does not own); a failed rename raises."""
        files, info = os.path.join(trash, "files"), os.path.join(trash, "info")
        stem, ext = os.path.splitext(os.path.basename(path))
        stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        body = f"[Trash Info]\nPath={quote(path)}\nDeletionDate={stamp}\n"
        try:
            for folder in (files, info):
                os.makedirs(folder, mode=0o700, exist_ok=True)
        except OSError:
            return None  # no trash this user can make here
        for n in range(10000):
            name = f"{stem}{ext}" if not n else f"{stem}.{n}{ext}"
            record = os.path.join(info, name + ".trashinfo")
            try:
                fd = os.open(record, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            except OSError:
                return None  # no trash this user can write here
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(body)
            target = os.path.join(files, name)
            if os.path.lexists(target):  # a stray file with no record: skip it
                os.remove(record)
                continue
            try:
                os.rename(path, target)
            except BaseException:
                os.remove(record)
                raise
            return target
        return None

    # -- macOS -------------------------------------------------------------------

    @classmethod
    def _macos(cls, path: str, trash_dirs: Optional[List[str]] = None) -> Optional[str]:
        """``~/.Trash``, or the volume's ``.Trashes/$uid``, when the file shares
        its device: a rename under a free name (Finder's own "name 2" rule).
        ``None`` where neither is there, or the system refuses the move."""
        device = os.lstat(path).st_dev
        if trash_dirs is None:
            trash_dirs = cls._macos_candidates(path)
        stem, ext = os.path.splitext(os.path.basename(path))
        for trash in trash_dirs:
            if not os.path.isdir(trash) or cls._device_of(trash) != device:
                continue
            for n in range(1, 10000):
                name = f"{stem}{ext}" if n == 1 else f"{stem} {n}{ext}"
                target = os.path.join(trash, name)
                if os.path.lexists(target):
                    continue
                try:
                    os.rename(path, target)
                except PermissionError:
                    break  # the system guards this trash: the next, or none
                return target
        return None

    @classmethod
    def _macos_candidates(cls, path: str) -> List[str]:
        """``~/.Trash``, then *path*'s volume's ``.Trashes/$uid``."""
        trash_dirs = [os.path.join(os.path.expanduser("~"), ".Trash")]
        top = cls._mount_point(path)
        if top is not None and hasattr(os, "getuid"):
            trash_dirs.append(os.path.join(top, ".Trashes", str(os.getuid())))
        return trash_dirs
