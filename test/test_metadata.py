#!/usr/bin/python
# coding=utf-8
"""Regression tests for pythontk.file_utils.metadata (Metadata / MetadataInternal)."""

import os
import sys
import types
import tempfile
import unittest
from unittest.mock import patch

try:
    import pythontk.file_utils  # noqa: F401
except ImportError:
    pass

from pythontk.file_utils.metadata import Metadata


class _FakeItem:
    """Stand-in for a Shell FolderItem."""

    def ExtendedProperty(self, key):
        return f"value::{key}"


class _FakeFolder:
    """Stand-in for a Shell Folder."""

    def ParseName(self, name):
        return _FakeItem()


class _FakeShell:
    """Faithful stand-in for Shell.Application.

    Mirrors the real behavior that makes the bug reachable: NameSpace('')
    (an empty/invalid path) returns None rather than a Folder.
    """

    def NameSpace(self, path):
        return _FakeFolder() if path else None


def _make_fake_win32com():
    win32com = types.ModuleType("win32com")
    client = types.ModuleType("win32com.client")
    client.Dispatch = lambda progid: _FakeShell()
    win32com.client = client
    return win32com, client


class _FakePropertySystem:
    """One file's property store behind stand-ins for every pywin32 module
    ``Metadata`` imports: ``win32com.client`` (the Shell read path) and
    ``win32com.propsys`` / ``win32com.shell`` / ``pythoncom`` (the write path).
    While ``refuse`` is set Commit raises, as a cloud placeholder's store does;
    a committed ``VT_EMPTY`` clears the property."""

    def __init__(self):
        self.values = {}
        self.refuse = False

    def modules(self):
        system = self

        class Store:
            def __init__(self):
                self.pending = {}

            def SetValue(self, pkey, value):
                self.pending[pkey] = value

            def Commit(self):
                if system.refuse:
                    raise OSError("the property store refused the commit")
                for pkey, value in self.pending.items():
                    if value is None:
                        system.values.pop(pkey, None)
                    else:
                        system.values[pkey] = value

        class Item:
            def ExtendedProperty(self, key):
                return system.values.get(key)

        class Folder:
            def ParseName(self, name):
                return Item()

        class Shell:
            def NameSpace(self, path):
                return Folder() if path else None

        def module(name, **attrs):
            mod = types.ModuleType(name)
            mod.__dict__.update(attrs)
            return mod

        propsys = module(
            "win32com.propsys.propsys",
            IID_IPropertyStore=object(),
            SHGetPropertyStoreFromParsingName=lambda *args: Store(),
            PSGetPropertyKeyFromName=lambda name: name,
            PROPVARIANTType=lambda value, vt: None,  # VT_EMPTY: the clear
        )
        shellcon = module("win32com.shell.shellcon", GPS_READWRITE=2)
        client = module("win32com.client", Dispatch=lambda progid: Shell())
        propsys_pkg = module("win32com.propsys", propsys=propsys)
        shell_pkg = module("win32com.shell", shellcon=shellcon)
        win32com = module(
            "win32com", client=client, propsys=propsys_pkg, shell=shell_pkg
        )
        return {
            "win32com": win32com,
            "win32com.client": client,
            "win32com.propsys": propsys_pkg,
            "win32com.propsys.propsys": propsys,
            "win32com.shell": shell_pkg,
            "win32com.shell.shellcon": shellcon,
            "pythoncom": module("pythoncom", VT_EMPTY=0),
        }


class TestMetadataBareFilenameWindows(unittest.TestCase):
    """Regression: Metadata._get must not crash on a bare/relative filename on Windows.

    fix_groups finding (metadata.py:105): os.path.dirname('report.txt') == '' ->
    shell.NameSpace('') returns None -> folder.ParseName(...) raised AttributeError.
    The fix normalizes to an absolute path (and guards folder/item for None).
    """

    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._tmp = tempfile.mkdtemp()
        os.chdir(self._tmp)
        # Inject a faithful fake win32com so the nt branch runs without pywin32.
        self._win32com, self._client = _make_fake_win32com()
        self._saved = {k: sys.modules.get(k) for k in ("win32com", "win32com.client")}
        sys.modules["win32com"] = self._win32com
        sys.modules["win32com.client"] = self._client

    def tearDown(self):
        os.chdir(self._orig_cwd)
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        try:
            for name in os.listdir(self._tmp):
                os.remove(os.path.join(self._tmp, name))
            os.rmdir(self._tmp)
        except OSError:
            pass

    @patch.object(os, "name", "nt")
    def test_get_bare_filename_does_not_crash(self):
        # A bare filename present in cwd: dirname == '' triggered the crash.
        with open("report.txt", "w") as f:
            f.write("x")

        # Before the fix this raised AttributeError (NameSpace('') -> None);
        # after, abspath makes the folder resolvable and the property returns.
        result = Metadata.get("report.txt", "Title")

        self.assertEqual(result, {"Title": "value::Title"})


@unittest.skipIf(os.name == "nt", "POSIX: extended attributes")
class TestMetadataXattrRefusedPosix(unittest.TestCase):
    """NFS, exFAT and many FUSE/cloud mounts refuse xattrs. The refusal was
    printed and the metadata silently lost; it now takes the sidecar when that
    is enabled -- as a failed Windows property store does -- and raises
    otherwise."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = os.path.join(folder.name, "scene.ma")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("x")
        self.refused = OSError(95, "Operation not supported")  # ENOTSUP

    def test_a_refused_key_goes_to_the_sidecar_and_reads_back(self):
        with (
            patch.object(Metadata, "enable_sidecar", True),
            patch("os.setxattr", side_effect=self.refused),
        ):
            Metadata.set(self.path, Comments="kept")
            self.assertEqual(Metadata.get(self.path, "Comments"), {"Comments": "kept"})

    def test_without_the_sidecar_a_refusal_raises(self):
        with (
            patch.object(Metadata, "enable_sidecar", False),
            patch("os.setxattr", side_effect=self.refused),
        ):
            with self.assertRaisesRegex(RuntimeError, "enable_sidecar"):
                Metadata.set(self.path, Comments="lost")

    def test_a_cleared_key_leaves_the_sidecar_too(self):
        """A clear (None) removed the xattr -- which such a mount refuses as
        well -- but never the copy a refused set had parked in the sidecar, so
        every read still returned the old value. A clear with nothing parked
        writes no sidecar."""
        sidecar = Metadata._get_sidecar_path(self.path)
        with (
            patch.object(Metadata, "enable_sidecar", True),
            patch("os.setxattr", side_effect=self.refused),
            patch("os.removexattr", side_effect=self.refused),
            patch("os.getxattr", side_effect=self.refused),
        ):
            Metadata.set(self.path, Comments=None)
            self.assertFalse(os.path.exists(sidecar))
            Metadata.set(self.path, Comments="old", Title="kept")
            Metadata.set(self.path, Comments=None)
            self.assertEqual(
                Metadata.get(self.path, "Comments", "Title"),
                {"Comments": None, "Title": "kept"},
            )

    def test_a_value_the_mount_takes_retires_the_parked_copy(self):
        """The sidecar overlays every read, so a copy parked there by a
        refusal must not outlive a value the xattrs later hold (the file
        moved, or the mount changed) -- it read back instead of it."""
        xattrs = {}

        def getxattr(path, key):
            try:
                return xattrs[key]
            except KeyError:
                raise OSError(61, "No data available") from None  # ENODATA

        def setxattr(path, key, value):
            xattrs[key] = value

        with patch.object(Metadata, "enable_sidecar", True):
            with patch("os.setxattr", side_effect=self.refused):
                Metadata.set(self.path, Comments="old")
            with (
                patch("os.setxattr", side_effect=setxattr),
                patch("os.getxattr", side_effect=getxattr),
            ):
                Metadata.set(self.path, Comments="new")
                self.assertEqual(
                    Metadata.get(self.path, "Comments"), {"Comments": "new"}
                )


@unittest.skipUnless(sys.platform == "win32", "the sidecar is hidden via kernel32")
class TestMetadataSidecarWindows(unittest.TestCase):
    """The POSIX case's Windows sibling: a commit the property store refused
    parks the value in the sidecar, which overlays every read -- so a later
    clear, or value, the store DID take left the parked copy winning."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = os.path.join(folder.name, "scene.ma")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("x")
        self.system = _FakePropertySystem()
        # Only the injected names are restored: patch.dict(sys.modules) would
        # also drop every module first imported during the test.
        fakes = self.system.modules()
        saved = {name: sys.modules.get(name) for name in fakes}
        sys.modules.update(fakes)
        self.addCleanup(self._restore_modules, saved)
        patcher = patch.object(Metadata, "enable_sidecar", True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.system.refuse = True
        Metadata.set(self.path, PtkNote="old")  # parked in the sidecar
        self.assertEqual(Metadata.get(self.path, "PtkNote"), {"PtkNote": "old"})
        self.system.refuse = False

    @staticmethod
    def _restore_modules(saved):
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_a_clear_the_store_takes_retires_the_parked_copy(self):
        Metadata.set(self.path, PtkNote=None)
        self.assertEqual(Metadata.get(self.path, "PtkNote"), {"PtkNote": None})

    def test_a_value_the_store_takes_retires_the_parked_copy(self):
        Metadata.set(self.path, PtkNote="new")
        self.assertEqual(Metadata.get(self.path, "PtkNote"), {"PtkNote": "new"})


if __name__ == "__main__":
    unittest.main()
