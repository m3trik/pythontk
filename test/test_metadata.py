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


if __name__ == "__main__":
    unittest.main()
