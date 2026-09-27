# !/usr/bin/python
# coding=utf-8
"""Unit tests for FileUtils.atomic_write_text."""

import os
import stat
import tempfile
import unittest
from unittest.mock import patch

from pythontk import FileUtils


class AtomicWriteText(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.path = os.path.join(self.dir, "target.txt")

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        FileUtils.atomic_write_text(self.path, "hello world")
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "hello world")

    def test_no_temp_left_on_success(self):
        FileUtils.atomic_write_text(self.path, "ok")
        self.assertEqual(os.listdir(self.dir), [os.path.basename(self.path)])

    def test_overwrites_existing(self):
        with open(self.path, "w") as f:
            f.write("original")
        FileUtils.atomic_write_text(self.path, "replaced")
        with open(self.path) as f:
            self.assertEqual(f.read(), "replaced")

    def test_unicode_round_trip(self):
        text = "héllo — 世界"
        FileUtils.atomic_write_text(self.path, text)
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(f.read(), text)

    def test_failure_does_not_modify_target(self):
        with open(self.path, "w") as f:
            f.write("untouched")

        # Patch os.replace to raise mid-operation. The temp file will exist
        # at the moment of the patched call; we then assert the target was
        # not modified and the temp was cleaned up.
        with patch("os.replace", side_effect=OSError("simulated")):
            with self.assertRaises(OSError):
                FileUtils.atomic_write_text(self.path, "should fail")

        with open(self.path) as f:
            self.assertEqual(f.read(), "untouched")

        # Temp files should be cleaned up after failure
        leftover = [n for n in os.listdir(self.dir) if n != os.path.basename(self.path)]
        self.assertEqual(leftover, [])

    @unittest.skipUnless(os.name == "nt", "Windows lock codes")
    def test_a_momentarily_held_target_is_still_replaced(self):
        # The live failure: a synced drive's client held a just-rewritten
        # sidecar and os.replace raised [WinError 5] on the next rewrite.
        from test_file import ReplaceFileTest

        with open(self.path, "w") as f:
            f.write("old")
        replace, calls = ReplaceFileTest.held(2, winerror=5)
        with patch("os.replace", side_effect=replace), patch("time.sleep"):
            FileUtils.atomic_write_text(self.path, "new")
        with open(self.path) as f:
            self.assertEqual(f.read(), "new")
        self.assertEqual(len(calls), 3)

    def test_creates_target_when_absent(self):
        path = os.path.join(self.dir, "new.txt")
        self.assertFalse(os.path.exists(path))
        FileUtils.atomic_write_text(path, "fresh")
        self.assertTrue(os.path.isfile(path))

    def test_newlines_are_the_platforms(self):
        # Text mode, as the writer always was: "\n" lands as os.linesep.
        FileUtils.atomic_write_text(self.path, "a\nb")
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), b"a" + os.linesep.encode() + b"b")

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_a_new_file_gets_the_umask_mode_not_owner_only(self):
        # tempfile creates 0600 and os.replace keeps the temp's mode, so every
        # rewrite of a shared file (the scene-data sidecar beside a delivery)
        # left it readable by its owner alone.
        umask = os.umask(0)
        os.umask(umask)
        FileUtils.atomic_write_text(self.path, "x")
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o666 & ~umask)

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_a_rewrite_keeps_the_files_mode(self):
        with open(self.path, "w") as f:
            f.write("old")
        os.chmod(self.path, 0o640)
        FileUtils.atomic_write_text(self.path, "new")
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o640)


if __name__ == "__main__":
    unittest.main()
