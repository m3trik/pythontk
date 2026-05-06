#!/usr/bin/python
# coding=utf-8
"""
Tests for FileUtils.move_file and FileUtils.copy_file.

These are entry points consumed by mayatk.MatUpdater's transfer logic, so
their behavior under all flag combinations and error paths must be locked
down.
"""
import os
import shutil
import tempfile
import unittest

from pythontk import FileUtils

from conftest import BaseTestCase


class MoveFileTest(BaseTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ptk_move_")
        self.src_dir = os.path.join(self.tmp, "src")
        self.dst_dir = os.path.join(self.tmp, "dst")
        os.makedirs(self.src_dir, exist_ok=True)
        os.makedirs(self.dst_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _src(self, name: str, content: str = "x") -> str:
        path = os.path.join(self.src_dir, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_move_single_file_returns_string(self):
        src = self._src("a.txt")
        out = FileUtils.move_file(src, self.dst_dir)
        self.assertIsInstance(out, str)
        self.assertTrue(os.path.isfile(out))
        self.assertFalse(os.path.exists(src))
        self.assertEqual(os.path.basename(out), "a.txt")

    def test_move_list_returns_list(self):
        srcs = [self._src("a.txt"), self._src("b.txt")]
        out = FileUtils.move_file(srcs, self.dst_dir)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        for p in out:
            self.assertTrue(os.path.isfile(p))
        for p in srcs:
            self.assertFalse(os.path.exists(p))

    def test_move_with_new_name_single(self):
        src = self._src("a.txt")
        out = FileUtils.move_file(src, self.dst_dir, new_name="renamed.txt")
        self.assertEqual(os.path.basename(out), "renamed.txt")
        self.assertTrue(os.path.isfile(out))

    def test_move_creates_destination(self):
        src = self._src("a.txt")
        new_dst = os.path.join(self.tmp, "newdst")
        self.assertFalse(os.path.exists(new_dst))
        FileUtils.move_file(src, new_dst, create_dir=True)
        self.assertTrue(os.path.isfile(os.path.join(new_dst, "a.txt")))

    def test_move_overwrite_replaces_existing(self):
        src = self._src("a.txt", "new content")
        existing = os.path.join(self.dst_dir, "a.txt")
        with open(existing, "w") as f:
            f.write("old")
        out = FileUtils.move_file(src, self.dst_dir, overwrite=True)
        with open(out) as f:
            self.assertEqual(f.read(), "new content")

    def test_move_no_overwrite_raises(self):
        src = self._src("a.txt")
        with open(os.path.join(self.dst_dir, "a.txt"), "w") as f:
            f.write("old")
        with self.assertRaises(FileExistsError):
            FileUtils.move_file(src, self.dst_dir, overwrite=False)

    def test_move_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            FileUtils.move_file(
                os.path.join(self.src_dir, "nope.txt"), self.dst_dir
            )

    def test_move_tuple_form(self):
        """List entries can be (dir, filename) tuples."""
        self._src("c.txt")
        out = FileUtils.move_file(
            [(self.src_dir, "c.txt")], self.dst_dir
        )
        self.assertIsInstance(out, list)
        self.assertTrue(os.path.isfile(out[0]))

    def test_move_returns_forward_slashes(self):
        src = self._src("a.txt")
        out = FileUtils.move_file(src, self.dst_dir)
        self.assertNotIn("\\", out)


class CopyFileTest(BaseTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ptk_copy_")
        self.src_dir = os.path.join(self.tmp, "src")
        self.dst_dir = os.path.join(self.tmp, "dst")
        os.makedirs(self.src_dir, exist_ok=True)
        os.makedirs(self.dst_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _src(self, name: str, content: str = "x") -> str:
        path = os.path.join(self.src_dir, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_copy_basic(self):
        src = self._src("a.txt", "hello")
        out = FileUtils.copy_file(src, self.dst_dir)
        self.assertTrue(os.path.isfile(out))
        self.assertTrue(os.path.exists(src), "Source must remain after copy")
        with open(out) as f:
            self.assertEqual(f.read(), "hello")

    def test_copy_with_new_name(self):
        src = self._src("a.txt")
        out = FileUtils.copy_file(src, self.dst_dir, new_name="renamed.txt")
        self.assertEqual(os.path.basename(out), "renamed.txt")

    def test_copy_creates_destination(self):
        src = self._src("a.txt")
        new_dst = os.path.join(self.tmp, "deep", "nested")
        FileUtils.copy_file(src, new_dst, create_dir=True)
        self.assertTrue(os.path.isfile(os.path.join(new_dst, "a.txt")))

    def test_copy_overwrite_replaces(self):
        src = self._src("a.txt", "new")
        existing = os.path.join(self.dst_dir, "a.txt")
        with open(existing, "w") as f:
            f.write("old")
        out = FileUtils.copy_file(src, self.dst_dir, overwrite=True)
        with open(out) as f:
            self.assertEqual(f.read(), "new")

    def test_copy_no_overwrite_raises(self):
        src = self._src("a.txt")
        with open(os.path.join(self.dst_dir, "a.txt"), "w") as f:
            f.write("old")
        with self.assertRaises(FileExistsError):
            FileUtils.copy_file(src, self.dst_dir, overwrite=False)

    def test_copy_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            FileUtils.copy_file(
                os.path.join(self.src_dir, "nope.txt"), self.dst_dir
            )


if __name__ == "__main__":
    unittest.main()
