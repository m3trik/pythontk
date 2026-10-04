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
import time
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

    def test_a_failed_overwrite_keeps_both_files(self):
        """Overwriting deleted the destination BEFORE moving the source, so a
        move that failed (a full disk, a locked file) lost the old file and
        left the new one wherever it was -- the delete-then-move the lightmap
        baker documents losing a finished map to. Pinned across volumes (the
        rename refused with EXDEV), where the move is a copy a full disk can
        fail."""
        import errno
        from unittest import mock

        src = self._src("a.txt", "new content")
        existing = os.path.join(self.dst_dir, "a.txt")
        with open(existing, "w") as f:
            f.write("old")
        with mock.patch(
            "os.replace", side_effect=OSError(errno.EXDEV, "cross-device link")
        ), mock.patch("shutil.move", side_effect=OSError(28, "No space left")):
            with self.assertRaises(OSError):
                FileUtils.move_file(src, self.dst_dir, overwrite=True)
        with open(existing) as f:
            self.assertEqual(f.read(), "old")
        self.assertTrue(os.path.isfile(src))

    def test_a_refused_swap_puts_the_source_back(self):
        """The destination held open (Windows refuses the rename): the old file
        stays, and the source is back where the caller can retry it under
        another name."""
        from unittest import mock

        src = self._src("a.txt", "new content")
        existing = os.path.join(self.dst_dir, "a.txt")
        with open(existing, "w") as f:
            f.write("old")
        with mock.patch.object(
            FileUtils, "replace_file", side_effect=PermissionError(13, "held open")
        ):
            with self.assertRaises(PermissionError):
                FileUtils.move_file(src, self.dst_dir, overwrite=True)
        with open(existing) as f:
            self.assertEqual(f.read(), "old")
        with open(src) as f:
            self.assertEqual(f.read(), "new content")
        self.assertEqual(sorted(os.listdir(self.dst_dir)), ["a.txt"])

    def test_a_destination_near_the_name_limit_still_takes_an_overwrite(self):
        """The stage was the destination's name plus ~23 characters, so a
        destination near a limit -- a name's 255 characters, or a path's 260
        in a process that is not long-path aware (Maya) -- staged under a name
        that cannot exist, and every overwrite into it failed: a soldering
        table's lightmap atlas, 238 characters under its texture folder, was
        never placed and the bake lost it (2026-10-03). The stage is a
        fixed-length sibling. Measured on the name limit, which every OS has."""
        name = "n" * 246 + ".txt"
        src = self._src("a.txt", "new content")
        existing = os.path.join(self.dst_dir, name)
        try:
            with open(existing, "w") as f:
                f.write("old")
        except OSError:
            self.skipTest("this process cannot hold a 250-character name here")
        out = FileUtils.move_file(src, self.dst_dir, new_name=name, overwrite=True)
        with open(out) as f:
            self.assertEqual(f.read(), "new content")
        self.assertEqual(os.listdir(self.dst_dir), [name])

    def test_a_same_volume_overwrite_stages_nothing(self):
        """The fixed-length stage (``.<12 hex>.moving``, 20 characters) is
        LONGER than a short destination name, so beside a destination near a
        path's 260 characters in a process that is not long-path aware (Maya)
        it could not exist and the overwrite failed (2026-10-04 review). On
        one volume nothing needs staging: one rename replaces the destination
        atomically, and a refused one leaves the source where it was."""
        import errno
        from unittest import mock

        src = self._src("a.txt", "new content")
        existing = os.path.join(self.dst_dir, "a.txt")
        with open(existing, "w") as f:
            f.write("old")
        limit = len(os.path.abspath(existing))
        real_move, real_replace = shutil.move, FileUtils.replace_file
        swaps = []

        def capped(source, target, *args, **kwargs):
            # The path limit, at the destination's length: nothing longer
            # can exist in this folder.
            if len(os.path.abspath(target)) > limit:
                raise OSError(errno.ENAMETOOLONG, "path too long", target)
            return real_move(source, target, *args, **kwargs)

        def recorded(source, target):
            swaps.append((source, target))
            real_replace(source, target)

        with mock.patch("shutil.move", side_effect=capped), mock.patch.object(
            FileUtils, "replace_file", side_effect=recorded
        ):
            FileUtils.move_file(src, self.dst_dir, overwrite=True)
        with open(existing) as f:
            self.assertEqual(f.read(), "new content")
        self.assertFalse(os.path.exists(src))
        self.assertEqual(os.listdir(self.dst_dir), ["a.txt"])
        self.assertEqual(
            [os.path.normcase(os.path.abspath(p)) for swap in swaps for p in swap],
            [os.path.normcase(os.path.abspath(p)) for p in (src, existing)],
            "one rename, from the source itself: no .moving stage",
        )

    @staticmethod
    def _age(path: str, days: float = 30.0) -> None:
        """Backdate *path* past TempArtifacts' stale-sweep age gate."""
        stamp = time.time() - days * 86400
        os.utime(path, (stamp, stamp))

    def test_a_sibling_sweep_never_takes_a_staged_move(self):
        """An overwrite stages the source beside the destination, and a moved
        file keeps its source's mtime. Staged as an ``atomic_write_*`` temp, a
        week-old source read as stale to the sweep any concurrent
        ``FileUtils.atomic_write`` into the folder runs first, and was deleted
        between the stage and the swap."""
        from unittest import mock

        src = self._src("a.txt", "new content")
        self._age(src)
        existing = os.path.join(self.dst_dir, "a.txt")
        with open(existing, "w") as f:
            f.write("old")
        real_replace = FileUtils.replace_file
        interposed = []

        def sibling_write_then_swap(staged, dst):
            if not interposed:  # the move's swap; the sibling's own promote passes
                interposed.append(staged)
                FileUtils.atomic_write(
                    os.path.join(self.dst_dir, "other.bin"),
                    lambda part: open(part, "wb").close(),
                )
            real_replace(staged, dst)

        with mock.patch.object(
            FileUtils, "replace_file", side_effect=sibling_write_then_swap
        ):
            FileUtils.move_file(src, self.dst_dir, overwrite=True)
        with open(existing) as f:
            self.assertEqual(f.read(), "new content")
        self.assertFalse(os.path.exists(src))
        self.assertEqual(sorted(os.listdir(self.dst_dir)), ["a.txt", "other.bin"])

    def test_an_interrupted_directory_move_keeps_the_whole_copy(self):
        """Across volumes a directory is copied, then its source removed. A
        removal that fails part way leaves the source PARTIAL, so the stage
        holds the only whole copy: it must survive, and the log must say where.
        (A file's unlink is all or nothing, so a file's stage may be dropped.)"""
        import errno
        from unittest import mock

        src = os.path.join(self.src_dir, "pkg")
        os.makedirs(src)
        for name in ("a.txt", "b.txt"):
            with open(os.path.join(src, name), "w") as f:
                f.write(name)
        existing = os.path.join(self.dst_dir, "pkg")
        with open(existing, "w") as f:
            f.write("old")

        def cross_volume(*_args, **_kwargs):
            raise OSError(errno.EXDEV, "cross-device link")

        def partial_rmtree(path, *_args, **_kwargs):
            os.remove(os.path.join(path, "a.txt"))
            raise PermissionError(13, "held open")

        with mock.patch("os.rename", side_effect=cross_volume), mock.patch(
            "os.replace", side_effect=cross_volume
        ), mock.patch(
            "shutil.rmtree", side_effect=partial_rmtree
        ), self.assertLogs("pythontk.file_utils._file_utils", "ERROR") as logs:
            with self.assertRaises(PermissionError):
                FileUtils.move_file(src, self.dst_dir, overwrite=True)
        stages = [n for n in os.listdir(self.dst_dir) if n.endswith(".moving")]
        self.assertEqual(len(stages), 1, os.listdir(self.dst_dir))
        stage = os.path.join(self.dst_dir, stages[0])
        self.assertEqual(sorted(os.listdir(stage)), ["a.txt", "b.txt"])
        self.assertIn(stages[0], "".join(logs.output))
        with open(existing) as f:
            self.assertEqual(f.read(), "old")

    def test_concurrent_overwrites_into_one_folder_lose_nothing(self):
        """The shape that lost textures: a thread pool archiving week-old files
        into one folder (``MapOptimizer.optimize_maps``), each move's staging
        sweep deleting the others' stages."""
        from concurrent.futures import ThreadPoolExecutor

        names = [f"t{i:02d}.txt" for i in range(32)]
        for name in names:
            self._age(self._src(name, name))
            with open(os.path.join(self.dst_dir, name), "w") as f:
                f.write("old")
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda name: FileUtils.move_file(
                        os.path.join(self.src_dir, name), self.dst_dir
                    ),
                    names,
                )
            )
        for name in names:
            with open(os.path.join(self.dst_dir, name)) as f:
                self.assertEqual(f.read(), name)
        self.assertEqual(sorted(os.listdir(self.dst_dir)), names)

    def test_move_no_overwrite_raises(self):
        src = self._src("a.txt")
        with open(os.path.join(self.dst_dir, "a.txt"), "w") as f:
            f.write("old")
        with self.assertRaises(FileExistsError):
            FileUtils.move_file(src, self.dst_dir, overwrite=False)

    def test_move_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            FileUtils.move_file(os.path.join(self.src_dir, "nope.txt"), self.dst_dir)

    def test_move_tuple_form(self):
        """List entries can be (dir, filename) tuples."""
        self._src("c.txt")
        out = FileUtils.move_file([(self.src_dir, "c.txt")], self.dst_dir)
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
            FileUtils.copy_file(os.path.join(self.src_dir, "nope.txt"), self.dst_dir)


if __name__ == "__main__":
    unittest.main()
