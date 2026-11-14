#!/usr/bin/python
# coding=utf-8
"""Tests for ``pythontk.PendingFiles`` (``file_utils/pending_files.py``).

Files written for the open scene wait in a session folder until it is saved:
the home folder -- what the scene file on disk reads -- is untouched until
then, and an unsaved session leaves nothing in it.
"""

import os
import unittest

import pythontk as ptk
from pythontk import FileDependencies, PendingFiles

from conftest import BaseTestCase


class TestPendingFiles(BaseTestCase):
    def setUp(self):
        self.store = ptk.TempArtifacts("ptk_pending_files")
        self.addCleanup(self.store.cleanup)
        self.home = self.store.dir_path()
        self.pending = PendingFiles(prefix="ptk_pending_files_session")
        self.addCleanup(self.pending.discard)

    def _write(self, path, data):
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    def _read(self, path):
        with open(path, "rb") as fh:
            return fh.read()

    def test_a_staged_write_leaves_the_home_file_as_the_saved_scene_reads_it(self):
        home = self._write(os.path.join(self.home, "Room_Lightmap.exr"), b"old")
        staged = self.pending.stage(home)
        self._write(staged, b"new")
        self.assertEqual(os.path.basename(staged), "Room_Lightmap.exr")
        self.assertNotEqual(os.path.dirname(staged), self.home)
        self.assertEqual(self._read(home), b"old")
        self.assertTrue(self.pending.holds(staged))
        self.assertEqual(os.path.normcase(self.pending.home(staged)), os.path.normcase(home))
        self.assertEqual(self.pending.stage(staged), staged)  # its own

    def test_discarding_an_unsaved_session_leaves_no_file_behind(self):
        home = os.path.join(self.home, "Room_Lightmap.exr")
        staged = self._write(self.pending.stage(home), b"new")
        self.pending.retire([self._write(os.path.join(self.home, "Old.exr"), b"x")])
        self.pending.discard()
        self.assertFalse(os.path.exists(staged))
        self.assertFalse(os.path.exists(home))
        self.assertTrue(os.path.exists(os.path.join(self.home, "Old.exr")))
        self.assertFalse(self.pending)

    def test_a_save_moves_the_kept_files_home_and_drops_the_rest(self):
        home = self._write(os.path.join(self.home, "A.exr"), b"old")
        kept = self._write(self.pending.stage(home), b"new")
        dropped = self._write(self.pending.stage(os.path.join(self.home, "B.exr")), b"undone")
        moved = {}
        self.pending.on_save("maps", lambda _t, _f: moved.update(self.pending.commit([kept])))
        self.pending.saving(os.path.join(self.home, "scene.ma"))
        self.assertEqual(list(moved), [kept])
        self.assertEqual(self._read(home), b"new")
        self.assertFalse(os.path.exists(os.path.join(self.home, "B.exr")))
        self.assertFalse(os.path.exists(dropped))
        self.assertFalse(self.pending)

    def test_each_tools_hook_keeps_its_own_files(self):
        """One tool's commit never drops another's: the save drops what no
        hook kept, once all have run."""
        a = self._write(self.pending.stage(os.path.join(self.home, "A.exr")), b"a")
        b = self._write(self.pending.stage(os.path.join(self.home, "B.exr")), b"b")
        self.pending.on_save("first", lambda _t, _f: self.pending.commit([a]))
        self.pending.on_save("second", lambda _t, _f: self.pending.commit([b]))
        self.pending.saving(os.path.join(self.home, "scene.ma"))
        self.assertEqual(self._read(os.path.join(self.home, "A.exr")), b"a")
        self.assertEqual(self._read(os.path.join(self.home, "B.exr")), b"b")

    def test_a_rename_moves_a_file_to_another_home_and_the_old_one_stays(self):
        home = self._write(os.path.join(self.home, "A.exr"), b"old")
        staged = self._write(self.pending.stage(home), b"new")
        forked = os.path.join(self.home, "A_1.exr")
        moved = self.pending.commit([staged], rename=lambda _path: forked)
        self.assertEqual(moved, {staged: forked})
        self.assertEqual(self._read(home), b"old")
        self.assertEqual(self._read(forked), b"new")

    def test_a_file_that_cannot_be_written_home_stays_staged(self):
        staged = self._write(self.pending.stage(os.path.join(self.home, "A.exr")), b"new")
        blocker = os.path.join(self.home, "A.exr")
        os.makedirs(blocker)  # a folder where the file goes: the write fails
        moved = {}
        self.pending.on_save("maps", lambda _t, _f: moved.update(self.pending.commit([staged])))
        self.pending.saving(os.path.join(self.home, "scene.ma"))
        self.assertEqual(moved, {})
        self.assertTrue(os.path.isfile(staged))
        self.assertTrue(self.pending)

    def test_saving_tells_each_hook_whether_the_save_forks_the_scene(self):
        seen = []
        self.pending.on_save("maps", lambda target, fork: seen.append((target, fork)))
        scene = self._write(os.path.join(self.home, "scene.ma"), b"//Maya ASCII")
        copy = os.path.join(self.home, "scene_v2.ma")
        gone = os.path.join(self.home, "renamed_away.ma")
        for target, saved_from in ((scene, ""), (scene, scene), (copy, scene), (copy, gone)):
            # Something waits each time (a save drops what no hook kept).
            self._write(self.pending.stage(os.path.join(self.home, "A.exr")), b"new")
            self.pending.saving(target, saved_from=saved_from)
        self.assertEqual(
            [f for _t, f in seen],
            [False, False, True, False],
            "first save, save, Save As, a save of a renamed scene",
        )

    def test_a_hook_that_raises_never_fails_the_save(self):
        self._write(self.pending.stage(os.path.join(self.home, "A.exr")), b"new")

        def broken(_target, _fork):
            raise RuntimeError("boom")

        self.pending.on_save("maps", broken)
        self.pending.saving("C:/p/scene.ma")  # no raise

    def test_a_hook_that_raises_leaves_every_staged_file_for_the_next_save(self):
        """REGRESSION (2026-10-24): a hook that raised before its commit left
        its files staged -- the saved scene reading them from the session
        folder -- and the sweep then deleted them, so the scene opened with
        the map in neither place. The ledger cannot tell whose files a failed
        hook held, so after one the save drops nothing."""
        home = os.path.join(self.home, "A.exr")
        staged = self._write(self.pending.stage(home), b"new")
        other = self._write(self.pending.stage(os.path.join(self.home, "B.exr")), b"b")

        def broken(_target, _fork):
            raise RuntimeError("boom")

        self.pending.on_save("maps", broken)
        self.pending.on_save("other", lambda _t, _f: self.pending.commit([other]))
        self.pending.saving(os.path.join(self.home, "scene.ma"))
        self.assertTrue(os.path.isfile(staged), "the staged file survives")
        self.assertFalse(os.path.exists(home))
        self.assertEqual(self._read(os.path.join(self.home, "B.exr")), b"b")
        self.assertTrue(self.pending, "the next save tries again")

    def test_retirements_wait_for_the_save_and_are_handed_over_once(self):
        old = self._write(os.path.join(self.home, "Old.exr"), b"x")
        self.pending.retire([old, old])
        self.assertEqual(self.pending.take_retiring(), [os.path.abspath(old)])
        self.assertEqual(self.pending.take_retiring(), [])


class TestSuperseded(BaseTestCase):
    def setUp(self):
        self.store = ptk.TempArtifacts("ptk_pending_files")
        self.addCleanup(self.store.cleanup)
        self.tmp = self.store.dir_path()

    def test_superseded_names_what_remove_superseded_would_take_and_takes_nothing(self):
        old = os.path.join(self.tmp, "Old.exr")
        kept = os.path.join(self.tmp, "Kept.exr")
        for path in (old, kept):
            with open(path, "wb") as fh:
                fh.write(path.encode())
        after = [("obj", "Kept.exr", kept)]
        self.assertEqual(FileDependencies.superseded([old, kept], after), [old])
        self.assertTrue(os.path.exists(old))


if __name__ == "__main__":
    unittest.main()
