#!/usr/bin/python
# coding=utf-8
"""Tests for ``pythontk.FileDependencies`` (``file_utils/file_dependencies.py``).

Files a record names by name plus the folder it was written from, found where
they are NOW. The rules take plain ``(owner, name, recorded folder)``
references, so real files in a scratch folder are all a test needs -- no host.
"""

import os
import unittest
import unittest.mock

import pythontk as ptk
from pythontk import FileDependencies

from conftest import BaseTestCase


class _FilesCase(BaseTestCase):
    """A scratch folder per test."""

    def setUp(self):
        self.store = ptk.TempArtifacts("ptk_file_dependencies")
        self.addCleanup(self.store.cleanup)
        self.tmp = self.store.dir_path()

    def _file(self, *parts, data=b"x"):
        path = os.path.join(self.tmp, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    def _dir(self, *parts):
        path = os.path.join(self.tmp, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _same(a, b):
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(
            os.path.abspath(b)
        )


class TestClaims(unittest.TestCase):
    def test_claims_name_every_reader_of_every_file(self):
        claims = FileDependencies.claims(
            [
                ("|a", "Crate_Lightmap.exr", "x"),
                ("|b", "some/where/crate_lightmap.EXR"),  # a path; any case
                ("|c", "Floor_Lightmap.exr", "x"),
                ("|d", "", "x"),  # names no file: not a reader
            ]
        )
        self.assertEqual(
            claims,
            {
                "crate_lightmap.exr": frozenset({"|a", "|b"}),
                "floor_lightmap.exr": frozenset({"|c"}),
            },
        )

    def test_claims_feed_the_naming_rule(self):
        """What the two are for together: a batch never lands on a file
        something else reads, and keeps its own."""
        claims = FileDependencies.claims([("|a", "Crate.exr"), ("|b", "Crate_1.exr")])
        mine = ptk.FileUtils.unique_path(
            "out", "Crate", ".exr", claims=claims, owners=["|b"]
        )
        self.assertEqual(os.path.basename(mine), "Crate_1.exr")


class TestRemoveSuperseded(_FilesCase):
    """What a re-bake may set aside: the files its owners read before, once
    nothing reads them. Every keep rule is a reader the move would strand.

    Pinned to a volume with no trash, so each file goes into the
    ``superseded`` temp store (the run's sandbox root), not into this
    machine's Recycle Bin (:class:`TestSetAside` covers the trash branch).
    """

    def setUp(self):
        super().setUp()
        patcher = unittest.mock.patch.object(
            ptk.FileUtils, "move_to_trash", return_value=None
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _aside(self, path):
        import glob

        found = glob.glob(
            os.path.join(
                ptk.TempArtifacts.root(),
                FileDependencies.SUPERSEDED_STORE,
                "*",
                os.path.basename(path),
            )
        )
        return max(found, key=os.path.getmtime)

    def test_a_file_nothing_reads_is_set_aside_and_a_read_one_kept(self):
        old = self._file("old", "Crate_Lightmap.exr", data=b"old bake")
        shared = self._file("old", "Floor_Lightmap.exr")
        new = self._file("new", "Crate_Lightmap.exr")
        after = [
            ("|crate", "Crate_Lightmap.exr", new),  # moved to the new folder
            ("|floor", "Floor_Lightmap.exr", shared),  # still reads its file
        ]

        removed = FileDependencies.remove_superseded([old, shared, new], after)

        self.assertEqual(removed, [old])
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(shared))
        self.assertTrue(os.path.exists(new))
        with open(self._aside(old), "rb") as fh:
            self.assertEqual(fh.read(), b"old bake", "set aside, never deleted")

    def test_a_name_found_nowhere_keeps_every_file_of_that_name(self):
        """A reader whose file resolved nowhere may still find this one by a
        search the host did not run -- a walk of its texture tree."""
        old = self._file("old", "Crate_Lightmap.exr")
        after = [("|crate", "CRATE_lightmap.exr", None)]
        self.assertEqual(FileDependencies.remove_superseded([old], after), [])
        self.assertTrue(os.path.exists(old))

    def test_each_file_is_set_aside_once_however_it_is_spelled(self):
        old = self._file("old", "Crate_Lightmap.exr")
        spelled = os.path.join(self.tmp, "old", ".", "Crate_Lightmap.exr")
        removed = FileDependencies.remove_superseded([old, spelled], [])
        self.assertEqual(removed, [old])

    def test_a_file_already_gone_or_held_open_is_not_reported_set_aside(self):
        gone = os.path.join(self.tmp, "old", "Gone_Lightmap.exr")
        held = self._file("old", "Held_Lightmap.exr")

        def trash(path):
            if os.path.basename(path) == "Held_Lightmap.exr":
                raise PermissionError(13, "held open", path)
            raise FileNotFoundError(2, "No such file", path)

        with unittest.mock.patch.object(
            ptk.FileUtils, "move_to_trash", side_effect=trash
        ):
            with self.assertLogs(FileDependencies.logger, level="WARNING") as caught:
                removed = FileDependencies.remove_superseded([gone, held], [])

        self.assertEqual(removed, [])
        self.assertTrue(os.path.exists(held))
        self.assertTrue(any("Held_Lightmap.exr" in m for m in caught.output))
        self.assertFalse(any("Gone_Lightmap.exr" in m for m in caught.output))

    def _link_dir(self, target, link):
        """*link* naming the folder *target*: a junction on Windows (no
        privilege needed), a symlink elsewhere; removed before the scratch."""
        try:
            if os.name == "nt":
                import _winapi

                _winapi.CreateJunction(target, link)
            else:
                os.symlink(target, link, target_is_directory=True)
        except (ImportError, AttributeError, OSError) as error:
            self.skipTest(f"no folder link here ({error})")
        self.addCleanup(os.unlink if os.path.islink(link) else os.rmdir, link)

    def test_a_file_read_under_another_spelling_is_kept(self):
        """A junction, a ``subst`` or mapped drive names one file two ways,
        and no comparison of spellings sees it: a re-bake that reaches its old
        folder under another spelling wrote its NEW map over the old path, and
        deleting that path deletes what every reader reads."""
        old = self._file("old", "Crate_Lightmap.exr")
        alias = os.path.join(self.tmp, "alias")
        self._link_dir(os.path.dirname(old), alias)
        after = [
            ("|crate", "Crate_Lightmap.exr", os.path.join(alias, "Crate_Lightmap.exr"))
        ]
        self.assertEqual(FileDependencies.remove_superseded([old], after), [])
        self.assertTrue(os.path.exists(old))

    def test_a_relative_spelling_is_never_deleted(self):
        """Relative to what? Against the process CWD -- in a DCC, wherever it
        was launched from -- it could name any file: never a guess."""
        old = self._file("old", "Crate_Lightmap.exr")
        cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, cwd)
        relative = os.path.join("old", "Crate_Lightmap.exr")
        self.assertEqual(FileDependencies.remove_superseded([relative], []), [])
        self.assertTrue(os.path.exists(old))

    def test_a_reader_spelled_relative_keeps_every_file_of_its_name(self):
        """A reader the host could not place is one found nowhere: whatever
        the CWD, its name keeps the file."""
        old = self._file("old", "Crate_Lightmap.exr")
        after = [("|crate", "Crate_Lightmap.exr", "new/Crate_Lightmap.exr")]
        self.assertEqual(FileDependencies.remove_superseded([old], after), [])
        self.assertTrue(os.path.exists(old))

    def test_a_folder_is_neither_deleted_nor_reported(self):
        folder = self._dir("old", "Crate_Lightmap.exr")  # named like a map
        with unittest.mock.patch.object(FileDependencies.logger, "warning") as warn:
            removed = FileDependencies.remove_superseded([folder], [])
        self.assertEqual(removed, [])
        self.assertTrue(os.path.isdir(folder))
        warn.assert_not_called()


class TestSetAside(_FilesCase):
    """Where a superseded file goes: the platform trash, else the
    ``superseded`` temp store (aged out after a month) -- never a permanent
    delete (BACKLOG 2026-09-23, decided 2026-09-27: a scene cannot see another
    file's reads, so what it retires must stay one restore away), and since
    2026-10-13 never a ``_superseded`` folder left in the project either. The
    trash itself is pinned here; its backends are ``FileUtils.move_to_trash``'s
    tests."""

    def test_the_trash_takes_it_when_the_volume_has_one(self):
        path = self._file("maps", "Crate_Lightmap.exr")
        bin_dir = self._dir("bin")

        def trash(src):
            dst = os.path.join(bin_dir, os.path.basename(src))
            os.replace(src, dst)
            return dst

        with unittest.mock.patch.object(
            ptk.FileUtils, "move_to_trash", side_effect=trash
        ):
            where = FileDependencies.set_aside(path)
        self.assertTrue(self._same(where, os.path.join(bin_dir, "Crate_Lightmap.exr")))
        self.assertFalse(
            os.path.exists(
                os.path.join(self.tmp, "maps", FileDependencies.SUPERSEDED_DIR)
            )
        )

    def test_with_no_trash_it_goes_to_the_temp_store_under_its_name(self):
        first = self._file("maps", "Crate_Lightmap.exr", data=b"first")
        with unittest.mock.patch.object(
            ptk.FileUtils, "move_to_trash", return_value=None
        ):
            where = FileDependencies.set_aside(first)
            second = self._file("maps", "Crate_Lightmap.exr", data=b"second")
            again = FileDependencies.set_aside(second)
        store = os.path.normcase(
            os.path.realpath(
                os.path.join(
                    ptk.TempArtifacts.root(), FileDependencies.SUPERSEDED_STORE
                )
            )
        )
        for path in (where, again):
            self.assertEqual(os.path.basename(path), "Crate_Lightmap.exr")
            self.assertTrue(
                os.path.normcase(os.path.realpath(path)).startswith(store), path
            )
        self.assertNotEqual(os.path.dirname(where), os.path.dirname(again))
        self.assertFalse(
            os.path.exists(
                os.path.join(self.tmp, "maps", FileDependencies.SUPERSEDED_DIR)
            ),
            "a folder was left in the project",
        )
        self.assertFalse(os.path.exists(first))
        for path, data in ((where, b"first"), (again, b"second")):
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), data)

    def test_a_file_that_cannot_move_stays_where_it_was(self):
        path = self._file("maps", "Held_Lightmap.exr")
        with unittest.mock.patch.object(
            ptk.FileUtils,
            "move_to_trash",
            side_effect=PermissionError(13, "held open", path),
        ):
            with self.assertRaises(PermissionError):
                FileDependencies.set_aside(path)
        self.assertTrue(os.path.isfile(path))

    def _rename_fails(self, path, error):
        """``os.rename`` of *path* raising *error*; any other rename runs."""
        real = os.rename

        def rename(src, dst, *a, **kw):
            if os.path.normcase(os.path.abspath(src)) == os.path.normcase(path):
                raise error
            return real(src, dst, *a, **kw)

        return unittest.mock.patch("os.rename", side_effect=rename)

    def test_another_volume_is_copied_then_the_original_removed(self):
        import errno

        path = self._file("maps", "Far_Lightmap.exr", data=b"far")
        with unittest.mock.patch.object(
            ptk.FileUtils, "move_to_trash", return_value=None
        ), self._rename_fails(
            os.path.normcase(path), OSError(errno.EXDEV, "cross-device", path)
        ):
            where = FileDependencies.set_aside(path)
        self.assertFalse(os.path.exists(path))
        with open(where, "rb") as fh:
            self.assertEqual(fh.read(), b"far")

    def test_a_rename_refused_for_another_reason_is_not_copied(self):
        """Held open (or denied) is not "another volume": a copy then a remove
        would only race the holder -- it raises, nothing duplicated."""
        path = self._file("maps", "Held_Lightmap.exr")
        copies = []
        with unittest.mock.patch.object(
            ptk.FileUtils, "move_to_trash", return_value=None
        ), self._rename_fails(
            os.path.normcase(path), PermissionError(13, "held open", path)
        ), unittest.mock.patch(
            "shutil.copy2", side_effect=lambda *a, **k: copies.append(a)
        ):
            with self.assertRaises(PermissionError):
                FileDependencies.set_aside(path)
        self.assertEqual(copies, [])
        self.assertTrue(os.path.isfile(path))

    def test_a_failed_copy_leaves_no_partial_file_behind(self):
        import errno

        path = self._file("maps", "Big_Lightmap.exr", data=b"big")
        made = []

        def partial(src, dst, *a, **k):
            with open(dst, "wb") as fh:
                fh.write(b"b")
            made.append(dst)
            raise OSError(errno.ENOSPC, "disk full", dst)

        with unittest.mock.patch.object(
            ptk.FileUtils, "move_to_trash", return_value=None
        ), self._rename_fails(
            os.path.normcase(path), OSError(errno.EXDEV, "cross-device", path)
        ), unittest.mock.patch("shutil.copy2", side_effect=partial):
            with self.assertRaises(OSError):
                FileDependencies.set_aside(path)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(len(made), 1)
        self.assertFalse(os.path.exists(made[0]), "a partial copy was left")
        self.assertFalse(os.path.exists(os.path.dirname(made[0])))

    def test_a_batch_sweeps_the_store_once_not_per_file(self):
        """Each call is a fresh store; ``sweep_interval`` keeps N set-asides
        from rescanning the store N times -- the first still sweeps it."""
        real = ptk.TempArtifacts.sweep_stale
        swept = []

        def sweep(store):
            if store.prefix == FileDependencies.SUPERSEDED_STORE:
                swept.append(store)
            return real(store)

        paths = [self._file("maps", f"Lightmap_{i}.exr") for i in range(3)]
        with unittest.mock.patch.object(
            ptk.FileUtils, "move_to_trash", return_value=None
        ), unittest.mock.patch.dict(
            ptk.TempArtifacts._swept_at, clear=True
        ), unittest.mock.patch.object(
            ptk.TempArtifacts, "sweep_stale", autospec=True, side_effect=sweep
        ):
            for path in paths:
                FileDependencies.set_aside(path)
        self.assertEqual(len(swept), 1)
        self.assertEqual(swept[0].max_age_days, FileDependencies.SUPERSEDED_DAYS)

    def test_a_walk_never_finds_a_file_set_aside(self):
        """It keeps its name: found by a walk, the stale map it replaced would
        be bound again -- the leftover the move exists to retire."""
        path = self._file("tex", "maps", "Crate_Lightmap.exr")
        with unittest.mock.patch.object(
            ptk.FileUtils, "move_to_trash", return_value=None
        ):
            FileDependencies.set_aside(path)
        root = os.path.join(self.tmp, "tex")
        self.assertEqual(FileDependencies.find_files(["Crate_Lightmap.exr"], root), [])
        dep = FileDependencies.resolve(
            [("|crate", "Crate_Lightmap.exr", "")], walk_root=root
        )[0]
        self.assertIsNone(dep["path"])
        plan = FileDependencies.relocate(
            [dep], self._dir("gathered"), source_dir=root, dry_run=True
        )
        self.assertEqual(len(plan["missing"]), 1, plan)


class TestWalk(_FilesCase):
    """One pruning for every search that finds files by name (2026-09-27):
    sync-client caches, the OS's trash and volume folders, version control,
    dependency and bytecode caches, and ``_superseded``. Each holds stale
    same-named copies a walk would bind as current. mayatk's texture walk
    pruned them (with ``$RECYCLE.BIN`` spelled as FAT spells it, so NTFS's
    ``$Recycle.Bin`` got through); blendertk's pruned nothing."""

    STALE = (
        ".dropbox.cache",
        ".Dropbox",
        "$Recycle.Bin",
        "System Volume Information",
        ".git",
        ".svn",
        ".hg",
        "node_modules",
        "__pycache__",
        "_Superseded",
        ".Trash-1000",
        ".Trashes",
    )

    def test_a_walk_never_enters_a_folder_of_stale_copies(self):
        wanted = self._file("tex", "maps", "Crate.exr")
        for folder in self.STALE:
            self._file("tex", folder, "Crate.exr")
            self._file("tex", "maps", folder, "deeper", "Crate.exr")
        root = os.path.join(self.tmp, "tex")
        found = FileDependencies.find_files(["crate.EXR"], root)
        self.assertEqual(
            [os.path.normcase(p) for p in found], [os.path.normcase(wanted)]
        )
        walked = {
            os.path.basename(folder) for folder, _d, _f in FileDependencies.walk(root)
        }
        self.assertEqual(walked, {"tex", "maps"})

    def test_the_skip_names_compare_without_case(self):
        for folder in self.STALE:
            self.assertTrue(FileDependencies._skips_dir(folder), folder)
            self.assertTrue(FileDependencies._skips_dir(folder.upper()), folder)
        for folder in ("maps", "trash", "git", "superseded", "Dropbox"):
            self.assertFalse(FileDependencies._skips_dir(folder), folder)


class TestWrittenHere(_FilesCase):
    """Whether a writer record's entry makes a file the scene's own to delete:
    the one rule both DCCs' re-bakes apply before deleting anything."""

    def test_the_writer_decides_whose_file_it_is(self):
        here = self._file("proj", "scenes", "room.ma")
        source = self._file("proj", "scenes", "source.ma")
        base = os.path.join(self.tmp, "proj")
        own = FileDependencies.written_here
        self.assertFalse(own(None, here, base), "no entry: nobody's to delete")
        self.assertTrue(own("", "", None), "written while unsaved, still unsaved")
        self.assertTrue(
            own("scenes/room.ma", here, base), "this scene, spelled from its project"
        )
        self.assertTrue(own(here, here, base), "this scene, absolute")
        self.assertFalse(
            own("scenes/source.ma", here, base), "a copy's source, still there"
        )
        os.remove(source)
        self.assertTrue(
            own("scenes/source.ma", here, base), "renamed: the old file is gone"
        )

    def test_an_entry_made_unsaved_is_own_only_until_the_first_save(self):
        """Saved, the scene can be copied: a Save As writes the same ``""``
        into the copy, and nothing tells the two files apart -- each would
        delete the maps the other still reads."""
        here = self._file("proj", "scenes", "room.ma")
        base = os.path.join(self.tmp, "proj")
        self.assertFalse(FileDependencies.written_here("", here, base))

    def test_a_relative_writer_with_no_project_to_read_it_from_is_nobodys(self):
        """Unresolvable is not gone: against the CWD it names nothing, and
        "the writer is gone" would hand its maps to this scene."""
        self.assertFalse(FileDependencies.written_here("scenes/source.ma", "", None))


class TestResolve(_FilesCase):
    def test_the_hint_then_the_search_folders(self):
        hinted = self._file("hint", "A.exr")
        found = self._file("tex", "B.exr")
        deps = {
            d["name"]: d
            for d in FileDependencies.resolve(
                [
                    ("|a", "A.exr", os.path.dirname(hinted)),
                    ("|b", "B.exr", self._dir("gone")),
                    ("|b2", "B.exr", self._dir("gone")),
                ],
                search_dirs=[self._dir("tex")],
            )
        }
        self.assertEqual(deps["A.exr"]["found_by"], FileDependencies.FOUND_BY_HINT)
        self.assertTrue(self._same(deps["A.exr"]["path"], hinted))
        self.assertEqual(deps["B.exr"]["found_by"], FileDependencies.FOUND_BY_SEARCH)
        self.assertEqual(deps["B.exr"]["owners"], ["|b", "|b2"])
        self.assertTrue(self._same(deps["B.exr"]["path"], found))

    def test_a_portable_hint_resolves_through_the_host(self):
        real = self._file("project", "lm", "C.exr")
        project = os.path.join(self.tmp, "project")
        (dep,) = FileDependencies.resolve(
            [("|c", "C.exr", "lm")],  # stored relative to the project
            resolve_hint=lambda folder, _name: os.path.join(project, folder),
        )
        self.assertEqual(dep["found_by"], FileDependencies.FOUND_BY_HINT)
        self.assertTrue(self._same(dep["path"], real))
        self.assertEqual(dep["dir"], "lm", "the recorded spelling is reported as is")

    def test_the_walk_takes_a_unique_hit_and_refuses_to_guess(self):
        self._file("root", "deep", "U.exr")
        self._file("root", "one", "D.exr")
        self._file("root", "two", "D.exr")
        refs = [("|u", "U.exr", ""), ("|d", "D.exr", "")]
        deps = {
            d["name"]: d
            for d in FileDependencies.resolve(refs, walk_root=self._dir("root"))
        }
        self.assertEqual(deps["U.exr"]["found_by"], FileDependencies.FOUND_BY_SEARCH)
        self.assertIsNone(deps["D.exr"]["path"])
        self.assertIn("ambiguous: 2", deps["D.exr"]["note"])
        # No walk root, no walk.
        self.assertIsNone(FileDependencies.resolve(refs)[0]["path"])

    def test_a_host_walk_replaces_the_default(self):
        seen = []

        def find(names, root):
            seen.append((sorted(names), root))
            return []

        root = self._dir("root")
        FileDependencies.resolve([("|x", "X.exr", "")], walk_root=root, find_files=find)
        self.assertEqual(seen, [(["X.exr"], root)])


class TestSearchDirs(_FilesCase):
    def test_the_most_read_folder_leads_then_the_host_folders(self):
        few = self._file("few", "F.exr")
        many = self._file("many", "M.exr")
        deps = FileDependencies.resolve(
            [
                ("|f", "F.exr", os.path.dirname(few)),
                ("|m1", "M.exr", os.path.dirname(many)),
                ("|m2", "M.exr", os.path.dirname(many)),
                ("|lost", "L.exr", self._dir("gone")),
            ]
        )
        tex = self._dir("tex")
        dirs = FileDependencies.search_dirs(deps, then=[tex, os.path.dirname(few)])
        self.assertEqual(len(dirs), 3, dirs)
        for got, want in zip(dirs, (os.path.dirname(many), os.path.dirname(few), tex)):
            self.assertTrue(self._same(got, want), dirs)

    def test_a_folder_holding_a_stale_twin_follows_the_maps_own_folder(self):
        """A consumer joins each name onto the list and takes the first hit, so
        a more-read folder holding a same-named leftover must not lead: on a
        production room the texture root (105 readers) held a 512 px atlas
        from an earlier bake under the name the current 2048 px one carried in
        its own folder (46 readers), and the GLB shipped the stale map through
        the current rects."""
        root = self._dir("sourceimages")
        self._file("sourceimages", "Cab.exr")
        # The exact name, so the shadow is real on any file system (there it
        # differed in case alone, which a case-folding one joins the same).
        self._file("sourceimages", "Room_Lightmap.exr", data=b"old")
        current = self._file("sourceimages", "Room", "Room_Lightmap.exr", data=b"new")
        refs = [(f"|c{i}", "Cab.exr", root) for i in range(3)]
        refs += [
            (f"|r{i}", "Room_Lightmap.exr", os.path.dirname(current)) for i in range(2)
        ]
        deps = FileDependencies.resolve(refs)
        dirs = FileDependencies.search_dirs(deps, then=[root])

        def first_hit(name):
            return next(
                os.path.join(d, name)
                for d in dirs
                if os.path.isfile(os.path.join(d, name))
            )

        self.assertTrue(self._same(first_hit("Room_Lightmap.exr"), current), dirs)
        self.assertTrue(self._same(first_hit("Cab.exr"), os.path.join(root, "Cab.exr")))

    def test_twins_that_shadow_each_other_keep_the_read_order_and_warn(self):
        """Two maps each shadowed by the other's folder cannot both be first:
        no folder order serves them, so the most-read order stands and says so."""
        a, b = self._dir("a"), self._dir("b")
        self._file("a", "X.exr")
        self._file("a", "Y.exr")
        self._file("b", "X.exr")
        self._file("b", "Y.exr")
        refs = [("|x1", "X.exr", a), ("|x2", "X.exr", a), ("|y", "Y.exr", b)]
        deps = FileDependencies.resolve(refs)
        with self.assertLogs(FileDependencies.logger, level="WARNING") as logs:
            dirs = FileDependencies.search_dirs(deps)
        self.assertTrue(self._same(dirs[0], a) and self._same(dirs[1], b), dirs)
        self.assertIn("Y.exr", "\n".join(logs.output))

    def test_each_folder_is_listed_once_and_only_hits_are_stated(self):
        """The shadow check lists every folder once; a stat per dependency per
        folder made a large scene's texture set cost thousands of probes."""
        maps = self._dir("maps")
        refs = []
        for i in range(20):
            self._file("maps", f"M{i}.exr")
            refs.append((f"|m{i}", f"M{i}.exr", maps))
        self._file("other", "M3.exr", data=b"stale twin")
        host = [self._dir("other"), self._dir("tex")]
        deps = FileDependencies.resolve(refs)
        listed, stated = [], []
        real_list, real_isfile = os.listdir, os.path.isfile

        def listdir(path="."):
            listed.append(os.path.normcase(os.path.abspath(path)))
            return real_list(path)

        def isfile(path):
            stated.append(path)
            return real_isfile(path)

        with unittest.mock.patch("os.listdir", side_effect=listdir), unittest.mock.patch(
            "os.path.isfile", side_effect=isfile
        ):
            dirs = FileDependencies.search_dirs(deps, then=host)
        self.assertEqual(len(listed), len(set(listed)), listed)
        self.assertEqual([os.path.basename(p) for p in stated], ["M3.exr"])
        self.assertTrue(self._same(dirs[0], maps), dirs)


class TestRelocate(_FilesCase):
    def test_plan_then_copy_newest_wins(self):
        dest = self._dir("dest")
        here = self._file("dest", "In.exr")
        src = self._file("old", "Out.exr")
        older = self._file("pool", "a", "Lost.exr", data=b"old")
        self._file("pool", "b", "Lost.exr", data=b"new")
        os.utime(older, (1_000_000, 1_000_000))
        deps = FileDependencies.resolve(
            [
                ("|in", "In.exr", os.path.dirname(here)),
                ("|out", "Out.exr", os.path.dirname(src)),
                ("|lost", "Lost.exr", self._dir("gone")),
                ("|none", "None.exr", self._dir("gone")),
            ]
        )

        plan = FileDependencies.relocate(
            deps, dest, source_dir=self._dir("pool"), dry_run=True
        )
        self.assertEqual(len(plan["in_place"]), 1)
        self.assertEqual(
            sorted(os.path.basename(d) for _s, d in plan["relocate"]),
            ["Lost.exr", "Out.exr"],
        )
        self.assertEqual([m["name"] for m in plan["missing"]], ["None.exr"])
        self.assertFalse(os.path.exists(os.path.join(dest, "Out.exr")), "dry run wrote")

        done = FileDependencies.relocate(deps, dest, source_dir=self._dir("pool"))
        self.assertEqual(len(done["copied"]), 2)
        with open(os.path.join(dest, "Lost.exr"), "rb") as fh:
            self.assertEqual(fh.read(), b"new", "the newest same-named file wins")
        self.assertTrue(os.path.exists(src), "a copy leaves its source")

    def test_a_host_copy_replaces_the_default(self):
        src = self._file("old", "Out.exr")
        deps = FileDependencies.resolve([("|o", "Out.exr", os.path.dirname(src))])
        calls = []

        def copy(sources, dest_dir, mode):
            calls.append((list(sources), dest_dir, mode))
            return []

        dest = self._dir("dest")
        result = FileDependencies.relocate(deps, dest, mode="move", copy=copy)
        self.assertEqual(
            calls, [([src.replace("\\", "/")], dest.replace("\\", "/"), "move")]
        )
        self.assertEqual(result["copied"], [])


class TestCopyFiles(_FilesCase):
    def test_a_different_file_of_the_same_name_is_never_overwritten(self):
        dest = self._dir("dest")
        theirs = self._file("dest", "Map.exr", data=b"theirs!")
        ours = self._file("new", "Map.exr", data=b"ours")
        self.assertEqual(FileDependencies.copy_files([ours], dest), [])
        with open(theirs, "rb") as fh:
            self.assertEqual(fh.read(), b"theirs!")

    def test_the_same_file_is_reused_and_a_move_drops_the_source(self):
        dest = self._dir("dest")
        self._file("dest", "Map.exr", data=b"same")
        ours = self._file("new", "Map.exr", data=b"same")
        landed = FileDependencies.copy_files([ours], dest, mode="move")
        self.assertEqual(len(landed), 1)
        self.assertFalse(os.path.exists(ours))

    def test_a_new_file_is_copied(self):
        dest = self._dir("dest")
        ours = self._file("new", "Map.exr")
        (landed,) = FileDependencies.copy_files([ours], dest)
        self.assertTrue(os.path.isfile(landed[1]))
        self.assertTrue(os.path.isfile(ours))

    def test_a_same_size_different_file_is_never_taken_for_the_source(self):
        """Two bakes of one map are the same size in any uncompressed format.
        Size was the test: a move deleted the NEW bake as a redundant copy and
        reported the stale file as landed."""
        dest = self._dir("dest")
        stale = self._file("dest", "Map.exr", data=b"OLD-BAKE")
        fresh = self._file("new", "Map.exr", data=b"NEW-BAKE")
        self.assertEqual(FileDependencies.copy_files([fresh], dest, mode="move"), [])
        with open(fresh, "rb") as fh:
            self.assertEqual(fh.read(), b"NEW-BAKE", "the new bake was deleted")
        with open(stale, "rb") as fh:
            self.assertEqual(fh.read(), b"OLD-BAKE")


def _alias_dir(target, alias):
    """A junction (Windows) or symlink to *target*; False when neither can be made."""
    try:
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(target, alias)
        else:
            os.symlink(target, alias, target_is_directory=True)
    except (OSError, AttributeError, NotImplementedError):
        return False
    return True


class TestAliasedFolders(_FilesCase):
    """A folder reached through another spelling of itself -- a junction, a
    symlink, a ``subst`` or mapped drive -- is the SAME folder. Compared as
    strings it read as a second one, and a move "dropped the redundant source":
    the only copy."""

    def setUp(self):
        super().setUp()
        self.real = self._dir("real")
        self.alias = os.path.join(self.tmp, "alias")
        if not _alias_dir(self.real, self.alias):
            self.skipTest("cannot create a directory junction / symlink here")
        # Unlinked first (cleanups run last-in, first-out), so the folder's
        # own cleanup never walks through the link into its target.
        self.addCleanup(self._unalias)

    def _unalias(self):
        if os.path.islink(self.alias):
            os.unlink(self.alias)
        elif os.path.isdir(self.alias):
            os.rmdir(self.alias)  # a junction: removes the link, not the target

    def test_a_move_into_an_alias_of_the_source_folder_keeps_the_file(self):
        src = self._file("real", "Map.exr", data=b"only copy")
        deps = FileDependencies.resolve([("|m", "Map.exr", self.real)])
        result = FileDependencies.relocate(deps, self.alias, mode="move")
        self.assertEqual(result["relocate"], [])
        self.assertEqual(len(result["in_place"]), 1)
        self.assertTrue(os.path.isfile(src), "the only copy was deleted")

    def test_copy_files_leaves_a_source_that_is_its_own_destination(self):
        src = self._file("real", "Map.exr", data=b"only copy")
        landed = FileDependencies.copy_files([src], self.alias, mode="move")
        self.assertEqual(len(landed), 1)
        self.assertTrue(os.path.isfile(src), "the only copy was deleted")


class TestResolveHints(_FilesCase):
    def test_every_recorded_folder_is_tried_before_the_search(self):
        """After a partial repath one file's owners record different folders;
        only the first was tried, so a stale one hid the folder holding it."""
        good = self._file("good", "Atlas_Lightmap.exr")
        (dep,) = FileDependencies.resolve(
            [
                ("|a", "Atlas_Lightmap.exr", self._dir("stale")),
                ("|b", "Atlas_Lightmap.exr", os.path.dirname(good)),
            ]
        )
        self.assertEqual(dep["found_by"], FileDependencies.FOUND_BY_HINT)
        self.assertTrue(self._same(dep["path"], good))
        self.assertTrue(self._same(dep["dir"], os.path.dirname(good)))

    def test_the_host_resolver_gets_the_folder_as_recorded(self):
        """A UNC folder respelled with forward slashes is ``//server/share``,
        which Blender reads as relative to the .blend."""
        seen = []
        FileDependencies.resolve(
            [("|u", "U.exr", "\\\\nas\\share\\lm")],
            resolve_hint=lambda folder, _name: seen.append(folder) or "",
        )
        self.assertEqual(seen, ["\\\\nas\\share\\lm"])

    def test_a_unc_path_keeps_its_prefix_when_respelled(self):
        self.assertEqual(
            FileDependencies._spelled("\\\\nas\\share\\lm\\A.exr"),
            "\\\\nas/share/lm/A.exr",
        )
        self.assertEqual(FileDependencies._spelled("C:\\lm\\A.exr"), "C:/lm/A.exr")

    def test_search_dirs_lists_existing_folders_only(self):
        deps = FileDependencies.resolve([("|x", "X.exr", "")])
        tex = self._dir("tex")
        gone = os.path.join(self.tmp, "gone")
        dirs = FileDependencies.search_dirs(deps, then=[gone, tex])
        self.assertEqual(len(dirs), 1, dirs)
        self.assertTrue(self._same(dirs[0], tex))


class TestReferenceBytes(_FilesCase):
    """The part of a file that can name another, read without its app: a Maya
    ASCII scene's header, a ``.blend`` decompressed, anything else whole --
    the scan a purge of derived files reads (hoisted from the DCC bake
    stores, 2026-10-13)."""

    def test_a_maya_ascii_scene_is_read_up_to_its_first_node(self):
        path = self._file(
            "a.ma",
            data=b'file -r "x.ma";\nrequires maya;\ncreateNode t;\nfile -r "y";\n',
        )
        self.assertEqual(
            FileDependencies.reference_bytes(path),
            b'file -r "x.ma";\nrequires maya;\n',
        )

    def test_any_other_file_is_read_whole(self):
        path = self._file("a.mb", data=b"FOR4\x00createNode bake.ma")
        self.assertEqual(
            FileDependencies.reference_bytes(path), b"FOR4\x00createNode bake.ma"
        )

    def test_a_raw_blend_is_read_as_it_is(self):
        path = self._file("a.blend", data=b"BLENDER-v300lib.blend")
        self.assertEqual(
            FileDependencies.reference_bytes(path), b"BLENDER-v300lib.blend"
        )

    def test_a_gzip_blend_is_decompressed(self):
        import gzip

        path = self._file("a.blend", data=gzip.compress(b"BLENDER-v279lib.blend"))
        self.assertEqual(
            FileDependencies.reference_bytes(path), b"BLENDER-v279lib.blend"
        )

    def test_a_zstd_file_with_no_decoder_here_is_unreadable(self):
        path = self._file("a.blend", data=b"\x28\xb5\x2f\xfd" + b"\x00" * 8)
        with unittest.mock.patch.dict(
            "sys.modules", {"compression": None, "zstandard": None}
        ):
            self.assertIsNone(FileDependencies.open_decompressed(path))
            self.assertIsNone(FileDependencies.reference_bytes(path))

    def test_a_zstd_file_reads_through_the_optional_zstandard(self):
        """``compression.zstd`` (Python 3.14+) first, else ``zstandard``
        (Blender bundles it), across frames -- Blender writes one per block."""
        import io
        import types

        seen = {}

        class _Decompressor:
            def stream_reader(self, source, read_across_frames=False, closefd=True):
                seen["across"] = read_across_frames
                source.close()
                return io.BytesIO(b"BLENDER-v400lib.blend")

        fake = types.SimpleNamespace(ZstdDecompressor=_Decompressor)
        path = self._file("a.blend", data=b"\x28\xb5\x2f\xfd" + b"\x00" * 8)
        with unittest.mock.patch.dict(
            "sys.modules", {"compression": None, "zstandard": fake}
        ):
            self.assertEqual(
                FileDependencies.reference_bytes(path), b"BLENDER-v400lib.blend"
            )
        self.assertTrue(seen["across"])

    def test_a_missing_file_is_unreadable(self):
        self.assertIsNone(
            FileDependencies.reference_bytes(os.path.join(self.tmp, "gone.ma"))
        )

    @staticmethod
    def _corrupt_decoders():
        """Stub ``compression.zstd`` and ``zstandard`` whose streams fail as a
        corrupt frame does: with the module's own ``ZstdError``, an
        ``Exception`` -- not an ``OSError`` -- in both real decoders."""
        import types

        def failing(error):
            class _Stream:
                def read(self, *a):
                    raise error("Unknown frame descriptor")

                def close(self):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    self.close()

            return _Stream()

        class StdError(Exception):
            pass

        class ZstandardError(Exception):
            pass

        class _Decompressor:
            def stream_reader(self, source, **kw):
                source.close()
                return failing(ZstandardError)

        std = types.SimpleNamespace(
            ZstdError=StdError, open=lambda path, mode: failing(StdError)
        )
        return {
            "stdlib": {"compression": types.SimpleNamespace(zstd=std)},
            "zstandard": {
                "compression": None,
                "zstandard": types.SimpleNamespace(
                    ZstdDecompressor=_Decompressor, ZstdError=ZstandardError
                ),
            },
        }

    def test_a_corrupt_zstd_file_is_unreadable_not_an_error(self):
        path = self._file("a.blend", data=b"\x28\xb5\x2f\xfd" + b"\x00" * 8)
        for decoder, modules in self._corrupt_decoders().items():
            with self.subTest(decoder), unittest.mock.patch.dict(
                "sys.modules", modules
            ):
                self.assertIsNone(FileDependencies.reference_bytes(path))
                self.assertIsNone(
                    FileDependencies.scan_references(self.tmp, rb"lib", [".blend"])
                )
                with FileDependencies.open_decompressed(path) as stream:
                    with self.assertRaises(OSError):
                        stream.read(16)


class TestScanReferences(_FilesCase):
    """Every name a pattern finds in the files under a folder -- what still
    reads a derived file -- or None when one of them cannot be read."""

    PATTERN = rb"bk_cache_([a-z0-9]+)\."

    def test_matches_from_every_listed_type_are_gathered(self):
        self._file("s", "a.ma", data=b'file -r "bk_cache_one.ma";\ncreateNode t;\n')
        self._file("s", "deep", "b.mb", data=b"\x00bk_cache_two.ma\x00")
        self._file("s", "c.txt", data=b"bk_cache_three.ma")
        self.assertEqual(
            FileDependencies.scan_references(self.tmp, self.PATTERN, (".ma", ".mb")),
            {"one", "two"},
        )

    def test_a_skipped_folder_is_not_entered(self):
        self._file("cache", "a.ma", data=b"bk_cache_one.ma")
        self._file("cache", "deep", "b.ma", data=b"bk_cache_two.ma")
        self._file("s", "c.ma", data=b"bk_cache_three.ma")
        self.assertEqual(
            FileDependencies.scan_references(
                self.tmp, self.PATTERN, (".ma",), skip=[os.path.join(self.tmp, "cache")]
            ),
            {"three"},
        )

    def test_a_folder_of_stale_copies_is_not_entered(self):
        self._file(".git", "a.ma", data=b"bk_cache_one.ma")
        self.assertEqual(
            FileDependencies.scan_references(self.tmp, self.PATTERN, (".ma",)), set()
        )

    def test_an_unreadable_file_answers_none(self):
        self._file("s", "a.ma", data=b"bk_cache_one.ma")
        with unittest.mock.patch.object(
            FileDependencies, "reference_bytes", return_value=None
        ):
            self.assertIsNone(
                FileDependencies.scan_references(self.tmp, self.PATTERN, (".ma",))
            )


if __name__ == "__main__":
    unittest.main()
