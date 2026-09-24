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
    """What a re-bake may delete: the files its owners read before, once
    nothing reads them. Every keep rule is a reader the delete would strand."""

    def test_a_file_nothing_reads_is_deleted_and_a_read_one_kept(self):
        old = self._file("old", "Crate_Lightmap.exr")
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

    def test_a_name_found_nowhere_keeps_every_file_of_that_name(self):
        """A reader whose file resolved nowhere may still find this one by a
        search the host did not run -- a walk of its texture tree."""
        old = self._file("old", "Crate_Lightmap.exr")
        after = [("|crate", "CRATE_lightmap.exr", None)]
        self.assertEqual(FileDependencies.remove_superseded([old], after), [])
        self.assertTrue(os.path.exists(old))

    def test_each_file_is_deleted_once_however_it_is_spelled(self):
        old = self._file("old", "Crate_Lightmap.exr")
        spelled = os.path.join(self.tmp, "old", ".", "Crate_Lightmap.exr")
        removed = FileDependencies.remove_superseded([old, spelled], [])
        self.assertEqual(removed, [old])

    def test_a_file_already_gone_or_held_open_is_not_reported_deleted(self):
        gone = os.path.join(self.tmp, "old", "Gone_Lightmap.exr")
        held = self._file("old", "Held_Lightmap.exr")
        real_remove = os.remove

        def remove(path):
            if os.path.basename(path) == "Held_Lightmap.exr":
                raise PermissionError(13, "held open", path)
            real_remove(path)

        with unittest.mock.patch("os.remove", side_effect=remove):
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


if __name__ == "__main__":
    unittest.main()
