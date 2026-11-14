#!/usr/bin/python
# coding=utf-8
"""Tests for ``pythontk.FileCollect`` (``file_utils/file_collect.py``).

A Collect gathers the files a set of readers use into one folder, keyed by
READER: names kept, a clash kept apart by the shortest run of parent folders,
identical files shared, files already in the destination counted, tile sets
moved whole. Real files in a scratch folder and the pure host are all a test
needs.
"""

import os
import posixpath
from unittest import mock

import pythontk as ptk
from pythontk import FileCollect, PureCollectHost

from conftest import BaseTestCase


class _CollectCase(BaseTestCase):
    """A scratch folder per test."""

    def setUp(self):
        self.store = ptk.TempArtifacts("ptk_file_collect")
        self.addCleanup(self.store.cleanup)
        self.tmp = self.store.dir_path()
        self.dest = os.path.join(self.tmp, "dest")

    def _file(self, *parts, data=b"x"):
        path = os.path.join(self.tmp, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    def _plan(self, readers, search_dir="", choices=None):
        """``{reader: stored path}`` resolved and planned into ``self.dest``."""
        collect = FileCollect(PureCollectHost(readers))
        entries = collect.entries(list(readers), search_dir=search_dir)
        return {r["owner"]: r for r in FileCollect.plan(entries, self.dest, choices)}

    def _at(self, *parts):
        return os.path.join(self.dest, *parts).replace("\\", "/")

    def _same(self, a, b):
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(
            os.path.abspath(b)
        )

    def _junction(self, link, target):
        """A directory junction *link* naming *target* (no elevation needed);
        removed -- never its target -- before the scratch folder is; skipped
        where none can be made."""
        try:
            import _winapi

            _winapi.CreateJunction(os.path.normpath(target), os.path.normpath(link))
        except (ImportError, AttributeError, OSError) as error:
            self.skipTest(f"no directory junction can be made here: {error}")
        self.addCleanup(lambda: os.path.lexists(link) and os.rmdir(link))
        return link

    @staticmethod
    def _read(path):
        with open(path, "rb") as fh:
            return fh.read()


class TestPlan(_CollectCase):
    def test_a_path_stored_on_windows_is_searched_by_its_file_name_on_posix(self):
        """A reader storing ``C:\\gone\\wood.png`` read on POSIX resolves to
        ``<cwd>/C:\\gone\\wood.png``, whose ``os.path.basename`` is the whole
        Windows path: the search looked for that and found nothing
        (``posixpath`` stands in for a POSIX host). Added: 2026-11-01"""
        wood = self._file("tex", "wood.png").replace("\\", "/")
        collect = FileCollect(PureCollectHost({"lamp": "C:\\gone_2026\\wood.png"}))
        with mock.patch("os.path.basename", posixpath.basename):
            (entry,) = collect.entries(
                ["lamp"], search_dir=self.tmp, find_files=lambda names, root: [wood]
            )
        self.assertTrue(entry["source"] and self._same(entry["source"], wood), entry)

    def test_a_lone_file_goes_flat(self):
        rows = self._plan({"a": self._file("ext", "valid.png")})
        self.assertEqual(rows["a"]["status"], FileCollect.RELOCATE)
        self.assertTrue(self._same(rows["a"]["destination"], self._at("valid.png")))
        self.assertEqual(rows["a"]["folder"], "")

    def test_same_named_different_files_each_keep_their_parent_folder(self):
        """BACKLOG 2026-10-23 (S1): ``wood/diffuse.png`` and
        ``metal/diffuse.png`` collapsed into ONE copy, and both nodes were
        repointed at it while the log reported success."""
        rows = self._plan(
            {
                "wood_file": self._file("tex", "wood", "diffuse.png", data=b"wood"),
                "metal_file": self._file("tex", "metal", "diffuse.png", data=b"metal"),
            }
        )
        self.assertTrue(
            self._same(rows["wood_file"]["destination"], self._at("wood", "diffuse.png"))
        )
        self.assertTrue(
            self._same(
                rows["metal_file"]["destination"], self._at("metal", "diffuse.png")
            )
        )
        self.assertEqual(rows["wood_file"]["folder"], "wood")
        self.assertEqual(
            {r["status"] for r in rows.values()}, {FileCollect.RELOCATE}
        )

    def test_the_shortest_run_that_tells_them_apart(self):
        rows = self._plan(
            {
                "a": self._file("a", "x", "wood", "d.png", data=b"a"),
                "b": self._file("b", "x", "wood", "d.png", data=b"b"),
                "c": self._file("c", "metal", "d.png", data=b"c"),
            }
        )
        self.assertEqual(rows["a"]["folder"], "a/x/wood")
        self.assertEqual(rows["b"]["folder"], "b/x/wood")
        self.assertEqual(rows["c"]["folder"], "metal")

    def test_identical_files_of_a_name_share_one_copy(self):
        rows = self._plan(
            {
                "a": self._file("one", "d.png", data=b"same"),
                "b": self._file("two", "d.png", data=b"same"),
            }
        )
        self.assertEqual(rows["a"]["destination"], rows["b"]["destination"])
        self.assertTrue(self._same(rows["a"]["destination"], self._at("d.png")))
        self.assertEqual(rows["a"]["status"], FileCollect.RELOCATE)
        self.assertEqual(rows["b"]["status"], FileCollect.REUSE)

    def test_two_readers_of_one_file_are_one_copy(self):
        path = self._file("ext", "shared.png")
        rows = self._plan({"a": path, "b": path})
        self.assertEqual(rows["a"]["destination"], rows["b"]["destination"])
        self.assertEqual(rows["b"]["status"], FileCollect.RELOCATE)

    def test_a_different_file_already_in_the_destination_keeps_the_newcomer_apart(
        self,
    ):
        self._file("dest", "d.png", data=b"theirs")
        rows = self._plan({"a": self._file("ext", "d.png", data=b"ours")})
        self.assertTrue(self._same(rows["a"]["destination"], self._at("ext", "d.png")))

    def test_an_identical_file_already_in_the_destination_is_reused_flat(self):
        self._file("dest", "d.png", data=b"same")
        rows = self._plan({"a": self._file("ext", "d.png", data=b"same")})
        self.assertTrue(self._same(rows["a"]["destination"], self._at("d.png")))
        self.assertEqual(rows["a"]["status"], FileCollect.REUSE)

    def test_a_taken_parent_folder_reaches_one_further_up(self):
        self._file("dest", "wood", "d.png", data=b"theirs")
        rows = self._plan(
            {
                "a": self._file("set1", "wood", "d.png", data=b"one"),
                "b": self._file("set2", "metal", "d.png", data=b"two"),
            }
        )
        self.assertEqual(rows["a"]["folder"], "set1/wood")
        self.assertEqual(rows["b"]["folder"], "metal")

    def test_a_file_already_under_the_destination_stays_where_it_is(self):
        inside = self._file("dest", "sub", "d.png", data=b"inside")
        rows = self._plan(
            {
                "inside": inside,
                "twin": self._file("ext", "d.png", data=b"inside"),
                "other": self._file("other", "d.png", data=b"different"),
            }
        )
        self.assertEqual(rows["inside"]["status"], FileCollect.IN_PLACE)
        self.assertTrue(self._same(rows["inside"]["destination"], inside))
        self.assertEqual(rows["twin"]["status"], FileCollect.REUSE, "shares it")
        self.assertTrue(self._same(rows["twin"]["destination"], inside))
        self.assertTrue(
            self._same(rows["other"]["destination"], self._at("d.png")),
            "its name's one newcomer, the name free at the root: flat",
        )

    def test_a_namesake_identical_to_the_flat_file_reuses_it(self):
        """Files already at the destination count: the flat one holding the
        same bytes as one member is that member's; only the other is kept apart."""
        self._file("dest", "diffuse.png", data=b"wood")
        rows = self._plan(
            {
                "wood": self._file("tex", "wood", "diffuse.png", data=b"wood"),
                "metal": self._file("tex", "metal", "diffuse.png", data=b"metal"),
            }
        )
        self.assertTrue(self._same(rows["wood"]["destination"], self._at("diffuse.png")))
        self.assertEqual(rows["wood"]["status"], FileCollect.REUSE)
        self.assertEqual(rows["metal"]["folder"], "metal")

    def test_a_lone_newcomer_beside_a_namesake_already_inside_goes_flat(self):
        inside = self._file("dest", "sub", "d.png", data=b"inside")
        rows = self._plan(
            {"inside": inside, "new": self._file("ext", "d.png", data=b"new")}
        )
        self.assertEqual(rows["inside"]["status"], FileCollect.IN_PLACE)
        self.assertTrue(self._same(rows["new"]["destination"], self._at("d.png")))

    def test_identical_files_already_inside_stay_their_own(self):
        """Merged, a move deleted the second and a copy repathed its reader."""
        a = self._file("dest", "a", "d.png", data=b"same")
        b = self._file("dest", "b", "d.png", data=b"same")
        rows = self._plan({"a": a, "b": b})
        self.assertEqual(
            {r["status"] for r in rows.values()}, {FileCollect.IN_PLACE}
        )
        self.assertTrue(self._same(rows["b"]["destination"], b))

    def test_a_pick_is_its_rows_alone(self):
        """A texture and a lightmap may share an owner name (Blender names an
        image after its file): a pick for one never re-sources the other."""
        a = self._file("pool", "a", "Room_LM.exr", data=b"a")
        b = self._file("pool", "b", "Room_LM.exr", data=b"b")
        entries = [
            {"owner": "Room_LM.exr", "kind": "lightmap", "candidates": [a, b]},
            {"owner": "Room_LM.exr", "kind": "texture", "source": b},
        ]
        lightmap, texture = FileCollect.plan(
            entries, self.dest, choices={("lightmap", "Room_LM.exr"): a}
        )
        self.assertTrue(self._same(lightmap["source"], a))
        self.assertTrue(self._same(texture["source"], b), "the other row re-sourced")

    def test_a_tile_set_is_placed_as_one(self):
        for folder, data in (("a", b"a"), ("b", b"b")):
            self._file("sets", folder, "rock.1001.png", data=data)
            self._file("sets", folder, "rock.1002.png", data=data)
        rows = self._plan(
            {
                "a": os.path.join(self.tmp, "sets", "a", "rock.<UDIM>.png"),
                "b": os.path.join(self.tmp, "sets", "b", "rock.<UDIM>.png"),
            }
        )
        self.assertTrue(
            self._same(rows["a"]["destination"], self._at("a", "rock.<UDIM>.png"))
        )
        self.assertTrue(
            self._same(rows["b"]["destination"], self._at("b", "rock.<UDIM>.png"))
        )

    def test_a_folder_holding_another_tile_of_the_set_is_not_its_place(self):
        """The set read there would be a different set."""
        self._file("dest", "rock.1003.png", data=b"stray")
        self._file("ext", "rock.1001.png")
        rows = self._plan({"a": os.path.join(self.tmp, "ext", "rock.<UDIM>.png")})
        self.assertEqual(rows["a"]["folder"], "ext")

    def test_an_ambiguous_reader_waits_for_its_pick(self):
        a = self._file("pool", "a", "d.png", data=b"a")
        b = self._file("pool", "b", "d.png", data=b"b")
        gone = os.path.join(self.tmp, "gone", "d.png")
        search = os.path.join(self.tmp, "pool")

        rows = self._plan({"n": gone}, search_dir=search)
        self.assertEqual(rows["n"]["status"], FileCollect.AMBIGUOUS)
        self.assertIsNone(rows["n"]["destination"])
        self.assertEqual(len(rows["n"]["candidates"]), 2)

        picked = self._plan({"n": gone}, search_dir=search, choices={("file", "n"): b})
        self.assertEqual(picked["n"]["status"], FileCollect.RELOCATE)
        self.assertTrue(self._same(picked["n"]["source"], b))
        self.assertTrue(self._same(picked["n"]["destination"], self._at("d.png")))
        self.assertTrue(os.path.exists(a))

    def test_a_reader_found_nowhere_is_missing(self):
        rows = self._plan(
            {"n": os.path.join(self.tmp, "gone", "x.png"), "empty": ""},
            search_dir=self.tmp,
        )
        self.assertEqual(rows["n"]["status"], FileCollect.MISSING)
        self.assertEqual(rows["empty"]["status"], FileCollect.MISSING)
        self.assertEqual(rows["empty"]["note"], "no path")

    def test_a_resolving_reader_is_never_re_sourced_by_the_search(self):
        live = self._file("live", "d.png", data=b"live")
        self._file("pool", "d.png", data=b"stranger")
        rows = self._plan({"n": live}, search_dir=os.path.join(self.tmp, "pool"))
        self.assertTrue(self._same(rows["n"]["source"], live))


class TestApply(_CollectCase):
    def _apply(self, readers, mode="copy", copy=None):
        rows = list(self._plan(readers).values())
        return rows, FileCollect.apply(rows, mode=mode, copy=copy)

    def test_each_file_lands_at_its_own_destination(self):
        rows, landed = self._apply(
            {
                "wood": self._file("tex", "wood", "diffuse.png", data=b"wood"),
                "metal": self._file("tex", "metal", "diffuse.png", data=b"metal"),
            }
        )
        self.assertEqual(len(landed), 2)
        self.assertEqual(self._read(self._at("wood", "diffuse.png")), b"wood")
        self.assertEqual(self._read(self._at("metal", "diffuse.png")), b"metal")
        self.assertFalse(os.path.exists(self._at("diffuse.png")))

    def test_identical_files_land_once(self):
        _rows, landed = self._apply(
            {
                "a": self._file("one", "d.png", data=b"same"),
                "b": self._file("two", "d.png", data=b"same"),
            }
        )
        self.assertEqual(len(landed), 2)
        self.assertEqual(os.listdir(self.dest), ["d.png"])

    def test_a_move_takes_the_source_away_once_it_landed(self):
        a = self._file("one", "d.png", data=b"same")
        b = self._file("two", "d.png", data=b"same")
        self._apply({"a": a, "b": b}, mode="move")
        self.assertFalse(os.path.exists(a))
        self.assertFalse(os.path.exists(b), "a redundant identical source goes too")
        self.assertEqual(self._read(self._at("d.png")), b"same")

    def test_a_move_lands_every_reader_of_one_file(self):
        """Two readers of one file are one copy: the first row's move took the
        source away before the second row was compared with it, so the second
        reader was left out -- on the deleted path."""
        path = self._file("ext", "shared.png", data=b"shared")
        rows, landed = self._apply({"a": path, "b": path}, mode="move")
        self.assertEqual([r["owner"] for r in landed], ["a", "b"])
        self.assertFalse(os.path.exists(path))
        self.assertEqual(self._read(self._at("shared.png")), b"shared")

    def test_a_move_keeps_a_source_marked_to_keep(self):
        src = self._file("ext", "shared.exr")
        entries = [{"owner": "m", "source": src, "keep_source": True}]
        rows = FileCollect.plan(entries, self.dest)
        FileCollect.apply(rows, mode="move")
        self.assertTrue(os.path.exists(src))
        self.assertTrue(os.path.exists(self._at("shared.exr")))

    def test_a_failed_copy_keeps_its_source_and_its_reader(self):
        src = self._file("ext", "d.png")
        _rows, landed = self._apply(
            {"a": src}, mode="move", copy=lambda sources, folder, mode, progress=None: []
        )
        self.assertEqual(landed, [])
        self.assertTrue(os.path.exists(src))

    def test_a_different_file_that_appeared_since_the_plan_is_never_taken(self):
        """A copier that reuses a same-named file at the destination on a weak
        test (size, a partial hash) called it landed, and a move then deleted
        the only copy of the source."""
        src = self._file("ext", "d.png", data=b"ours")
        rows = list(self._plan({"a": src}).values())
        self._file("dest", "d.png", data=b"them")  # since the preview

        def lenient(sources, folder, mode, progress=None):
            return [(s, os.path.join(folder, os.path.basename(s))) for s in sources]

        landed = FileCollect.apply(rows, mode="move", copy=lenient)

        self.assertEqual(landed, [])
        self.assertEqual(self._read(src), b"ours")

    def test_a_file_one_reader_keeps_is_kept_whatever_the_others_say(self):
        path = self._file("ext", "shared.exr")
        rows = FileCollect.plan(
            [
                {"owner": "a", "source": path, "keep_source": True},
                {"owner": "b", "source": path},
            ],
            self.dest,
        )
        FileCollect.apply(rows, mode="move")
        self.assertTrue(os.path.exists(path))

    def test_a_file_a_reader_outside_the_plan_reads_stays_on_a_move(self):
        """A local and a linked image (or a scoped and an Arnold-only file node)
        reading one file: the move of the scoped one stranded the other."""
        shared = self._file("tex", "wood.png")

        class Host(PureCollectHost):
            def read_elsewhere(self, nodes):
                return [shared] if "outside" not in nodes else []

        collect = FileCollect(Host({"inside": shared, "outside": shared}))
        rows = collect.plan_nodes(["inside"], self.dest, kind="texture")

        self.assertEqual([r["keep_source"] for r in rows], [True])
        landed = FileCollect.apply(rows, mode="move")
        self.assertEqual(len(landed), 1, "the copy still lands and repaths")
        self.assertTrue(os.path.exists(shared), "the outside reader was stranded")
        self.assertTrue(os.path.exists(self._at("wood.png")))

    def test_what_a_move_keeps_is_decided_per_file_in_the_plan(self):
        """One row's flag is every row's of that file: the view and the move
        read the one decision."""
        path = self._file("ext", "shared.exr")
        rows = FileCollect.plan(
            [
                {"owner": "a", "kind": "lightmap", "source": path, "keep_source": True},
                {"owner": "b", "kind": "texture", "source": path},
                {"owner": "c", "source": self._file("ext", "own.png")},
            ],
            self.dest,
        )
        self.assertEqual([r["keep_source"] for r in rows], [True, True, False])
        self.assertEqual(
            FileCollect.kept_files(rows), {os.path.normcase(os.path.abspath(path))}
        )

    def test_a_move_takes_no_source_for_a_file_that_appeared_after_the_copier_looked(
        self,
    ):
        """A destination that appeared between the existence scan and the
        copier's own test (which reuses on a partial hash) is compared whole."""
        src = self._file("ext", "d.png", data=b"ours")
        rows = list(self._plan({"a": src}).values())

        def racing(sources, folder, mode, progress=None):
            for s in sources:  # someone else's bytes land first, same size
                with open(os.path.join(folder, os.path.basename(s)), "wb") as fh:
                    fh.write(b"them")
            return [(s, os.path.join(folder, os.path.basename(s))) for s in sources]

        landed = FileCollect.apply(rows, mode="move", copy=racing)

        self.assertEqual(landed, [], "the reader would be repathed to their file")
        self.assertEqual(self._read(src), b"ours")

    def test_a_cancel_says_what_was_already_copied(self):
        rows = list(
            self._plan({"a": self._file("one", "a.png"), "b": self._file("two", "b.png")}).values()
        )
        with self.assertRaises(ptk.OperationCancelled) as caught:
            FileCollect.apply(rows, mode="move", progress=lambda done, total: False)
        copied = len(os.listdir(self.dest))  # the copier stops at the next file
        self.assertIn(f"after {copied} of 2 file(s) were copied", str(caught.exception))

    def test_a_cancel_stops_before_anything_is_taken_away(self):
        a = self._file("one", "a.png")
        b = self._file("two", "b.png")
        rows = list(self._plan({"a": a, "b": b}).values())
        with self.assertRaises(ptk.OperationCancelled):
            FileCollect.apply(rows, mode="move", progress=lambda done, total: False)
        self.assertTrue(os.path.exists(a) and os.path.exists(b))

    def test_a_tile_set_moves_whole(self):
        self._file("ext", "rock.1001.png", data=b"1")
        self._file("ext", "rock.1002.png", data=b"2")
        _rows, landed = self._apply(
            {"a": os.path.join(self.tmp, "ext", "rock.<UDIM>.png")}, mode="move"
        )
        self.assertEqual(len(landed), 1)
        self.assertEqual(sorted(os.listdir(self.dest)), ["rock.1001.png", "rock.1002.png"])
        self.assertEqual(os.listdir(os.path.join(self.tmp, "ext")), [])

    def test_run_previews_writing_nothing_then_commits_the_plan_shown(self):
        wood = self._file("tex", "wood", "diffuse.png", data=b"wood")
        metal = self._file("tex", "metal", "diffuse.png", data=b"metal")
        gone = os.path.join(self.tmp, "gone", "lost.png")
        host = PureCollectHost({"wood": wood, "metal": metal, "lost": gone})
        said = []
        collect = FileCollect(
            host, report=lambda message, level, items: said.append((level, message))
        )
        rows = FileCollect.plan(collect.entries(list(host.paths)), self.dest)

        self.assertTrue(collect.run(rows, self.dest, dry_run=True))
        self.assertFalse(os.path.exists(self.dest), "a preview wrote")
        self.assertEqual(host.paths["wood"], wood)
        self.assertIn("Nothing has been written", said[-1][1])
        self.assertNotIn("Apply", said[-1][1], "a panel's button is the panel's to name")

        said.clear()
        self.assertTrue(collect.run(rows, self.dest))
        self.assertTrue(self._same(host.paths["wood"], self._at("wood", "diffuse.png")))
        self.assertTrue(
            self._same(host.paths["metal"], self._at("metal", "diffuse.png"))
        )
        self.assertEqual(host.paths["lost"], gone, "an unplaced reader keeps its path")
        self.assertTrue(
            any("still unresolved" in m and level == "warning" for level, m in said),
            said,
        )

    def test_a_rerun_over_the_collected_files_is_in_place(self):
        readers = {
            "wood": self._file("tex", "wood", "diffuse.png", data=b"wood"),
            "metal": self._file("tex", "metal", "diffuse.png", data=b"metal"),
        }
        _rows, landed = self._apply(readers)
        moved = {row["owner"]: row["destination"] for row in landed}

        again = self._plan(moved)

        self.assertEqual(
            {r["status"] for r in again.values()}, {FileCollect.IN_PLACE}
        )
        self.assertEqual({k: r["destination"] for k, r in again.items()}, moved)


class TestFlatPlan(_CollectCase):
    """``plan(flat=True)``: a repath to the destination itself (a Set
    Directory) -- every file directly in it, nothing kept apart."""

    def _flat(self, readers):
        collect = FileCollect(PureCollectHost(readers))
        entries = collect.entries(list(readers))
        return {r["owner"]: r for r in FileCollect.plan(entries, self.dest, flat=True)}

    def test_a_different_file_of_the_name_there_blocks_rather_than_keeps_apart(self):
        """Same size, other bytes: never a subfolder, never a rebind to theirs."""
        self._file("dest", "wood.tga", data=b"THEIRS!!")
        rows = self._flat({"a": self._file("ext", "wood.tga", data=b"OURS!!!!")})
        self.assertEqual(rows["a"]["status"], FileCollect.BLOCKED)
        self.assertIsNone(rows["a"]["destination"])

    def test_two_different_files_of_one_name_are_both_blocked(self):
        """Neither is the file of that name: the order they came in never decides."""
        rows = self._flat(
            {
                "wood": self._file("tex", "wood", "d.png", data=b"wood"),
                "metal": self._file("tex", "metal", "d.png", data=b"meta"),
            }
        )
        self.assertEqual(
            {r["status"] for r in rows.values()}, {FileCollect.BLOCKED}
        )

    def test_identical_files_of_a_name_still_share_one_copy(self):
        rows = self._flat(
            {
                "a": self._file("one", "d.png", data=b"same"),
                "b": self._file("two", "d.png", data=b"same"),
            }
        )
        self.assertEqual(
            [r["status"] for r in rows.values()],
            [FileCollect.RELOCATE, FileCollect.REUSE],
        )
        self.assertTrue(self._same(rows["b"]["destination"], self._at("d.png")))

    def test_a_file_below_the_destination_is_brought_up_to_it(self):
        rows = self._flat({"a": self._file("dest", "sub", "d.png")})
        self.assertEqual(rows["a"]["status"], FileCollect.RELOCATE)
        self.assertTrue(self._same(rows["a"]["destination"], self._at("d.png")))

    def test_a_file_already_there_stays_in_place(self):
        there = self._file("dest", "d.png")
        rows = self._flat({"a": there})
        self.assertEqual(rows["a"]["status"], FileCollect.IN_PLACE)
        self.assertTrue(self._same(rows["a"]["destination"], there))

    def test_a_destination_aliasing_the_source_folder_is_in_place_as_spelled(self):
        """BACKLOG 2026-10-23 (S1): a junction naming the source's own folder
        held a same-sized "copy" that was the source itself, and a move removed
        it -- the only copy. In place, it is never moved; the reader reads it
        through the folder it was pointed at."""
        real = os.path.join(self.tmp, "real")
        src = self._file("real", "metal.png", data=b"ONLY COPY")
        self.dest = self._junction(os.path.join(self.tmp, "alias"), real)
        rows = self._flat({"a": src})
        self.assertEqual(rows["a"]["status"], FileCollect.IN_PLACE)
        self.assertTrue(self._same(rows["a"]["destination"], self._at("metal.png")))

        FileCollect.apply(list(rows.values()), mode="move")
        self.assertEqual(self._read(src), b"ONLY COPY")

    def test_a_flat_move_of_a_blocked_file_takes_nothing(self):
        theirs = self._file("dest", "wood.tga", data=b"THEIRS!!")
        src = self._file("ext", "wood.tga", data=b"OURS!!!!")
        rows = list(self._flat({"a": src}).values())
        self.assertEqual(FileCollect.apply(rows, mode="move"), [])
        self.assertEqual(self._read(src), b"OURS!!!!")
        self.assertEqual(self._read(theirs), b"THEIRS!!")

    def test_a_move_says_which_files_it_copied_for_another_reader(self):
        """A move that quietly copied left an artist asking why the original
        is still there."""
        shared = self._file("tex", "wood.png")

        class Host(PureCollectHost):
            def read_elsewhere(self, nodes):
                return [shared]

        rows = FileCollect(Host({"inside": shared})).plan_nodes(["inside"], self.dest)
        landed = FileCollect.apply(rows, mode="move")
        said = [m for _level, m, _items in FileCollect.outcome(rows, landed, "move")]
        self.assertTrue(any("left in place rather than moved" in m for m in said), said)
        said = [m for _level, m, _items in FileCollect.outcome(rows, landed, "copy")]
        self.assertFalse(any("left in place rather than moved" in m for m in said), said)

    def test_a_set_and_one_of_its_tiles_both_land_flat(self):
        """Two readers -- a tile set and one of its tiles -- carry different
        names, so the set's claim on the tile blocked the tile's reader, and
        the set's move then deleted the tile under it."""
        self._file("ext", "rock.1001.png", data=b"1")
        self._file("ext", "rock.1002.png", data=b"2")
        tile = os.path.join(self.tmp, "ext", "rock.1001.png")
        rows = self._flat(
            {"set": os.path.join(self.tmp, "ext", "rock.<UDIM>.png"), "tile": tile}
        )
        landed = FileCollect.apply(list(rows.values()), mode="move")
        self.assertEqual(sorted(r["owner"] for r in landed), ["set", "tile"])
        self.assertTrue(self._same(rows["tile"]["destination"], self._at("rock.1001.png")))
        self.assertEqual(sorted(os.listdir(self.dest)), ["rock.1001.png", "rock.1002.png"])


class TestApplyGuards(_CollectCase):
    """What a move never takes, and what a failed row leaves behind."""

    def _rows(self, readers, **plan):
        collect = FileCollect(PureCollectHost(readers))
        return FileCollect.plan(collect.entries(list(readers)), self.dest, **plan)

    def test_a_move_keeps_a_file_a_row_that_did_not_land_reads(self):
        """A reader left on its path keeps its file, whatever another reader
        of it did."""
        path = self._file("ext", "shared.png")
        rows = self._rows({"a": path, "b": path})
        rows[1]["status"] = FileCollect.BLOCKED  # held back, as a namesake would
        landed = FileCollect.apply(rows, mode="move")
        self.assertEqual([r["owner"] for r in landed], ["a"])
        self.assertTrue(os.path.exists(path), "the reader left behind was stranded")

    def test_a_set_whose_copy_fails_part_way_leaves_no_tile_behind(self):
        """A set lands whole or not at all: the tiles copied before one failed
        stayed in the destination, foreign to a set of that name placed there
        later."""
        self._file("ext", "rock.1001.png", data=b"1")
        self._file("ext", "rock.1002.png", data=b"2")
        rows = self._rows({"a": os.path.join(self.tmp, "ext", "rock.<UDIM>.png")})

        def flaky(sources, folder, mode, progress=None):
            first = sorted(sources)[0]
            target = os.path.join(folder, os.path.basename(first))
            with open(first, "rb") as src, open(target, "wb") as dst:
                dst.write(src.read())
            return [(first, target)]  # the second tile failed

        self.assertEqual(FileCollect.apply(rows, mode="move", copy=flaky), [])
        self.assertEqual(os.listdir(self.dest), [])
        self.assertEqual(len(os.listdir(os.path.join(self.tmp, "ext"))), 2)

    def test_a_move_keeps_a_file_an_outside_reader_spells_another_way(self):
        """The keep compared spellings: an outside reader storing the file
        through another name for its folder (a junction, a mapped drive, the
        share's UNC path) was stranded."""
        real = os.path.join(self.tmp, "real")
        src = self._file("real", "wood.png")
        alias = self._junction(os.path.join(self.tmp, "alias"), real)
        aliased = os.path.join(alias, "wood.png")

        class Host(PureCollectHost):
            def read_elsewhere(self, nodes):
                return [aliased]

        rows = FileCollect(Host({"inside": src})).plan_nodes(["inside"], self.dest)
        self.assertEqual([r["keep_source"] for r in rows], [True])
        FileCollect.apply(rows, mode="move")
        self.assertTrue(os.path.exists(src), "the outside reader was stranded")

    def test_a_folder_that_cannot_be_made_lands_nothing_and_raises_nothing(self):
        blocker = self._file("blocker", data=b"a file, not a folder")
        src = self._file("ext", "d.png")
        rows = FileCollect.plan(
            [{"owner": "a", "source": src}], os.path.join(blocker, "dest")
        )
        self.assertEqual(FileCollect.apply(rows, mode="move"), [])
        self.assertTrue(os.path.exists(src))


class TestSetDirectory(_CollectCase):
    """``FileCollect.set_directory``: Set Directory's file work over a host."""

    def _run(self, readers, mode, **kwargs):
        host = PureCollectHost(readers)
        done = FileCollect(host).set_directory(list(readers), self.dest, mode, **kwargs)
        return host, done

    def test_rewrite_points_only_at_a_file_that_is_there(self):
        there = self._file("dest", "a.png")
        host, done = self._run(
            {"a": self._file("ext", "a.png"), "b": self._file("ext", "b.png")}, "rewrite"
        )
        self.assertEqual([r["owner"] for r in done["repath"]], ["a"])
        self.assertTrue(self._same(done["repath"][0]["destination"], there))
        self.assertEqual(done["absent"], ["b"])
        self.assertEqual(done["rows"], [], "rewrite plans no Collect")

    def test_allow_missing_points_at_nothing_on_purpose(self):
        _host, done = self._run({"b": self._file("ext", "b.png")}, "rewrite", allow_missing=True)
        self.assertEqual([r["owner"] for r in done["repath"]], ["b"])
        self.assertEqual(done["absent"], [])

    def test_a_move_brings_the_file_and_never_takes_a_namesakes_place(self):
        self._file("dest", "wood.tga", data=b"THEIRS!!")
        ours = self._file("ext", "wood.tga", data=b"OURS!!!!")
        moss = self._file("ext", "moss.png", data=b"MOSS")
        _host, done = self._run({"wood": ours, "moss": moss}, "move")
        self.assertEqual([r["owner"] for r in done["repath"]], ["moss"])
        self.assertEqual(self._read(ours), b"OURS!!!!")
        self.assertFalse(os.path.exists(moss))
        self.assertTrue(any(level == "warning" for level, _m, _i in done["report"]))

    def test_a_reader_whose_file_is_nowhere_follows_the_rewrite_rule(self):
        there = self._file("dest", "lost.png")
        gone = os.path.join(self.tmp, "gone", "lost.png")
        _host, done = self._run({"lost": gone}, "copy")
        self.assertEqual([r["owner"] for r in done["repath"]], ["lost"])
        self.assertTrue(self._same(done["repath"][0]["destination"], there))

    def test_an_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            FileCollect().set_directory([], self.dest, "bogus")


class TestFinishMove(_CollectCase):
    """``FileCollect.finish_move``: the take-away of a move whose copies a host
    placed itself."""

    def test_a_source_goes_once_its_copy_holds_it(self):
        src = self._file("ext", "a.png", data=b"A")
        dst = self._file("dest", "a.png", data=b"A")
        taken, kept = FileCollect.finish_move([(src, dst), (src, dst)])
        self.assertEqual((len(taken), kept), (1, []))
        self.assertFalse(os.path.exists(src))

    def test_a_file_a_reader_still_reads_stays_however_spelled(self):
        real = os.path.join(self.tmp, "real")
        src = self._file("real", "a.png", data=b"A")
        dst = self._file("dest", "a.png", data=b"A")
        alias = self._junction(os.path.join(self.tmp, "alias"), real)
        taken, kept = FileCollect.finish_move([(src, dst)], keep=[os.path.join(alias, "a.png")])
        self.assertEqual((taken, len(kept)), ([], 1))
        self.assertTrue(os.path.exists(src))

    def test_never_a_source_whose_copy_differs_or_is_itself(self):
        src = self._file("ext", "a.png", data=b"OURS")
        theirs = self._file("dest", "a.png", data=b"THEM")
        real = os.path.join(self.tmp, "real")
        only = self._file("real", "b.png", data=b"ONLY")
        alias = self._junction(os.path.join(self.tmp, "alias"), real)
        taken, _kept = FileCollect.finish_move(
            [(src, theirs), (only, os.path.join(alias, "b.png"))]
        )
        self.assertEqual(taken, [])
        self.assertTrue(os.path.exists(src) and os.path.exists(only))

    def test_a_tile_set_goes_by_its_tiles(self):
        for tile in ("1001", "1002"):
            self._file("ext", f"rock.{tile}.png", data=tile.encode())
            self._file("dest", f"rock.{tile}.png", data=tile.encode())
        taken, _kept = FileCollect.finish_move(
            [(os.path.join(self.tmp, "ext", "rock.<UDIM>.png"), self._at("rock.<UDIM>.png"))]
        )
        self.assertEqual(len(taken), 2)
        self.assertEqual(os.listdir(os.path.join(self.tmp, "ext")), [])
