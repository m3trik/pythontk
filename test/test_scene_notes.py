#!/usr/bin/python
# coding=utf-8
"""Tests for ``pythontk.SceneNotes`` (``file_utils/scene_notes.py``).

The Reference Manager's Notes column, kept with the PROJECT: ONE
``<project>/<data rule>/scene_notes.json`` per project, keyed by each scene's
project-relative path, and the same file both DCCs' panels read and write --
not a ``<scene>.metadata.json`` beside every listed scene (file-handling audit
2026-10-13; blendertk kept its notes in the panel's settings, so neither DCC
saw the other's). Hoisted from the two DCC packages' identical copies.
Real files in a scratch folder are all a test needs -- no host.
"""

import json
import os
import unittest
import unittest.mock

import pythontk as ptk

from conftest import BaseTestCase


class _ProjectCase(BaseTestCase):
    """A marked project holding one scene."""

    def setUp(self):
        self.tmp_store = ptk.TempArtifacts("ptk_scene_notes")
        self.addCleanup(self.tmp_store.cleanup)
        self.root = self.tmp_store.dir_path()
        self.proj = os.path.join(self.root, "proj")
        os.makedirs(os.path.join(self.proj, "scenes", "hero"))
        with open(os.path.join(self.proj, "workspace.mel"), "w") as fh:
            fh.write('workspace -fr "scene" "scenes";\n')
        self.scene = os.path.join(self.proj, "scenes", "hero", "hero_v01.ma")
        open(self.scene, "w").close()
        self.notes = ptk.SceneNotes()

    def _store(self):
        return os.path.join(self.proj, "data", "scene_notes.json")


class TestSceneNotesStore(_ProjectCase):
    def test_a_note_lands_in_the_projects_one_file(self):
        self.notes.set(self.scene, "lighting pass")

        self.assertEqual(
            ptk.FileUtils.read_json(self._store()),
            {"version": 1, "notes": {"scenes/hero/hero_v01.ma": "lighting pass"}},
        )
        self.assertFalse(os.path.exists(self.scene + ".metadata.json"))
        self.assertEqual(ptk.SceneNotes().get(self.scene), "lighting pass")

    def test_the_store_path_names_the_projects_file(self):
        self.assertEqual(
            os.path.normcase(self.notes.store_path(self.scene)),
            os.path.normcase(self._store()),
        )

    def test_the_data_rule_names_the_folder(self):
        with open(os.path.join(self.proj, "workspace.mel"), "a") as fh:
            fh.write('workspace -fr "data" "meta";\n')
        self.notes.set(self.scene, "x")
        self.assertTrue(
            os.path.isfile(os.path.join(self.proj, "meta", "scene_notes.json"))
        )

    def test_a_key_differing_only_in_case_finds_the_note(self):
        self.notes.set(self.scene, "x")
        data = ptk.FileUtils.read_json(self._store())
        data["notes"] = {"Scenes/Hero/HERO_v01.ma": "x"}
        ptk.FileUtils.write_json(self._store(), data)
        if os.path.normcase("A") == "A":  # a case-sensitive file system
            self.skipTest("paths compare with case here")
        self.assertEqual(ptk.SceneNotes().get(self.scene), "x")

    def test_move_and_remove_follow_the_scene(self):
        self.notes.set(self.scene, "keep me")
        renamed = os.path.join(self.proj, "scenes", "hero", "hero_v02.ma")
        self.notes.move(self.scene, renamed)
        self.assertEqual(self.notes.get(renamed), "keep me")
        self.assertEqual(self.notes.get(self.scene), "")
        self.notes.remove(renamed)
        self.assertEqual(ptk.FileUtils.read_json(self._store())["notes"], {})

    def test_an_empty_note_removes_it(self):
        self.notes.set(self.scene, "x")
        self.notes.set(self.scene, "")
        self.assertEqual(ptk.FileUtils.read_json(self._store())["notes"], {})

    def test_a_scene_no_project_holds_has_no_note_and_refuses_one(self):
        self.assertEqual(self.notes.get(""), "")
        self.assertIsNone(self.notes.store_path(""))
        with self.assertRaises(OSError):
            self.notes.set("", "x")

    def test_an_unmarked_scene_resolves_under_the_session_root(self):
        loose = os.path.join(self.root, "loose")
        os.makedirs(os.path.join(loose, "shots"))
        scene = os.path.join(loose, "shots", "a.blend")
        open(scene, "w").close()
        notes = ptk.SceneNotes(session_root=loose)
        notes.set(scene, "pinned")
        self.assertEqual(
            ptk.FileUtils.read_json(os.path.join(loose, "data", "scene_notes.json"))[
                "notes"
            ],
            {"shots/a.blend": "pinned"},
        )


class TestUnreadableStore(_ProjectCase):
    """A store that exists but cannot be read (a sharing lock, corrupt JSON, a
    non-object document) is NOT an empty store: writing over it would erase
    every other scene's note (pre-release review 2026-10-13)."""

    def _seed(self, other="scenes/other.ma", text="keep"):
        os.makedirs(os.path.dirname(self._store()), exist_ok=True)
        ptk.FileUtils.write_json(self._store(), {"version": 1, "notes": {other: text}})

    @staticmethod
    def _locked():
        """``read_json`` failing as a sharing lock makes it (returns *default*)."""
        return unittest.mock.patch.object(
            ptk.FileUtils,
            "read_json",
            side_effect=lambda path, default=None, **kw: default,
        )

    def _raw(self):
        with open(self._store(), encoding="utf-8") as fh:
            return fh.read()

    def test_set_over_a_corrupt_store_raises_and_writes_nothing(self):
        os.makedirs(os.path.dirname(self._store()))
        with open(self._store(), "w", encoding="utf-8") as fh:
            fh.write('{"version": 1, "notes": {"scenes/other.ma": "ke')
        before = self._raw()
        with self.assertRaises(OSError):
            self.notes.set(self.scene, "x")
        self.assertEqual(self._raw(), before)
        self.assertEqual(self.notes.get(self.scene), "")

    def test_set_over_a_locked_store_raises_and_keeps_the_other_notes(self):
        self._seed()
        before = self._raw()
        with self._locked(), self.assertRaises(OSError):
            self.notes.set(self.scene, "x")
        self.assertEqual(self._raw(), before)

    def test_a_non_object_store_is_refused(self):
        os.makedirs(os.path.dirname(self._store()))
        with open(self._store(), "w", encoding="utf-8") as fh:
            json.dump(["not", "a", "store"], fh)
        with self.assertRaises(OSError):
            self.notes.set(self.scene, "x")
        self.assertEqual(ptk.FileUtils.read_json(self._store()), ["not", "a", "store"])

    def test_remove_over_an_unreadable_store_writes_nothing(self):
        self._seed(other="scenes/hero/hero_v01.ma")
        before = self._raw()
        with self._locked():
            self.notes.remove(self.scene)  # best-effort: logged, never raises
        self.assertEqual(self._raw(), before)

    def test_a_failed_read_is_not_cached(self):
        self._seed(other="scenes/hero/hero_v01.ma", text="mine")
        with self._locked():
            self.assertEqual(self.notes.get(self.scene), "")
        # Same file, same mtime: the lock is gone, the note must read again.
        self.assertEqual(self.notes.get(self.scene), "mine")

    def test_move_writes_the_new_note_before_dropping_the_old(self):
        self.notes.set(self.scene, "keep me")
        renamed = os.path.join(self.proj, "scenes", "hero", "hero_v02.ma")
        with unittest.mock.patch.object(
            ptk.SceneNotes, "set", side_effect=OSError("disk full")
        ):
            self.notes.move(self.scene, renamed)  # logged
        self.assertEqual(self.notes.get(self.scene), "keep me")


class TestLegacySidecar(_ProjectCase):
    """A note kept in ``ptk.Metadata``'s ``<scene>.metadata.json`` (before
    2026-10-13) is imported on its first read; the sidecar goes once the store
    holds it."""

    def test_a_legacy_sidecar_note_is_imported_then_its_sidecar_removed(self):
        sidecar = self.scene + ".metadata.json"
        with open(sidecar, "w") as fh:
            json.dump({"Comments": "from the sidecar"}, fh)

        self.assertEqual(self.notes.get(self.scene), "from the sidecar")

        self.assertFalse(os.path.exists(sidecar), "the sidecar outlived its import")
        self.assertEqual(
            ptk.FileUtils.read_json(self._store())["notes"],
            {"scenes/hero/hero_v01.ma": "from the sidecar"},
        )

    def test_a_sidecar_holding_more_than_the_note_keeps_the_rest(self):
        sidecar = self.scene + ".metadata.json"
        with open(sidecar, "w") as fh:
            json.dump({"Comments": "note", "Author": "me"}, fh)
        self.assertEqual(self.notes.get(self.scene), "note")
        self.assertEqual(ptk.FileUtils.read_json(sidecar), {"Author": "me"})

    def test_a_non_object_sidecar_is_left_alone(self):
        sidecar = self.scene + ".metadata.json"
        with open(sidecar, "w") as fh:
            json.dump(["a", "list"], fh)
        self.notes.set(self.scene, "x")  # must not crash on ``.items()``
        self.assertEqual(ptk.FileUtils.read_json(sidecar), ["a", "list"])

    def test_an_unreadable_sidecar_is_never_removed(self):
        sidecar = self.scene + ".metadata.json"
        with open(sidecar, "w") as fh:
            fh.write('{"Comments": "half a no')
        self.notes.set(self.scene, "x")
        self.assertTrue(os.path.isfile(sidecar), "an unreadable sidecar was deleted")


class TestImportNotes(_ProjectCase):
    """Notes a host kept elsewhere, keyed by absolute path (blendertk's panel
    settings before 2026-10-13), moved into each scene's project store."""

    def test_a_note_is_moved_into_its_projects_store(self):
        kept = self.notes.import_notes({self.scene: "from settings"})
        self.assertEqual(kept, {})
        self.assertEqual(ptk.SceneNotes().get(self.scene), "from settings")

    def test_a_note_whose_scene_is_gone_is_kept(self):
        gone = os.path.join(self.proj, "scenes", "gone.ma")
        self.assertEqual(self.notes.import_notes({gone: "orphan"}), {gone: "orphan"})

    def test_a_note_the_store_already_holds_wins(self):
        self.notes.set(self.scene, "current")
        self.assertEqual(self.notes.import_notes({self.scene: "older"}), {})
        self.assertEqual(self.notes.get(self.scene), "current")

    def test_a_note_that_cannot_be_written_is_kept(self):
        loose = os.path.join(self.root, "nowhere", "a.ma")
        with unittest.mock.patch.object(
            ptk.SceneNotes, "set", side_effect=OSError("read-only")
        ):
            kept = self.notes.import_notes({self.scene: "x", loose: ""})
        self.assertEqual(kept, {self.scene: "x"})


if __name__ == "__main__":
    unittest.main()
