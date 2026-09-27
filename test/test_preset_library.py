# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.PresetLibrary — every preset store under one root."""

import json
import os
import shutil
import time
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest import mock

from pythontk.core_utils.presets.library import (
    AUTO_BACKUP_KEEP,
    BACKUPS_DIR,
    BUNDLE_FORMAT,
    PresetLibrary,
)
from pythontk.core_utils.presets.store import (
    Codec,
    PresetReadOnlyError,
    PresetStore,
)
from pythontk.core_utils.user_config import CONFIG_ROOT_ENV_VAR


class _LibraryCase(unittest.TestCase):
    """A scratch presets root under ``test/temp_tests/``, set as the live root."""

    def setUp(self):
        base = os.path.join(os.path.dirname(__file__), "temp_tests")
        # Unique per run: a folder a sync client kept from being deleted, or a
        # concurrent run of the same test (routine with several sessions), must
        # never be this run's root -- both surfaced as FileExistsError here.
        run = uuid.uuid4().hex[:6]
        self.root = os.path.join(base, f"preset_lib_{self._testMethodName}_{run}")
        self.other = self.root + "_other"
        for d in (self.root, self.other):
            os.makedirs(d)
            self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        prev = os.environ.get(CONFIG_ROOT_ENV_VAR)
        os.environ[CONFIG_ROOT_ENV_VAR] = self.root
        self.addCleanup(
            lambda: (
                os.environ.pop(CONFIG_ROOT_ENV_VAR, None)
                if prev is None
                else os.environ.__setitem__(CONFIG_ROOT_ENV_VAR, prev)
            )
        )
        self.lib = PresetLibrary()
        self.bundle = os.path.join(self.other, "bundle.zip")

    # Stores as the tools build them: default dirs under the (redirected) root.
    def exporter(self, root=None):
        user_dir = os.path.join(root, "mayatk", "scene_exporter") if root else None
        return PresetStore("scene_exporter", "mayatk", user_dir=user_dir)

    def fbx(self):
        return PresetStore("fbx_presets", "blendertk")

    def snapshot(self, root):
        out = {}
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                p = os.path.join(dirpath, f)
                out[p] = os.path.getmtime(p)
        return out


class ScanTest(_LibraryCase):
    def setUp(self):
        super().setUp()
        self.exporter().save("Unity", {"_meta": {"version": 1}, "a": 1})
        self.fbx().save("game", {"axis": "Y"})  # raw payload: no _meta
        # A folder written before markers/sidecars existed (legacy panel preset).
        legacy = Path(self.root, "mayatk", "color_manager")
        legacy.mkdir(parents=True)
        (legacy / "default.json").write_text(json.dumps({"_meta": {}, "c": 1}))
        # A settings file that is NOT a preset store.
        shots = Path(self.root, "shots")
        shots.mkdir()
        (shots / "prefs.json").write_text(json.dumps({"fit_mode": "x"}))
        # Library housekeeping folders are never stores.
        Path(self.root, ".backups").mkdir()
        Path(self.root, ".collections").mkdir()

    def test_finds_marked_sidecar_and_legacy_stores_but_not_settings(self):
        keys = [d.key for d in self.lib.domains()]
        self.assertEqual(
            keys,
            ["blendertk/fbx_presets", "mayatk/color_manager", "mayatk/scene_exporter"],
        )

    def test_entries_carry_identity_and_legacy_presets_have_none_yet(self):
        by_name = {e.name: e for e in self.lib.entries()}
        self.assertTrue(by_name["Unity"].id)
        self.assertEqual(by_name["Unity"].label, "Unity")
        self.assertIsNone(by_name["default"].id)
        self.assertEqual(by_name["default"].tier, "user")

    def test_scanning_and_planning_write_nothing(self):
        self.lib.export(self.bundle)  # (export may stamp; take the snapshot after)
        before = self.snapshot(self.root)
        self.lib.domains()
        self.lib.entries()
        self.lib.plan_import(self.bundle)
        self.assertEqual(self.snapshot(self.root), before)

    def test_builtins_are_listed_from_the_marker(self):
        shipped = Path(self.root + "_other", "shipped")
        shipped.mkdir(parents=True)
        (shipped / "stock.json").write_text(json.dumps({"s": 1}))
        store = PresetStore("lightmap", "mayatk", builtin_dir=shipped)
        store.save("mine", {"s": 2})
        entries = {e.name: e for e in self.lib.entries("mayatk/lightmap")}
        self.assertEqual(entries["stock"].tier, "builtin")
        self.assertTrue(entries["stock"].read_only)
        self.assertFalse(entries["mine"].read_only)

    def test_a_pattern_names_a_folder_and_takes_its_stores_with_it(self):
        keys = lambda **kw: [d.key for d in self.lib.domains(**kw)]  # noqa: E731
        self.assertEqual(
            keys(inc="mayatk"), ["mayatk/color_manager", "mayatk/scene_exporter"]
        )
        self.assertEqual(keys(inc="MayaTK, blendertk"), keys(exc="shots"))
        self.assertEqual(keys(inc="*/scene_*"), ["mayatk/scene_exporter"])
        self.assertEqual(
            keys(inc="mayatk", exc="mayatk/color_manager"), ["mayatk/scene_exporter"]
        )
        self.assertEqual(keys(inc="maya"), [], "a folder name, not a prefix")

    def test_entries_are_scoped_the_same_way(self):
        names = [e.name for e in self.lib.entries(inc=["blendertk"])]
        self.assertEqual(names, ["game"])
        names = sorted(e.name for e in self.lib.entries(exc="blendertk"))
        self.assertEqual(names, ["Unity", "default"])

    def test_a_yaml_store_is_manageable_without_its_codec(self):
        codec = Codec(".yaml", json.loads, json.dumps)  # stands in for a YAML codec
        PresetStore("manifest_behaviors", "shots", codec=codec).save("b", {"x": 1})
        domain = self.lib.domain("shots/manifest_behaviors")
        self.assertEqual(domain.ext, ".yaml")
        [entry] = self.lib.entries("shots/manifest_behaviors")
        self.assertEqual(entry.path.suffix, ".yaml")
        self.assertEqual(self.lib.set_read_only([entry]), 1)


class EditTest(_LibraryCase):
    def setUp(self):
        super().setUp()
        self.exporter().save("Unity", {"a": 1})

    def unity(self):
        return self.lib.entry("mayatk/scene_exporter", "Unity")

    def test_lock_blocks_the_tool_save_and_unlock_releases_it(self):
        self.lib.set_read_only([self.unity()])
        self.assertTrue(self.unity().read_only)
        with self.assertRaises(PresetReadOnlyError):
            self.exporter().save("Unity", {"a": 2})
        self.lib.set_read_only([self.unity()], False)
        self.exporter().save("Unity", {"a": 2})

    def test_duplicate_makes_an_unlocked_untagged_copy(self):
        c = self.lib.create_collection("Studio")
        self.lib.assign([self.unity()], c["id"])
        self.lib.set_read_only([self.unity()])
        copy = self.lib.duplicate(self.unity())
        self.assertEqual(copy.name, "Unity copy")
        self.assertFalse(copy.read_only)
        self.assertIsNone(copy.collection)
        self.assertNotEqual(copy.id, self.unity().id)
        self.assertEqual(self.exporter().load("Unity copy"), {"a": 1})

    def test_rename_and_delete_respect_the_lock(self):
        self.lib.set_read_only([self.unity()])
        self.assertFalse(self.lib.rename(self.unity(), "U2"))
        self.assertEqual(self.lib.delete([self.unity()]), 0)
        self.assertEqual(self.lib.delete([self.unity()], force=True), 1)

    def test_renaming_the_active_preset_moves_the_pointer_whatever_its_name(self):
        # A tool sets the pointer with the name as typed; the Preset Editor
        # renames by the file stem. tentacle applies the pointed-at preset at
        # launch, and a pointer left behind fell back to the default bindings.
        store = self.exporter()
        store.save("m3trik (laptop)", {"a": 1})
        store.active = "m3trik (laptop)"
        [entry] = [
            e
            for e in self.lib.entries("mayatk/scene_exporter")
            if e.label == "m3trik (laptop)"
        ]
        self.assertTrue(self.lib.rename(entry, "m3trik laptop"))
        self.assertEqual(store.load(store.active), {"a": 1})

    def test_tags_replace_and_clear(self):
        self.lib.set_tags([self.unity()], ["unity", " web ", ""])
        self.assertEqual(self.unity().tags, ("unity", "web"))
        self.lib.set_tags([self.unity()], [])
        self.assertEqual(self.unity().tags, ())

    def test_assign_to_an_unknown_collection_raises(self):
        with self.assertRaises(KeyError):
            self.lib.assign([self.unity()], "nope")

    def test_a_taken_collection_name_is_refused_not_duplicated(self):
        self.lib.create_collection("Look Dev")
        with self.assertRaises(ValueError):
            self.lib.create_collection(" look dev ")
        self.assertEqual(len(self.lib.collections()), 1)

    def test_a_name_is_taken_ignoring_case_and_edge_spaces(self):
        studio = self.lib.create_collection("Studio")
        self.assertEqual(self.lib.collection_named("  STUDIO ")["id"], studio["id"])
        self.assertIsNone(self.lib.collection_named("studio", exclude=studio["id"]))
        self.assertIsNone(self.lib.collection_named("Other"))

    def test_renaming_onto_a_taken_name_is_refused(self):
        self.lib.create_collection("Look Dev")
        other = self.lib.create_collection("Other")
        with self.assertRaises(ValueError):
            self.lib.update_collection(other["id"], name="LOOK DEV")
        # Re-casing its own name is not a clash.
        renamed = self.lib.update_collection(other["id"], name="other")
        self.assertEqual(renamed["name"], "other")

    def test_a_namesake_blocks_a_rename_whichever_is_found_first(self):
        # An install can put two same-named collections side by side; renaming
        # one must see the OTHER, not stop at itself.
        first = self.lib.create_collection("Alpha")
        second = self.lib.create_collection("Beta")
        self.lib._write_collection({**second, "name": "Alpha"})  # installed beside
        with self.assertRaises(ValueError):
            self.lib.update_collection(first["id"], name="alpha")

    def test_any_collection_name_makes_an_id_its_own_bundle_accepts(self):
        # The import side refuses ids outside the slug grammar (they name a file
        # under .collections/), so every id create_collection makes must fit it.
        bundle = os.path.join(self.other, "b.zip")
        for name in ("My_Set", "a__b", "Ünïcode Set", "日本語"):
            with self.subTest(name=name):
                cid = self.lib.create_collection(name)["id"]
                self.lib.export(bundle, collection=cid)
                self.assertEqual(PresetLibrary.read_header(bundle)["id"], cid)


class HiddenAndDescriptionTest(_LibraryCase):
    """Hide presets from their tools' dropdowns; say what a preset is for."""

    def setUp(self):
        super().setUp()
        shipped = Path(self.other, "shipped")
        shipped.mkdir()
        (shipped / "Stock.json").write_text(
            json.dumps({"_meta": {"description": "The shipped look."}, "a": 0})
        )
        self.store = PresetStore("scene_exporter", "mayatk", builtin_dir=shipped)
        self.store.save("Unity", {"a": 1})

    def entries(self):
        return {e.name: e for e in self.lib.entries("mayatk/scene_exporter")}

    def test_set_hidden_takes_builtins_too_and_entries_carry_it(self):
        e = self.entries()
        self.assertFalse(e["Stock"].hidden or e["Unity"].hidden)
        self.assertEqual(self.lib.set_hidden([e["Stock"], e["Unity"]]), 2)
        e = self.entries()
        self.assertTrue(e["Stock"].hidden and e["Unity"].hidden)
        self.assertTrue(self.store.is_hidden("Stock"))
        self.assertTrue(self.lib.entry("mayatk/scene_exporter", "Stock").hidden)
        self.assertEqual(self.lib.set_hidden([e["Stock"]], False), 1)
        self.assertFalse(self.entries()["Stock"].hidden)
        self.assertTrue(self.entries()["Unity"].hidden)

    def test_a_hidden_preset_follows_a_library_rename(self):
        self.lib.set_hidden([self.entries()["Unity"]])
        self.assertTrue(self.lib.rename(self.entries()["Unity"], "Unity 6"))
        self.assertTrue(self.entries()["Unity 6"].hidden)

    def test_descriptions_are_the_users_to_write_and_a_builtins_to_read(self):
        e = self.entries()
        self.assertEqual(e["Stock"].description, "The shipped look.")
        self.assertEqual(e["Unity"].description, "")
        # Built-ins are skipped, as for tags.
        self.assertEqual(
            self.lib.set_description([e["Unity"], e["Stock"]], "  For Unity. "), 1
        )
        self.assertEqual(self.entries()["Unity"].description, "For Unity.")
        self.assertEqual(self.entries()["Stock"].description, "The shipped look.")
        self.assertTrue(self.lib.rename(self.entries()["Unity"], "Unity 6"))
        self.assertEqual(self.entries()["Unity 6"].description, "For Unity.")
        self.lib.set_description([self.entries()["Unity 6"]], " ")
        self.assertNotIn("description", self.store.info("Unity 6"))


class CollectionRoundTripTest(_LibraryCase):
    """Author exports a collection; an artist (second root) installs and updates."""

    def setUp(self):
        super().setUp()
        store = self.exporter()
        store.save("Unity", {"a": 1})
        store.save("WebXR", {"a": 2})
        store.save("Personal", {"a": 3})
        self.c = self.lib.create_collection("Acme Standard", "house presets")
        members = [
            e for e in self.lib.entries("mayatk/scene_exporter") if e.name != "Personal"
        ]
        self.lib.assign(members, self.c["id"])
        self.lib.set_read_only(members)
        self.artist = PresetLibrary(self.other)

    def publish(self):
        return self.lib.export(self.bundle, collection=self.c["id"])

    def install(self):
        plan = self.artist.plan_import(self.bundle)
        self.artist.apply(plan)
        return plan

    def test_bundle_holds_header_payloads_and_sidecars_only(self):
        self.publish()
        with zipfile.ZipFile(self.bundle) as zf:
            names = sorted(zf.namelist())
            header = json.loads(zf.read("collection.json"))
        self.assertEqual(header["format"], BUNDLE_FORMAT)
        self.assertEqual(header["kind"], "collection")
        self.assertEqual(header["version"], 1)
        self.assertIn("pythontk", header["packages"])
        self.assertEqual(
            names,
            [
                "collection.json",
                "presets/mayatk/scene_exporter/.Unity.preset",
                "presets/mayatk/scene_exporter/.WebXR.preset",
                "presets/mayatk/scene_exporter/Unity.json",
                "presets/mayatk/scene_exporter/WebXR.json",
            ],
        )

    def test_install_brings_presets_locked_tagged_and_the_header(self):
        self.publish()
        plan = self.install()
        self.assertEqual(plan.counts(), {"new": 2})
        [unity] = [
            e for e in self.artist.entries("mayatk/scene_exporter") if e.name == "Unity"
        ]
        self.assertTrue(unity.read_only)
        self.assertEqual(unity.collection, self.c["id"])
        self.assertFalse(self.artist.is_modified(unity))
        installed = self.artist.collection(self.c["id"])
        self.assertEqual(installed["version"], 1)
        self.assertEqual(installed["name"], "Acme Standard")

    def test_reimporting_the_same_version_is_all_identical(self):
        self.publish()
        self.install()
        plan = self.artist.plan_import(self.bundle)
        self.assertEqual(plan.counts(), {"identical": 2})
        self.assertEqual(plan.pending(), [])

    def test_a_new_version_updates_unedited_members_even_when_locked(self):
        self.publish()
        self.install()
        self.exporter().save("Unity", {"a": 10}, force=True)
        self.publish()
        plan = self.artist.plan_import(self.bundle)
        statuses = {i.name: i.status for i in plan.items}
        self.assertEqual(statuses, {"Unity": "update", "WebXR": "identical"})
        self.artist.apply(plan)
        artist_store = self.exporter(self.other)
        self.assertEqual(artist_store.load("Unity"), {"a": 10})
        self.assertTrue(artist_store.is_read_only("Unity"))
        self.assertEqual(self.artist.collection(self.c["id"])["version"], 2)

    def test_a_locally_edited_member_is_a_conflict_that_defaults_to_keep_mine(self):
        self.publish()
        self.install()
        self.exporter(self.other).save("Unity", {"a": 99}, force=True)
        self.exporter().save("Unity", {"a": 10}, force=True)
        self.publish()
        [item] = [
            i for i in self.artist.plan_import(self.bundle).items if i.name == "Unity"
        ]
        self.assertEqual(item.status, "conflict")
        self.assertEqual(item.action, "skip")
        self.assertIn("edited", item.reason)
        self.assertEqual(item.choices(), ("skip", "replace", "keep_both"))

    def test_keep_both_saves_the_incoming_copy_under_a_new_name(self):
        self.publish()
        self.exporter(self.other).save("Unity", {"theirs": 0})  # unrelated preset
        plan = self.artist.plan_import(self.bundle)
        [item] = [i for i in plan.items if i.name == "Unity"]
        self.assertEqual(item.status, "conflict")
        item.action = "keep_both"
        self.artist.apply(plan)
        store = self.exporter(self.other)
        self.assertEqual(store.load("Unity"), {"theirs": 0})
        self.assertEqual(store.load("Unity (Acme Standard)"), {"a": 1})

    def test_another_authors_same_named_collection_never_removes_mine(self):
        # The id is the lineage an update follows; the name is only a label.
        # Were ids derived from names, this install would read as an update of
        # mine and default to deleting my (published, so "unedited") members.
        self.publish()
        self.exporter(self.other).save("Theirs", {"t": 1})
        theirs = self.artist.create_collection("Acme Standard")
        self.artist.assign(self.artist.entries(builtin=False), theirs["id"])
        bundle = self.artist.export(
            os.path.join(self.other, "theirs.zip"), collection=theirs["id"]
        )
        self.assertNotEqual(theirs["id"], self.c["id"])
        plan = self.lib.plan_import(bundle)
        self.assertEqual(plan.counts(), {"new": 1})
        self.assertTrue(any("already have" in w for w in plan.warnings))

    def test_members_dropped_from_the_collection_are_offered_for_removal(self):
        self.publish()
        self.install()
        self.lib.assign([self.lib.entry("mayatk/scene_exporter", "WebXR")], None)
        self.publish()
        plan = self.artist.plan_import(self.bundle)
        [removed] = [i for i in plan.items if i.status == "removed"]
        self.assertEqual((removed.name, removed.action), ("WebXR", "remove"))
        self.artist.apply(plan)
        self.assertFalse(self.exporter(self.other).exists("WebXR"))

    def test_a_failed_export_bumps_no_version_and_stamps_no_baseline(self):
        from pythontk.file_utils._file_utils import FileUtils

        with mock.patch.object(FileUtils, "atomic_write", side_effect=OSError("full")):
            with self.assertRaises(OSError):
                self.publish()
        self.assertEqual(self.lib.collection(self.c["id"])["version"], 0)
        unity = self.lib.entry("mayatk/scene_exporter", "Unity")
        self.assertNotIn("origin_hash", unity.info)

    def test_an_invalid_action_fails_the_whole_plan_before_any_write(self):
        self.publish()
        plan = self.artist.plan_import(self.bundle)
        plan.items[0].action = "bogus"
        with self.assertRaises(ValueError):
            self.artist.apply(plan)
        self.assertEqual(self.artist.entries(), [])
        self.assertEqual(self.artist.backups(), [])

    def test_same_content_with_other_metadata_is_an_update_not_a_conflict(self):
        self.publish()
        # The artist already has an unstamped copy with identical bytes.
        other = self.exporter(self.other)
        other.path("Unity").parent.mkdir(parents=True, exist_ok=True)
        other.path("Unity").write_bytes(self.exporter().path("Unity").read_bytes())
        [item] = [
            i for i in self.artist.plan_import(self.bundle).items if i.name == "Unity"
        ]
        self.assertEqual((item.status, item.action), ("update", "replace"))

    def test_a_changed_description_is_an_update_not_identical(self):
        self.publish()
        self.install()
        [unity] = [e for e in self.lib.members(self.c["id"]) if e.name == "Unity"]
        self.lib.set_description([unity], "The house Unity export.")
        self.publish()
        [item] = [
            i for i in self.artist.plan_import(self.bundle).items if i.name == "Unity"
        ]
        self.assertEqual((item.status, item.action), ("update", "replace"))
        self.install()
        installed = self.artist.entry("mayatk/scene_exporter", "Unity")
        self.assertEqual(installed.description, "The house Unity export.")

    def test_hiding_is_the_artists_own_and_no_collection_carries_it(self):
        # The author hides a member in their own dropdown: it still installs
        # visible. The artist hides one: an update keeps it hidden.
        [unity] = [e for e in self.lib.members(self.c["id"]) if e.name == "Unity"]
        self.lib.set_hidden([unity])
        self.publish()
        with zipfile.ZipFile(self.bundle) as zf:
            sidecar = json.loads(zf.read("presets/mayatk/scene_exporter/.Unity.preset"))
        self.assertNotIn("hidden", sidecar)
        self.install()
        mine = self.artist.entry("mayatk/scene_exporter", "WebXR")
        self.assertFalse(self.artist.entry("mayatk/scene_exporter", "Unity").hidden)
        self.artist.set_hidden([mine])
        self.exporter().save("WebXR", {"a": 22}, force=True)  # a v2 of the member
        self.publish()
        plan = self.artist.plan_import(self.bundle)
        self.assertEqual(
            {i.name: i.status for i in plan.items},
            {"Unity": "identical", "WebXR": "update"},
        )
        self.artist.apply(plan)
        webxr = self.artist.entry("mayatk/scene_exporter", "WebXR")
        self.assertEqual(json.loads(webxr.path.read_text()), {"a": 22})
        self.assertTrue(webxr.hidden)

    def test_apply_backs_up_first(self):
        self.publish()
        self.exporter(self.other).save("Mine", {"m": 1})
        result = self.artist.apply(self.artist.plan_import(self.bundle))
        self.assertTrue(result.backup and result.backup.is_file())
        self.assertTrue(result.backup.name.startswith("auto-"))
        self.assertEqual(result.domains, ["mayatk/scene_exporter"])

    def test_delete_collection_deletes_unedited_members_and_untags_edited(self):
        self.publish()
        self.install()
        self.exporter(self.other).save("WebXR", {"edited": 1}, force=True)
        counts = self.artist.delete_collection(self.c["id"], delete_members=True)
        self.assertEqual(counts, {"deleted": 1, "untagged": 1})
        store = self.exporter(self.other)
        self.assertFalse(store.exists("Unity"))
        self.assertIsNone(store.info("WebXR").get("collection"))
        self.assertIsNone(self.artist.collection(self.c["id"]))


class BackupTest(_LibraryCase):
    def test_backup_round_trips_ids_tags_and_locks(self):
        self.exporter().save("Unity", {"a": 1})
        self.fbx().save("game", {"axis": "Y"})
        self.lib.set_read_only([self.lib.entry("mayatk/scene_exporter", "Unity")])
        path = self.lib.backup()
        self.assertEqual(path.parent.name, BACKUPS_DIR)
        fresh = PresetLibrary(self.other)
        plan = fresh.plan_import(path)
        self.assertEqual(plan.kind, "backup")
        fresh.apply(plan)
        restored = fresh.entry("mayatk/scene_exporter", "Unity")
        self.assertTrue(restored.read_only)
        self.assertEqual(
            restored.id, self.lib.entry("mayatk/scene_exporter", "Unity").id
        )
        self.assertEqual(json.loads(restored.path.read_text()), {"a": 1})
        self.assertIsNone(fresh.collection("backup"))  # a backup installs no header

    def test_a_restore_into_a_fresh_root_keeps_a_non_json_payload_format(self):
        # The destination has no .domain marker yet (its tool never ran there),
        # so only the bundle member's name says what the payload is: landed as
        # *.json, a YAML preset was invisible to its own store.
        codec = Codec(".yaml", json.loads, json.dumps)  # stands in for a YAML codec
        PresetStore("manifest_behaviors", "shots", codec=codec).save("b", {"x": 1})
        self.lib.export(self.bundle)
        fresh = PresetLibrary(self.other)
        fresh.apply(fresh.plan_import(self.bundle), backup=False)
        landed = Path(self.other, "shots", "manifest_behaviors")
        self.assertEqual(
            sorted(p.name for p in landed.iterdir() if not p.name.startswith(".")),
            ["b.yaml"],
        )
        tool = PresetStore("manifest_behaviors", codec=codec, user_dir=landed)
        self.assertEqual(tool.load("b"), {"x": 1})

    def test_two_backups_in_one_second_do_not_overwrite_each_other(self):
        from pythontk.core_utils.presets.store import _PresetStoreInternal

        store = self.exporter()
        store.save("A", {"a": 1})
        store.save("B", {"b": 1})
        with mock.patch.object(
            _PresetStoreInternal, "_now", return_value="2026-01-01T00:00:00Z"
        ):
            first = self.lib.backup(reason="delete")
            store.delete("A")
            second = self.lib.backup(reason="delete")
        self.assertNotEqual(first, second)
        with zipfile.ZipFile(first) as zf:  # still holds the deleted preset
            self.assertIn("presets/mayatk/scene_exporter/A.json", zf.namelist())

    def test_nothing_to_back_up_returns_none(self):
        self.assertIsNone(self.lib.backup())

    def test_a_backup_can_cover_some_stores_and_keeps_what_you_hid(self):
        self.exporter().save("Unity", {"a": 1})
        self.fbx().save("game", {"axis": "Y"})
        self.lib.set_hidden([self.lib.entry("blendertk/fbx_presets", "game")])
        path = self.lib.backup(keys=["blendertk/fbx_presets"], name="Blender › FBX")
        self.assertIsNone(self.lib.backup(keys=["nowhere/tool"]))
        self.assertFalse(path.name.startswith("auto-"), "a manual backup is kept")
        self.assertIn("blender-fbx", path.name)
        with zipfile.ZipFile(path) as zf:
            header = json.loads(zf.read("collection.json"))
            payloads = [n for n in zf.namelist() if n.endswith(".json")]
        self.assertEqual(header["name"], "Blender › FBX")
        self.assertEqual(
            payloads, ["collection.json", "presets/blendertk/fbx_presets/game.json"]
        )
        fresh = PresetLibrary(self.other)
        fresh.apply(fresh.plan_import(path), backup=False)
        self.assertTrue(fresh.entry("blendertk/fbx_presets", "game").hidden)

    def test_restoring_a_backup_in_place_restores_a_hide(self):
        # The payload is unchanged, so only the hide differs: that must still
        # read as an update, or the restore skips it as identical.
        self.fbx().save("game", {"axis": "Y"})
        game = self.lib.entry("blendertk/fbx_presets", "game")
        self.lib.set_hidden([game])
        path = self.lib.backup()
        self.lib.set_hidden([game], False)
        plan = self.lib.plan_import(path)
        self.assertEqual([(i.name, i.status) for i in plan.items], [("game", "update")])
        self.lib.apply(plan, backup=False)
        self.assertTrue(self.lib.entry("blendertk/fbx_presets", "game").hidden)

    def test_only_the_newest_automatic_backups_are_kept(self):
        d = Path(self.root, BACKUPS_DIR)
        d.mkdir()
        now = time.time()
        for i in range(AUTO_BACKUP_KEEP + 3):
            p = d / f"auto-{i:02d}-import.zip"
            p.write_bytes(b"")
            os.utime(p, (now - 1000 + i, now - 1000 + i))
        (d / "manual.zip").write_bytes(b"")
        os.utime(d / "manual.zip", (now - 5000, now - 5000))
        self.lib._prune_backups()
        autos = sorted(p.name for p in d.glob("auto-*.zip"))
        self.assertEqual(len(autos), AUTO_BACKUP_KEEP)
        self.assertEqual(autos[0], "auto-03-import.zip")
        self.assertTrue((d / "manual.zip").exists())


class ImportGuardTest(_LibraryCase):
    def test_member_paths_cannot_escape_the_root(self):
        # A bundle comes from another machine: its member names are untrusted.
        header = {"format": BUNDLE_FORMAT, "kind": "backup", "name": "x"}
        evil = [
            "presets/../../escaped/evil.json",
            "presets/mayatk/..\\..\\escaped/evil.json",
            "presets/C:/escaped/evil.json",
            "presets/.backups/evil.json",
        ]
        with zipfile.ZipFile(self.bundle, "w") as zf:
            zf.writestr("collection.json", json.dumps(header))
            for name in evil:
                zf.writestr(name, "{}")
            zf.writestr("presets/mayatk/scene_exporter/ok.json", "{}")
        lib = PresetLibrary(self.other)
        plan = lib.plan_import(self.bundle)
        self.assertEqual(
            [(i.domain, i.name) for i in plan.items], [("mayatk/scene_exporter", "ok")]
        )
        lib.apply(plan)
        escaped = Path(self.other).parent / "escaped"
        self.assertFalse(escaped.exists())
        self.assertTrue(
            Path(self.other, "mayatk", "scene_exporter", "ok.json").is_file()
        )

    def test_a_header_id_that_could_name_a_path_is_refused(self):
        # The collection id names .collections/<id>.json -- untrusted input.
        for cid in ("x/../../escaped", "..", "Acme Standard", "a\\b"):
            with self.subTest(cid=cid):
                with zipfile.ZipFile(self.bundle, "w") as zf:
                    zf.writestr(
                        "collection.json",
                        json.dumps(
                            {"format": BUNDLE_FORMAT, "kind": "collection", "id": cid}
                        ),
                    )
                with self.assertRaises(ValueError):
                    self.lib.plan_import(self.bundle)

    def test_not_a_bundle_and_a_newer_format_are_refused(self):
        Path(self.bundle).write_bytes(b"not a zip")
        with self.assertRaises(ValueError):
            self.lib.plan_import(self.bundle)
        with zipfile.ZipFile(self.bundle, "w") as zf:
            zf.writestr("collection.json", json.dumps({"format": BUNDLE_FORMAT + 1}))
        with self.assertRaises(ValueError):
            self.lib.plan_import(self.bundle)

    def test_a_bundle_from_newer_packages_warns(self):
        self.exporter().save("Unity", {"a": 1})
        self.lib.export(self.bundle)
        with zipfile.ZipFile(self.bundle) as zf:
            members = {n: zf.read(n) for n in zf.namelist()}
        header = json.loads(members["collection.json"])
        header["packages"] = {"pythontk": "999.0.0"}
        members["collection.json"] = json.dumps(header).encode()
        with zipfile.ZipFile(self.bundle, "w") as zf:
            for n, data in members.items():
                zf.writestr(n, data)
        warnings = PresetLibrary(self.other).plan_import(self.bundle).warnings
        self.assertEqual(len(warnings), 1)
        self.assertIn("pythontk 999.0.0", warnings[0])


class CleanUpTest(_LibraryCase):
    def test_removes_stray_sidecars_and_empty_folders_only(self):
        store = self.exporter()
        store.save("Unity", {"a": 1})
        store.save("Gone", {"a": 2})
        store.active = "Unity"
        # An older install deletes a preset without knowing about its sidecar.
        store.path("Gone").unlink()
        Path(self.root, "mayatk", "empty_tool").mkdir(parents=True)
        counts = self.lib.clean_up()
        self.assertEqual(counts, {"sidecars": 1, "folders": 1})
        self.assertTrue(store.info_path("Unity").exists())
        self.assertTrue((store.user_dir / ".active").exists())


if __name__ == "__main__":
    unittest.main()
