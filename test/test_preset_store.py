# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.PresetStore — Qt-free built-in + user named-preset store."""

import json
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

from pythontk.core_utils.presets.store import PresetStore
from pythontk.core_utils.user_config import UserConfig, CONFIG_ROOT_ENV_VAR


class PresetStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.builtin = os.path.join(self.tmp, "builtin")
        self.user = os.path.join(self.tmp, "user")
        os.makedirs(self.builtin)
        self._write(
            self.builtin,
            "specular_metal",
            {"depth_filter": "moderate", "align_downscale": 2},
        )
        self._write(self.builtin, "studio", {"depth_filter": "mild"})
        self.store = PresetStore(
            "photog", "extapps", builtin_dir=self.builtin, user_dir=self.user
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, d, name, data):
        with open(os.path.join(d, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    # --- discovery -----------------------------------------------------------
    def test_list_union_with_user_shadowing(self):
        self.store.save(
            "specular_metal", {"depth_filter": "aggressive"}
        )  # shadows builtin
        self.store.save("custom", {"x": 1})
        # Each name once; builtin + user unioned.
        self.assertEqual(self.store.list(), ["custom", "specular_metal", "studio"])
        self.assertEqual(self.store.list("builtin"), ["specular_metal", "studio"])
        self.assertEqual(self.store.list("user"), ["custom", "specular_metal"])

    def test_source_and_exists(self):
        self.assertEqual(self.store.source("studio"), "builtin")
        self.assertIsNone(self.store.source("nope"))
        self.assertFalse(self.store.exists("nope"))
        self.store.save("studio", {"depth_filter": "x"})
        self.assertEqual(self.store.source("studio"), "user")  # user now shadows

    # --- io ------------------------------------------------------------------
    def test_load_prefers_user_over_builtin(self):
        self.assertEqual(
            self.store.load("specular_metal")["align_downscale"], 2
        )  # builtin
        self.store.save("specular_metal", {"align_downscale": 99})
        self.assertEqual(
            self.store.load("specular_metal")["align_downscale"], 99
        )  # user wins

    def test_load_missing_raises_keyerror_with_available(self):
        with self.assertRaises(KeyError) as ctx:
            self.store.load("ghost")
        self.assertIn("specular_metal", str(ctx.exception))

    def test_save_writes_user_tier_only(self):
        self.store.save("studio", {"depth_filter": "x"})
        # built-in file untouched on disk.
        with open(os.path.join(self.builtin, "studio.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["depth_filter"], "mild")

    def test_delete_user_only_never_builtin(self):
        # Deleting a builtin-only name is a no-op (built-ins are read-only).
        self.assertFalse(self.store.delete("studio"))
        self.assertTrue(self.store.exists("studio"))  # builtin survives
        self.store.save("studio", {"x": 1})
        self.assertTrue(self.store.delete("studio"))  # removes the user shadow
        self.assertEqual(
            self.store.source("studio"), "builtin"
        )  # falls back to builtin

    def test_rename_user_preset(self):
        self.store.save("draft", {"x": 1})
        self.assertTrue(self.store.rename("draft", "final"))
        self.assertEqual(self.store.load("final")["x"], 1)
        self.assertFalse(self.store.exists("draft"))

    def test_rename_refuses_to_shadow_builtin(self):
        self.store.save("draft", {"x": 1})
        self.assertFalse(self.store.rename("draft", "studio"))  # 'studio' is a builtin
        self.assertTrue(self.store.exists("draft"))

    def test_name_sanitized_consistently(self):
        self.store.save("a/b:c", {"x": 1})
        # The same sanitized stem is used for save + load + path.
        self.assertEqual(self.store.load("a/b:c")["x"], 1)
        self.assertEqual(
            self.store.path("a/b:c", "user").name,
            PresetStore.sanitize_preset_name("a/b:c") + ".json",
        )

    # --- active pointer ------------------------------------------------------
    def test_active_round_trips_and_clears(self):
        self.assertIsNone(self.store.active)  # unset by default
        self.store.active = "studio"
        self.assertEqual(self.store.active, "studio")
        # A fresh store over the same dirs reads the same pointer (cross-session).
        other = PresetStore(
            "photog", "extapps", builtin_dir=self.builtin, user_dir=self.user
        )
        self.assertEqual(other.active, "studio")
        self.store.active = None
        self.assertIsNone(self.store.active)

    def test_active_sidecar_excluded_from_listing(self):
        self.store.save("custom", {"x": 1})
        self.store.active = "custom"
        # The .active dotfile must not surface as a preset.
        self.assertEqual(self.store.list("user"), ["custom"])
        self.assertFalse(self.store.exists(".active"))

    def test_delete_clears_dangling_active(self):
        self.store.save("draft", {"x": 1})
        self.store.active = "draft"
        self.store.delete("draft")
        self.assertIsNone(self.store.active)  # pointed-at preset is gone

    def test_delete_keeps_active_when_builtin_remains(self):
        # A user shadow deleted but the built-in of the same name survives.
        self.store.save("studio", {"x": 1})
        self.store.active = "studio"
        self.store.delete("studio")  # removes user shadow, builtin remains
        self.assertEqual(self.store.active, "studio")

    def test_rename_follows_active(self):
        self.store.save("draft", {"x": 1})
        self.store.active = "draft"
        self.store.rename("draft", "final")
        self.assertEqual(self.store.active, "final")

    # A name with punctuation is stored under a stem without it ("a (b)" ->
    # "a _b_"): the stem is what list() shows, what a combo holds and what the
    # Preset Editor renames and deletes by, so it is what the pointer holds.
    def test_active_holds_the_file_stem_of_a_name_with_punctuation(self):
        self.store.save("m3trik (laptop)", {"x": 1})
        self.store.active = "m3trik (laptop)"
        self.assertEqual(self.store.active, "m3trik _laptop_")
        self.assertIn(self.store.active, self.store.list("user"))
        self.assertEqual(self.store.load(self.store.active)["x"], 1)

    def test_rename_by_stem_follows_an_active_name_with_punctuation(self):
        self.store.save("m3trik (laptop)", {"x": 1})
        self.store.active = "m3trik (laptop)"
        self.assertTrue(self.store.rename("m3trik _laptop_", "m3trik laptop"))
        self.assertEqual(self.store.active, "m3trik laptop")
        self.assertEqual(self.store.load(self.store.active)["x"], 1)

    def test_delete_by_stem_clears_an_active_name_with_punctuation(self):
        self.store.save("draft (old)", {"x": 1})
        self.store.active = "draft (old)"
        self.assertTrue(self.store.delete("draft _old_"))
        self.assertIsNone(self.store.active)

    def test_a_pointer_written_as_typed_by_an_older_release_still_resolves(self):
        # The pointer used to hold the name as typed; that file outlives the
        # upgrade, so it must still name its preset and still follow a rename.
        self.store.save("m3trik (laptop)", {"x": 1})
        (self.store.user_dir / ".active").write_text(
            json.dumps({"name": "m3trik (laptop)"}), encoding="utf-8"
        )
        self.assertEqual(self.store.active, "m3trik _laptop_")
        self.assertTrue(self.store.rename("m3trik _laptop_", "m3trik laptop"))
        self.assertEqual(self.store.active, "m3trik laptop")

    def test_no_builtin_dir_is_user_only(self):
        store = PresetStore("p", "extapps", user_dir=self.user)
        self.assertIsNone(store.builtin_dir)
        self.assertEqual(store.list("builtin"), [])
        store.save("only", {"x": 1})
        self.assertEqual(store.list(), ["only"])


class PresetStoreAtomicWriteTest(unittest.TestCase):
    """Writes are atomic: a failed save can never corrupt an existing preset.

    Both tests force the atomic swap (``os.replace``) to fail — under a
    regression to a plain truncate-and-write (``write_text`` / ``json.dump``),
    the patched ``os.replace`` never fires, the write *succeeds*, and the
    asserts below fail. So these pin the routing through the atomic path, not
    just the primitive (which ``test_atomic_write.py`` covers on its own).
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = PresetStore("p", "extapps", user_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_failed_save_propagates_and_leaves_existing_preset_intact(self):
        from unittest import mock

        self.store.save("cfg", {"version": 1})
        with mock.patch("os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.save("cfg", {"version": 2})
        # Old content survives readable (a truncate-then-write would have
        # destroyed it even though the save failed), and no temp litter.
        self.assertEqual(self.store.load("cfg"), {"version": 1})
        self.assertEqual(self.store.list("user"), ["cfg"])
        self.assertFalse(
            [n for n in os.listdir(self.tmp) if n.endswith(".tmp")],
            "failed save left a temp file behind",
        )

    def test_failed_active_write_keeps_prior_pointer(self):
        from unittest import mock

        self.store.save("cfg", {"x": 1})
        self.store.active = "cfg"
        with mock.patch("os.replace", side_effect=OSError("disk full")):
            self.store.active = "other"  # setter swallows OSError by contract
        self.assertEqual(self.store.active, "cfg")  # prior pointer intact


class PresetStoreDefaultLocationTest(unittest.TestCase):
    """With no explicit user_dir, the user tier lands under user_config_root."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._prev = os.environ.get(CONFIG_ROOT_ENV_VAR)
        os.environ[CONFIG_ROOT_ENV_VAR] = self.tmp  # never touch the real user dir

    def tearDown(self):
        if self._prev is None:
            os.environ.pop(CONFIG_ROOT_ENV_VAR, None)
        else:
            os.environ[CONFIG_ROOT_ENV_VAR] = self._prev
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_user_dir_under_package_and_name(self):
        store = PresetStore("photog_presets", "extapps")
        self.assertEqual(
            store.user_dir, UserConfig.user_config_root() / "extapps" / "photog_presets"
        )
        # Lazily created on save, not on read.
        self.assertFalse(store.user_dir.exists())
        store.save("p", {"x": 1})
        self.assertTrue(store.user_dir.is_dir())


class PresetStoreCodecTest(unittest.TestCase):
    """A pluggable codec changes the on-disk format and extension."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_codec_is_json(self):
        store = PresetStore("p", user_dir=self.tmp)
        self.assertEqual(store.ext, ".json")

    def test_custom_codec_writes_its_extension_and_round_trips(self):
        from pythontk.core_utils.presets.store import Codec

        # A trivial non-JSON codec (here still JSON-encoded text, but a distinct
        # extension) proves discovery + IO route through the codec, not hardcoded.
        codec = Codec(".yaml", json.loads, lambda d: json.dumps(d))
        store = PresetStore("p", user_dir=self.tmp, codec=codec)
        path = store.save("cfg", {"a": 1})
        self.assertEqual(path.suffix, ".yaml")
        self.assertEqual(store.list(), ["cfg"])  # glob uses the codec ext
        self.assertEqual(store.load("cfg"), {"a": 1})

    def test_json_files_are_invisible_to_a_yaml_store(self):
        from pythontk.core_utils.presets.store import Codec

        json_store = PresetStore("p", user_dir=self.tmp)
        json_store.save("only_json", {"a": 1})
        yaml_store = PresetStore(
            "p", user_dir=self.tmp, codec=Codec(".yaml", json.loads, json.dumps)
        )
        self.assertEqual(yaml_store.list(), [])  # different extension, not found

    def test_codec_ext_is_normalized_with_leading_dot(self):
        from pythontk.core_utils.presets.store import Codec

        # A dotless ext is a natural public-API mistake; it must not produce
        # 'cfgyaml' filenames / '*yaml' globs. __post_init__ prepends the dot.
        store = PresetStore(
            "p", user_dir=self.tmp, codec=Codec("yaml", json.loads, json.dumps)
        )
        self.assertEqual(store.ext, ".yaml")
        self.assertEqual(store.save("cfg", {"a": 1}).suffix, ".yaml")
        self.assertEqual(store.list(), ["cfg"])


class _TempRootCase(unittest.TestCase):
    """Per-test scratch dir under ``test/temp_tests/``, set as the presets root."""

    def setUp(self):
        self.root = os.path.join(
            os.path.dirname(__file__),
            "temp_tests",
            f"preset_store_{self._testMethodName}_{uuid.uuid4().hex[:6]}",
        )
        # Unique per run: a folder a sync client kept from being deleted, or a
        # concurrent run of the same test (routine with several sessions), must
        # never be this run's root -- both surfaced as FileExistsError here.
        os.makedirs(self.root)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        prev = os.environ.get(CONFIG_ROOT_ENV_VAR)
        os.environ[CONFIG_ROOT_ENV_VAR] = self.root
        self.addCleanup(
            lambda: (
                os.environ.pop(CONFIG_ROOT_ENV_VAR, None)
                if prev is None
                else os.environ.__setitem__(CONFIG_ROOT_ENV_VAR, prev)
            )
        )


class PresetStoreInfoTest(_TempRootCase):
    """Management metadata lives in a sidecar; the payload is never touched."""

    def setUp(self):
        super().setUp()
        self.builtin = os.path.join(self.root, "_shipped")
        os.makedirs(self.builtin)
        with open(os.path.join(self.builtin, "stock.json"), "w") as fh:
            json.dump({"x": 0}, fh)
        self.store = PresetStore("tool", "pkg", builtin_dir=self.builtin)

    def test_payload_round_trips_byte_for_byte_with_no_metadata_key(self):
        # The reason metadata is a sidecar: raw stores splat the payload into
        # kwargs, and older installs read the same file.
        self.store.save("raw", {"axis": "Y"})
        self.assertEqual(self.store.load("raw"), {"axis": "Y"})
        with open(self.store.path("raw"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh), {"axis": "Y"})

    def test_first_save_stamps_identity_and_resave_keeps_it(self):
        self.store.save("a", {"v": 1})
        info = self.store.info("a")
        self.assertTrue(info["id"])
        self.assertEqual(info["label"], "a")
        self.assertIn("created", info)
        self.store.save("a", {"v": 2})
        self.assertEqual(self.store.info("a")["id"], info["id"])

    def test_label_keeps_the_punctuation_the_file_name_loses(self):
        name = "MRAO (Metallic, Roughness, AO)"
        path = self.store.save(name, {"v": 1})
        self.assertEqual(path.stem, "MRAO _Metallic_ Roughness_ AO_")
        self.assertEqual(self.store.info(name)["label"], name)

    def test_sidecar_and_marker_never_list_as_presets(self):
        self.store.save("a", {"v": 1})
        self.assertEqual(self.store.list("user"), ["a"])

    def test_set_info_merges_and_none_removes(self):
        self.store.save("a", {"v": 1})
        self.store.set_info("a", tags=["x"], collection="c")
        self.store.set_info("a", collection=None)
        info = self.store.info("a")
        self.assertEqual(info["tags"], ["x"])
        self.assertNotIn("collection", info)

    def test_set_info_on_a_builtin_or_missing_name_raises(self):
        with self.assertRaises(KeyError):
            self.store.set_info("stock", read_only=True)
        with self.assertRaises(KeyError):
            self.store.set_info("ghost", read_only=True)

    def test_locked_preset_refuses_save_delete_and_rename(self):
        from pythontk.core_utils.presets.store import PresetReadOnlyError

        self.store.save("a", {"v": 1})
        self.store.set_info("a", read_only=True)
        self.assertTrue(self.store.is_read_only("a"))
        with self.assertRaises(PresetReadOnlyError):
            self.store.save("a", {"v": 2})
        self.assertIsInstance(PresetReadOnlyError("x"), OSError)
        self.assertFalse(self.store.delete("a"))
        self.assertFalse(self.store.rename("a", "b"))
        self.assertEqual(self.store.load("a"), {"v": 1})
        # force: the collection-update path.
        self.store.save("a", {"v": 3}, force=True)
        self.assertEqual(self.store.load("a"), {"v": 3})
        self.assertTrue(self.store.delete("a", force=True))
        self.assertFalse(self.store.info_path("a").exists())

    def test_builtins_are_read_only_but_a_shadow_can_still_be_saved(self):
        self.assertTrue(self.store.is_read_only("stock"))
        self.store.save("stock", {"x": 9})  # duplicate-to-edit, as before
        self.assertFalse(self.store.is_read_only("stock"))

    def test_rename_moves_the_sidecar_keeps_the_id_and_relabels(self):
        self.store.save("a", {"v": 1})
        pid = self.store.info("a")["id"]
        self.assertTrue(self.store.rename("a", "b"))
        self.assertFalse(self.store.info_path("a").exists())
        self.assertEqual(self.store.info("b")["id"], pid)
        self.assertEqual(self.store.info("b")["label"], "b")

    def test_delete_removes_the_sidecar(self):
        self.store.save("a", {"v": 1})
        self.assertTrue(self.store.delete("a"))
        self.assertFalse(self.store.info_path("a").exists())

    def test_a_new_preset_never_inherits_a_stray_sidecar(self):
        # An older install deletes the payload but not the sidecar it doesn't
        # know about; a later, unrelated preset of that name must start clean.
        self.store.save("a", {"v": 1})
        self.store.set_info("a", read_only=True, collection="studio")
        dead_id = self.store.info("a")["id"]
        self.store.path("a").unlink()
        self.store.save("a", {"v": 2})  # must not raise: the lock was the dead one's
        info = self.store.info("a")
        self.assertNotEqual(info["id"], dead_id)
        self.assertNotIn("read_only", info)
        self.assertNotIn("collection", info)

    def test_unique_name_numbers_past_both_tiers(self):
        self.store.save("stock 2", {"v": 1})  # "stock" is a built-in
        self.assertEqual(self.store.unique_name("stock"), "stock 3")
        self.assertEqual(self.store.unique_name("fresh"), "fresh")

    def test_key_is_the_folder_under_the_root(self):
        self.assertEqual(self.store.key, "pkg/tool")
        outside = PresetStore("t", user_dir=os.path.join(self.root + "_elsewhere", "x"))
        self.assertIsNone(outside.key)


class PresetStoreHiddenTest(_TempRootCase):
    """Hiding a preset from its tool's dropdown: built-ins included, files kept."""

    def setUp(self):
        super().setUp()
        self.builtin = os.path.join(self.root, "_shipped")
        os.makedirs(self.builtin)
        for name in ("stock", "spare"):
            with open(os.path.join(self.builtin, f"{name}.json"), "w") as fh:
                json.dump({"x": 0}, fh)
        self.store = PresetStore("tool", "pkg", builtin_dir=self.builtin)
        self.store.save("mine", {"v": 1})

    def test_a_builtin_and_a_user_preset_can_be_hidden_and_shown(self):
        self.assertTrue(self.store.set_hidden("stock"))
        self.assertTrue(self.store.set_hidden("mine"))
        self.assertTrue(self.store.is_hidden("stock"))
        self.assertTrue(self.store.is_hidden("mine"))
        self.assertFalse(self.store.is_hidden("spare"))
        self.assertFalse(self.store.set_hidden("stock"), "already hidden: no change")
        self.assertTrue(self.store.set_hidden("stock", False))
        self.assertFalse(self.store.is_hidden("stock"))
        # Hidden from the dropdown, not from the store: every file is kept.
        self.assertEqual(self.store.list(), ["mine", "spare", "stock"])
        self.assertEqual(self.store.load("mine"), {"v": 1})

    def test_hiding_a_missing_preset_raises(self):
        with self.assertRaises(KeyError):
            self.store.set_hidden("ghost")

    def test_the_builtin_list_is_never_a_preset_and_is_written_atomically(self):
        from unittest import mock

        from pythontk.core_utils.presets.store import _PresetStoreInternal

        real = _PresetStoreInternal._atomic_write_text
        with mock.patch.object(
            _PresetStoreInternal, "_atomic_write_text", side_effect=real
        ) as write:
            self.store.set_hidden("stock")
        written = [Path(call.args[0]).name for call in write.call_args_list]
        self.assertEqual(written, [".hidden"])
        self.assertEqual(self.store.list("user"), ["mine"])
        self.assertEqual(self.store.list(), ["mine", "spare", "stock"])

    def test_a_user_presets_flag_follows_a_rename(self):
        self.store.save("a (b)", {"v": 1})
        self.store.set_hidden("a (b)")
        self.assertTrue(self.store.rename("a (b)", "c (d)"))
        self.assertTrue(self.store.is_hidden("c (d)"))
        self.assertTrue(self.store.is_hidden("c _d_"))  # the stem names it too

    def test_delete_drops_the_flag_and_a_new_namesake_starts_visible(self):
        self.store.set_hidden("mine")
        self.assertTrue(self.store.delete("mine"))
        self.store.save("mine", {"v": 2})
        self.assertFalse(self.store.is_hidden("mine"))

    def test_the_flag_belongs_to_the_preset_not_the_name(self):
        # A user copy shadowing a hidden built-in is the user's own, and shows;
        # deleting it brings the (still hidden) built-in back.
        self.store.set_hidden("stock")
        self.store.save("stock", {"x": 9})
        self.assertFalse(self.store.is_hidden("stock"))
        self.assertTrue(self.store.delete("stock"))
        self.assertTrue(self.store.is_hidden("stock"))

    def test_hiding_a_builtin_announces_a_store_that_had_no_folder(self):
        # A tool with only shipped presets has no user folder yet: hiding one
        # makes it, and the library must still find the store there.
        store = PresetStore("fresh", "pkg", builtin_dir=self.builtin)
        self.assertFalse(store.user_dir.exists())
        store.set_hidden("stock")
        self.assertTrue((store.user_dir / ".domain").is_file())


class PresetStoreDescriptionTest(_TempRootCase):
    """What a preset is for: the user's in its sidecar, a built-in's shipped."""

    def setUp(self):
        super().setUp()
        self.builtin = os.path.join(self.root, "_shipped")
        os.makedirs(self.builtin)
        shipped = {
            "documented": {"_meta": {"version": 1, "description": "For X."}, "x": 0},
            "bare": {"x": 0},
        }
        for name, data in shipped.items():
            with open(os.path.join(self.builtin, f"{name}.json"), "w") as fh:
                json.dump(data, fh)
        self.store = PresetStore("tool", "pkg", builtin_dir=self.builtin)

    def test_a_user_description_lives_in_the_sidecar_and_survives_a_rename(self):
        self.store.save("mine", {"v": 1})
        self.store.set_info("mine", description="For hero shots.")
        self.assertEqual(self.store.description("mine"), "For hero shots.")
        self.assertEqual(self.store.load("mine"), {"v": 1})  # payload untouched
        self.assertTrue(self.store.rename("mine", "hero"))
        self.assertEqual(self.store.description("hero"), "For hero shots.")

    def test_a_builtin_description_is_read_from_its_shipped_meta(self):
        self.assertEqual(self.store.description("documented"), "For X.")
        self.assertEqual(self.store.description("bare"), "")
        self.assertEqual(self.store.description("ghost"), "")
        with self.assertRaises(KeyError):  # read-only: built-ins have no sidecar
            self.store.set_info("documented", description="Mine now.")


class PresetStoreMarkerTest(_TempRootCase):
    """A store announces itself with a ``.domain`` marker, without creating dirs."""

    def _marker(self, store):
        return json.loads((store.user_dir / ".domain").read_text(encoding="utf-8"))

    def test_list_marks_an_existing_folder_with_ext_and_builtin_spec(self):
        builtin = os.path.join(self.root, "_shipped")
        os.makedirs(builtin)
        store = PresetStore("tool", "pkg", builtin_dir=builtin)
        store.user_dir.mkdir(parents=True)
        store.list()
        marker = self._marker(store)
        self.assertEqual(marker["key"], "pkg/tool")
        self.assertEqual(marker["ext"], ".json")
        # Whatever form was recorded, it must find this very dir again.
        self.assertEqual(
            PresetStore.resolve_builtin_spec(marker["builtin"]).resolve(),
            Path(builtin).resolve(),
        )

    def test_package_form_that_does_not_round_trip_falls_back_to_a_path(self):
        from unittest import mock

        # E.g. a repo's ``test`` package shadowed by the stdlib's in another host.
        builtin = os.path.join(self.root, "_shipped")
        os.makedirs(builtin)
        store = PresetStore("tool", "pkg", builtin_dir=builtin)
        store.user_dir.mkdir(parents=True)
        with mock.patch.object(PresetStore, "resolve_builtin_spec", return_value=None):
            store.list()
        self.assertEqual(
            self._marker(store)["builtin"], {"dir": str(Path(builtin).resolve())}
        )

    def test_a_read_never_creates_the_folder(self):
        store = PresetStore("never", "pkg")
        store.list()
        self.assertFalse(store.user_dir.exists())

    def test_mark_false_writes_nothing(self):
        store = PresetStore("quiet", "pkg", mark=False)
        store.save("a", {"v": 1})
        self.assertFalse((store.user_dir / ".domain").exists())

    def test_builtin_inside_a_package_is_recorded_package_relative(self):
        import pythontk

        pkg_dir = Path(pythontk.__file__).parent
        builtin = pkg_dir / "core_utils"  # any dir inside the package
        spec = PresetStore._builtin_spec(builtin)
        self.assertEqual(spec, {"package": "pythontk", "path": "core_utils"})
        self.assertEqual(
            PresetStore.resolve_builtin_spec(spec).resolve(), builtin.resolve()
        )

    def test_unknown_package_resolves_to_none(self):
        self.assertIsNone(
            PresetStore.resolve_builtin_spec({"package": "no_such_pkg_x", "path": "p"})
        )
        self.assertIsNone(PresetStore.resolve_builtin_spec(None))


if __name__ == "__main__":
    unittest.main()
