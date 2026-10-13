#!/usr/bin/python
# coding=utf-8
"""Tests for ``pythontk.BakeCache`` (``file_utils/bake_cache.py``).

A project's files baked from other files: one slot per source in the
project's cache tree, the source and its content key recorded INSIDE each
bake (read through a host-injected reader), and a purge by what the project's
files still reference. Hoisted from the twin bake stores of mayatk's
``BlenderSceneImport`` and blendertk's ``MayaSceneImport``; each host keeps how
a record is read from and written into its own file format. Here the
"format" is a JSON file, so real files in a scratch folder are all a test
needs -- no host.
"""

import json
import os
import re
import unittest
import unittest.mock

import pythontk as ptk

from conftest import BaseTestCase

SOURCE_FIELD = "t_bake_source"
KEY_FIELD = "t_bake_key"


def _read_info(path):
    """The test format's record: the bake file IS its JSON record."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _produce(out, info):
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(info), fh)


class _ProjectCase(BaseTestCase):
    """A marked project with one source file."""

    def setUp(self):
        self.tmp_store = ptk.TempArtifacts("ptk_bake_cache")
        self.addCleanup(self.tmp_store.cleanup)
        self.proj = os.path.join(self.tmp_store.dir_path(), "proj")
        os.makedirs(os.path.join(self.proj, "scenes"))
        with open(os.path.join(self.proj, "workspace.mel"), "w") as fh:
            fh.write('workspace -fr "scene" "scenes";\n')
        self.src = self._file("scenes", "hero v01.fbx", data=b"fbx")
        self.bakes = ptk.BakeCache(
            "t_bake",
            ".bk",
            read_info=_read_info,
            source_field=SOURCE_FIELD,
            key_field=KEY_FIELD,
            legacy_suffix=".source.json",
        )

    def _file(self, *parts, data=b"x"):
        path = os.path.join(self.proj, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return path


class TestSlots(_ProjectCase):
    def test_the_cache_dir_is_the_projects_bridge_cache(self):
        self.assertEqual(
            os.path.normcase(self.bakes.cache_dir(self.proj)),
            os.path.normcase(os.path.join(self.proj, "cache", "bridge")),
        )
        self.assertFalse(os.path.isdir(self.bakes.cache_dir(self.proj)))
        self.assertTrue(os.path.isdir(self.bakes.cache_dir(self.proj, create=True)))

    def test_a_cache_rule_names_the_folder(self):
        with open(os.path.join(self.proj, "workspace.mel"), "a") as fh:
            fh.write('workspace -fr "bridgeCache" "bakes";\n')
        self.assertEqual(
            os.path.normcase(self.bakes.cache_dir(self.proj)),
            os.path.normcase(os.path.join(self.proj, "bakes")),
        )

    def test_the_store_is_built_once_per_cache_folder(self):
        """``current`` / ``slot`` / ``write`` run per reference a browser lists;
        each built a fresh ``CachedArtifact`` (and its ``TempArtifacts``)."""
        built = []
        real = ptk.CachedArtifact.__init__

        def init(store, *a, **kw):
            built.append(kw.get("dir"))
            real(store, *a, **kw)

        with unittest.mock.patch.object(
            ptk.CachedArtifact, "__init__", autospec=True, side_effect=init
        ):
            first = self.bakes.store(self.proj)
            self.bakes.slot(self.src, self.proj)
            self.bakes.current(self.src, self.proj, "k")
            self.bakes.write(self.src, self.proj, _produce, key="k")
            self.assertIs(self.bakes.store(self.proj, create=True), first)
            self.assertEqual(len(built), 1, built)
            # A rule that moves the folder is another store.
            with open(os.path.join(self.proj, "workspace.mel"), "a") as fh:
                fh.write('workspace -fr "bridgeCache" "bakes";\n')
            moved = self.bakes.store(self.proj)
        self.assertIsNot(moved, first)
        self.assertEqual(
            os.path.normcase(moved.dir), os.path.normcase(os.path.join(self.proj, "bakes"))
        )

    def test_the_slot_key_is_the_stem_and_a_hash_of_the_relative_path(self):
        """The spelling the DCC bake stores have written since 2026-10-13:
        changing it orphans every bake a saved scene references."""
        import hashlib

        digest = hashlib.sha1(b"scenes/hero v01.fbx").hexdigest()[:8]
        self.assertEqual(
            ptk.BakeCache.slot_key(self.src, self.proj), f"hero_v01_{digest}"
        )

    def test_the_same_name_elsewhere_gets_another_slot(self):
        other = self._file("scenes", "b", "hero v01.fbx")
        self.assertNotEqual(
            ptk.BakeCache.slot_key(self.src, self.proj),
            ptk.BakeCache.slot_key(other, self.proj),
        )

    def test_a_name_with_no_word_characters_is_still_a_slot(self):
        odd = self._file("scenes", "★.fbx")
        self.assertRegex(ptk.BakeCache.slot_key(odd, self.proj), r"^_+_[0-9a-f]{8}$")

    def test_the_slot_is_named_for_the_store_and_the_source(self):
        slot = self.bakes.slot(self.src, self.proj)
        self.assertEqual(
            os.path.basename(slot),
            "t_bake_cache_" + ptk.BakeCache.slot_key(self.src, self.proj) + ".bk",
        )
        self.assertTrue(self.bakes.is_bake(slot))
        self.assertFalse(self.bakes.is_bake(self.src))


class TestSourceKey(_ProjectCase):
    def test_the_key_is_the_cached_artifact_key_of_the_parts_and_stamps(self):
        """Byte-identical to the DCC stores' keys before the hoist, so no
        existing bake reads as stale."""
        stat = os.stat(self.src)
        stamps = [(os.path.basename(self.src), stat.st_size, stat.st_mtime_ns)]
        self.assertEqual(
            ptk.BakeCache.source_key(self.src, 1, ("fbx", [])),
            ptk.CachedArtifact.key(1, ("fbx", []), stamps),
        )

    def test_an_existing_companion_joins_the_key(self):
        manifest = self.src + ".manifest.json"
        alone = ptk.BakeCache.source_key(self.src, 1, companions=[manifest])
        with open(manifest, "w") as fh:
            fh.write("{}")
        self.assertNotEqual(
            alone, ptk.BakeCache.source_key(self.src, 1, companions=[manifest])
        )

    def test_an_edit_to_the_source_changes_the_key(self):
        before = ptk.BakeCache.source_key(self.src, 1)
        with open(self.src, "ab") as fh:
            fh.write(b"more")
        self.assertNotEqual(before, ptk.BakeCache.source_key(self.src, 1))

    def test_no_path_in_the_key(self):
        """A moved project keeps its bakes current."""
        moved = os.path.join(self.proj, "scenes", "moved")
        os.makedirs(moved)
        twin = os.path.join(moved, os.path.basename(self.src))
        with open(twin, "wb") as fh:
            fh.write(b"fbx")
        st = os.stat(self.src)
        os.utime(twin, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertEqual(
            ptk.BakeCache.source_key(self.src, 1), ptk.BakeCache.source_key(twin, 1)
        )


class TestWriteAndRead(_ProjectCase):
    def test_a_write_records_the_source_and_key_inside_the_bake(self):
        slot = self.bakes.write(self.src, self.proj, _produce, key="k1")
        self.assertEqual(slot, self.bakes.slot(self.src, self.proj))
        self.assertEqual(
            _read_info(slot),
            {SOURCE_FIELD: "../../scenes/hero v01.fbx", KEY_FIELD: "k1"},
        )

    def test_current_is_the_slot_only_while_it_holds_the_key(self):
        self.assertIsNone(self.bakes.current(self.src, self.proj, "k1"))
        slot = self.bakes.write(self.src, self.proj, _produce, key="k1")
        self.assertEqual(self.bakes.current(self.src, self.proj, "k1"), slot)
        self.assertIsNone(self.bakes.current(self.src, self.proj, "k2"))

    def test_a_rewrite_replaces_the_one_slot(self):
        self.bakes.write(self.src, self.proj, _produce, key="k1")
        self.bakes.write(self.src, self.proj, _produce, key="k2")
        self.assertEqual(
            [
                n
                for n in os.listdir(self.bakes.cache_dir(self.proj))
                if n.endswith(".bk")
            ],
            [os.path.basename(self.bakes.slot(self.src, self.proj))],
        )

    def test_a_failed_produce_leaves_the_slot_as_it_was(self):
        slot = self.bakes.write(self.src, self.proj, _produce, key="k1")

        def _boom(out, info):
            with open(out, "w") as fh:
                fh.write("partial")
            raise RuntimeError("killed")

        with self.assertRaises(RuntimeError):
            self.bakes.write(self.src, self.proj, _boom, key="k2")
        self.assertEqual(_read_info(slot)[KEY_FIELD], "k1")

    def test_a_source_outside_the_project_is_recorded_absolute(self):
        outside = ptk.TempArtifacts("ptk_bake_cache_out")
        self.addCleanup(outside.cleanup)
        src = os.path.join(outside.dir_path(), "far.fbx")
        open(src, "w").close()
        self.assertEqual(
            self.bakes.spell_source(src, self.proj), src.replace("\\", "/")
        )

    def test_the_source_of_a_bake_is_read_back_absolute(self):
        slot = self.bakes.write(self.src, self.proj, _produce, key="k1")
        self.assertTrue(
            ptk.FileUtils.is_same_file(self.bakes.source_of(slot), self.src)
        )

    def test_a_file_not_named_as_a_bake_is_never_read(self):
        reads = []
        bakes = ptk.BakeCache(
            "t_bake",
            ".bk",
            read_info=lambda p: reads.append(p) or {},
            source_field=SOURCE_FIELD,
            key_field=KEY_FIELD,
        )
        self.assertIsNone(bakes.source_of(self.src))
        self.assertEqual(reads, [])

    def test_a_legacy_sidecar_names_the_source(self):
        old = self._file("old", "t_bake_cache_x_0123abcd.bk", data=b"{}")
        with open(old + ".source.json", "w") as fh:
            json.dump({"source": "C:/somewhere/hero.fbx"}, fh)
        self.assertEqual(self.bakes.source_of(old), "C:/somewhere/hero.fbx")
        self.assertIsNone(self.bakes.source_of(""))


class TestRehome(_ProjectCase):
    def _temp_bake(self):
        stray = ptk.TempArtifacts("ptk_bake_cache_stray")
        self.addCleanup(stray.cleanup)
        old = os.path.join(stray.dir_path(), "t_bake_cache_old_0123abcd.bk")
        with open(old, "w") as fh:
            json.dump({SOURCE_FIELD: self.src}, fh)
        return old

    def test_a_temp_bake_is_copied_into_its_sources_slot_recording_only_the_source(
        self,
    ):
        old = self._temp_bake()
        seen = []

        def _copy(baked, out, info):
            seen.append(baked)
            _produce(out, info)

        home = self.bakes.rehome(old, self.proj, _copy)
        self.assertEqual(home, self.bakes.slot(self.src, self.proj))
        self.assertEqual(seen, [old])
        self.assertEqual(_read_info(home), {SOURCE_FIELD: "../../scenes/hero v01.fbx"})

    def test_an_existing_slot_is_the_answer(self):
        slot = self.bakes.write(self.src, self.proj, _produce, key="k1")
        home = self.bakes.rehome(
            self._temp_bake(), self.proj, lambda *a: self.fail("produced")
        )
        self.assertEqual(home, slot)

    def test_a_bake_with_no_source_is_not_rehomed(self):
        self.assertIsNone(
            self.bakes.rehome(self.src, self.proj, lambda *a: self.fail("produced"))
        )


class TestPurge(_ProjectCase):
    """A bake no project file (and no live session reference) names goes,
    with anything named for it; every other one stays."""

    def setUp(self):
        super().setUp()
        self.kept = self.bakes.write(self.src, self.proj, _produce, key="k")
        other = self._file("scenes", "villain.fbx")
        self.unused = self.bakes.write(other, self.proj, _produce, key="k")
        self.unused_sidecar = os.path.splitext(self.unused)[0] + ".usd"
        open(self.unused_sidecar, "w").close()
        self.stranger = self._file("cache", "bridge", "notes.txt")

    def _scene(self, *names):
        refs = "".join(
            f'file -r "cache/bridge/{os.path.basename(n)}";\n' for n in names
        )
        return self._file("scenes", "shot.ma", data=(refs + "createNode x;\n").encode())

    def test_a_referenced_bake_stays_and_an_unreferenced_one_goes(self):
        self._scene(self.kept)
        removed = self.bakes.purge(self.proj, (".ma",))
        self.assertEqual(
            sorted(os.path.normcase(p) for p in removed),
            sorted(os.path.normcase(p) for p in (self.unused, self.unused_sidecar)),
        )
        self.assertTrue(os.path.isfile(self.kept))
        self.assertTrue(os.path.isfile(self.stranger), "not a bake: never touched")

    def test_a_dry_run_removes_nothing(self):
        self._scene(self.kept)
        removed = self.bakes.purge(self.proj, (".ma",), dry_run=True)
        self.assertEqual(len(removed), 2)
        self.assertTrue(os.path.isfile(self.unused))

    def test_a_live_reference_keeps_its_bake(self):
        self._scene()
        removed = self.bakes.purge(self.proj, (".ma",), live=[self.unused])
        self.assertNotIn(self.unused, removed)
        self.assertIn(self.kept, removed)

    def test_the_cache_folder_itself_is_not_scanned(self):
        """A bake naming another (its record, a sidecar) is not a reference."""
        self._file(
            "cache", "bridge", "index.ma", data=os.path.basename(self.unused).encode()
        )
        removed = self.bakes.purge(self.proj, (".ma",))
        self.assertIn(self.unused, removed)

    def test_an_unscannable_file_stops_the_purge(self):
        self._scene(self.kept)
        with unittest.mock.patch.object(
            ptk.FileDependencies, "reference_bytes", return_value=None
        ):
            self.assertEqual(self.bakes.purge(self.proj, (".ma",)), [])
        self.assertTrue(os.path.isfile(self.unused))

    def test_a_corrupt_zstd_scene_stops_the_purge_without_raising(self):
        """A decoder's own error (``ZstdError`` is an ``Exception``, not an
        ``OSError``) reads as "cannot be scanned": the purge refuses."""
        import types

        class ZstdError(Exception):
            pass

        class _Stream:
            def read(self, *a):
                raise ZstdError("Unknown frame descriptor")

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        class _Decompressor:
            def stream_reader(self, source, **kw):
                source.close()
                return _Stream()

        self._scene(self.kept)
        self._file("scenes", "shot.blend", data=b"\x28\xb5\x2f\xfd" + b"\x00" * 8)
        fake = types.SimpleNamespace(
            ZstdDecompressor=_Decompressor, ZstdError=ZstdError
        )
        with unittest.mock.patch.dict(
            "sys.modules", {"compression": None, "zstandard": fake}
        ):
            self.assertEqual(self.bakes.purge(self.proj, (".ma", ".blend")), [])
        self.assertTrue(os.path.isfile(self.unused))

    def test_no_cache_folder_purges_nothing(self):
        empty = os.path.join(self.tmp_store.dir_path(), "empty")
        os.makedirs(empty)
        self.assertEqual(self.bakes.purge(empty, (".ma",)), [])

    def test_the_name_pattern_captures_the_slot_key(self):
        key = ptk.BakeCache.slot_key(self.src, self.proj)
        match = self.bakes.name_pattern.match(os.path.basename(self.kept))
        self.assertEqual(match.group(1), key)
        self.assertIsInstance(self.bakes.name_pattern, re.Pattern)


if __name__ == "__main__":
    unittest.main()
