# !/usr/bin/python
# coding=utf-8
"""SceneRecords -- the one declaration of scene metadata, and the machinery
that stores, assembles and commits it.

Runs against an in-memory store: the DCC stores are thin and pinned by their
own suites; everything about records, ordering, contexts and the commit is
decided here, once.
"""

import json
import logging
import math
import os
import pathlib
import sys
import unittest
from typing import Any, Dict
from unittest import mock

from pythontk.core_utils.engines.scene_export.export_snapshot import (
    ExportContext,
    ExportSnapshot,
)
from pythontk.core_utils.engines.scene_export.handoff_record import HandoffRecord
from pythontk.core_utils.engines.scene_export.record_transfer import (
    RecordTransfer,
    TransferContext,
)
from pythontk.core_utils.engines.scene_export.scene_records import (
    Clause,
    Kind,
    Merge,
    RecordSpec,
    SceneRecords as SR,
    Scope,
)
from pythontk.core_utils.engines.scene_export.scene_store import SceneStoreBase


class DictStore(SceneStoreBase):
    """The smallest store there is: one dict per scope, created on first write."""

    data: Dict[Scope, Dict[str, Any]] = {}

    @classmethod
    def reset(cls):
        cls.data = {}

    @classmethod
    def read(cls, scope, key):
        value = cls.data.get(Scope(scope), {}).get(key)
        return value if isinstance(value, str) and value else None

    @classmethod
    def write(cls, scope, key, text):
        scope = Scope(scope)
        if not text:
            group = cls.data.get(scope)
            if group is None or key not in group:
                return None
            group[key] = ""
            return cls.name(scope)
        cls.data.setdefault(scope, {})[key] = text
        return cls.name(scope)

    @classmethod
    def values(cls, scope):
        return dict(cls.data.get(Scope(scope), {}))


class SceneRecordsCase(unittest.TestCase):
    def setUp(self):
        DictStore.reset()


# -- the registry ---------------------------------------------------------------


class TestRegistry(SceneRecordsCase):
    def test_keys_are_unique_per_scope(self):
        seen = set()
        for spec in SR.all():
            self.assertNotIn((spec.scope, spec.key), seen, spec.key)
            seen.add((spec.scope, spec.key))

    def test_every_dependency_names_a_declared_record(self):
        keys = {s.key for s in SR.all()}
        for spec in SR.all():
            for dep in spec.after:
                self.assertIn(dep, keys, f"{spec.key} depends on unknown {dep!r}")

    def test_by_key_prefers_the_deliverable_when_a_key_exists_in_both(self):
        """A bare key means the shipped record where one key names a record in
        each scope -- as ``emissive_groups`` did until the scene's registry
        took a key of its own (2026-10-20); two specs share one here."""
        private = RecordSpec("k", Scope.PRIVATE, 1, "t", "d")
        shipped = RecordSpec("k", Scope.DELIVERABLE, 1, "t", "d")
        with mock.patch.object(SR, "all", return_value=[private, shipped]):
            self.assertIs(SR.by_key("k"), shipped)
            self.assertIs(SR.by_key("k", Scope.PRIVATE), private)
        self.assertIs(SR.by_key(SR.EMISSIVE_REGISTRY.key), SR.EMISSIVE_REGISTRY)
        self.assertIsNone(SR.by_key("nope"))
        with self.assertRaises(KeyError):
            SR.resolve("nope")

    def test_ordered_puts_a_reader_after_what_it_reads(self):
        order = SR.ordered([SR.VISIBILITY, SR.AUDIO, SR.SHOTS])
        self.assertLess(order.index(SR.SHOTS), order.index(SR.AUDIO))
        self.assertLess(order.index(SR.SHOTS), order.index(SR.VISIBILITY))

    def test_ordered_follows_dependencies_against_declaration_order(self):
        """The registry's own records are DECLARED in dependency order, so they
        cannot prove ``after`` is honoured; two unregistered specs handed in
        reader-first can."""
        d = RecordSpec("d", Scope.DELIVERABLE, 1, "t", "d")
        r = RecordSpec("r", Scope.DELIVERABLE, 1, "t", "d", after=("d",))
        self.assertEqual(SR.ordered([r, d]), [d, r])

    def test_the_shot_record_states_no_range_and_a_version_1_is_refused(self):
        """One take per shot (version 3, 2026-11-05; 2 is skipped): the takes are the
        timing, so the record keeps what a clip cannot hold, and no reader
        reads take ranges from it -- they are the store's at export time
        (``ShotStore.declared_takes``) and the file's own after it. A version-1
        record, which stated ranges, is no current record: its reader refuses
        it, and the cure is a re-export."""
        import pythontk

        self.assertEqual(SR.SHOTS.version, 3)
        v1 = {"version": 1, "fps": 24.0, "shots": [{"clip": "A", "start": 1, "end": 2}]}
        self.assertFalse(SR.SHOTS.is_current(v1))
        self.assertTrue(
            SR.SHOTS.is_current({"version": 3, "fps": 24.0, "shots": [{"clip": "A"}]})
        )
        self.assertNotIn("declared_takes", vars(SR))
        self.assertEqual(
            pythontk.MOVED_NAMES["SceneRecords.declared_takes"][0],
            "pythontk.ShotStore.declared_takes",
        )
        self.assertIn("takes, one per shot", SR.SHOTS.assembly)
        for retired in ("'start'", "'end'", "clip_mode", "zero_frame", "default_clip"):
            self.assertNotIn(retired, SR.SHOTS.assembly)

    def test_the_shot_record_is_version_3_and_2_is_refused(self):
        """The record rides a GLB under the key HEAD's published GLB block
        rode (``extras.animation``), and that block was version 2: two shapes
        under one key and version would be read half-way. So the record is
        version 3 in both containers; version 1 (HEAD's FBX record) and
        version 2 (HEAD's GLB block, or this record before it took the rate)
        are no current record, and a reader refuses them -- the cure, a
        re-export. Added: 2026-11-05."""
        self.assertEqual(SR.SHOTS.version, 3)
        head_block = {
            "version": 2,
            "fps": 24.0,
            "shots": [{"clip": "A", "start": 1, "end": 10}],
            "clips": [{"name": "A", "animation": 0, "zero_frame": 1}],
            "default_clip": "A",
        }
        for old in (head_block, {"version": 2, "fps": 24.0, "shots": [{"clip": "A"}]}):
            with self.subTest(old=sorted(old)):
                self.assertFalse(SR.SHOTS.is_current(old))
                (problem,) = SR.problems(SR.SHOTS, old)
                self.assertIn("v2 is not the v3", problem)
                self.assertIsNone(SR.glb_block({"extras": {SR.SHOTS.key: old}}, SR.SHOTS))
        self.assertIn("2", SR.shape(SR.SHOTS).describe()[0].help)

    def test_the_shot_record_rides_a_glb_as_it_stands(self):
        """One layout, two containers (2026-11-05): the GLB carries the
        ``animation`` record itself -- its version and shape the FBX's, under
        its key in the root ``extras`` -- rather than a block bound to the
        file's animation indices, and states the frame rate glTF cannot hold
        (``fps``), which a reader with a native rate (Unity, off the FBX)
        passes over. The retired block's types are moved names."""
        import pythontk

        self.assertTrue(SR.SHOTS.glb_verbatim)
        self.assertIn(SR.SHOTS, SR.glb_records())
        self.assertIn("'fps'", SR.SHOTS.assembly)
        record = {"version": 3, "fps": 24.0, "shots": [{"clip": "A"}]}
        gltf = {"extras": {SR.SHOTS.key: record}}
        self.assertIs(SR.glb_block(gltf, SR.SHOTS), record)
        self.assertEqual(SR.problems(SR.SHOTS, record), [])
        self.assertEqual(SR.put_glb_block({}, SR.SHOTS, record), record)
        for name in ("AnimationBlock", "AnimationBlockClip"):
            self.assertEqual(pythontk.MOVED_NAMES[name][0], "pythontk.AnimationRecord")

    def test_ordered_does_not_wait_for_a_record_that_is_not_in_the_set(self):
        self.assertEqual(SR.ordered([SR.VISIBILITY]), [SR.VISIBILITY])

    def test_ordered_refuses_a_cycle(self):
        a = RecordSpec("a", Scope.DELIVERABLE, 1, "t", "d", after=("b",))
        b = RecordSpec("b", Scope.DELIVERABLE, 1, "t", "d", after=("a",))
        with self.assertRaises(ValueError):
            SR.ordered([a, b])

    def test_check_producers_rejects_private_keys_and_the_handoff(self):
        self.assertEqual(
            SR.check_producers({SR.SHOTS: None, SR.LIGHTMAPS.key: None}),
            [SR.SHOTS, SR.LIGHTMAPS],
        )
        with self.assertRaises(KeyError, msg="a key a record was renamed from"):
            SR.check_producers({"lightmap_metadata": None})
        with self.assertRaises(ValueError):
            SR.check_producers({SR.SHOT_STORE: None})
        with self.assertRaises(ValueError):
            SR.check_producers({SR.HANDOFF: None})
        with self.assertRaises(KeyError):
            SR.check_producers({"shot_metadat": None})

    def test_a_renamed_or_retired_key_is_no_record_of_its_own(self):
        """Every record takes its GLB block's concept name, and the scene's
        emissive registry a key of its own beside the manifest's (2026-10-20).
        A key a scene stored a record under before that (:attr:`RecordSpec.
        renamed_from`) is no record of that scope's -- it resolves to nothing
        there and names one record at most, the one a migration moves it to
        -- and one retired whole (:attr:`SceneRecords.RETIRED`) is no record
        at all."""
        renamed = {
            (spec.scope, old): spec for spec in SR.all() for old in spec.renamed_from
        }
        self.assertEqual(
            renamed,
            {
                (Scope.DELIVERABLE, "shot_metadata"): SR.SHOTS,
                (Scope.DELIVERABLE, "audio_manifest"): SR.AUDIO,
                (Scope.DELIVERABLE, "lightmap_metadata"): SR.LIGHTMAPS,
                (Scope.DELIVERABLE, "shadow_metadata"): SR.SHADOWS,
                (Scope.DELIVERABLE, "visibility_tracks"): SR.VISIBILITY,
                (Scope.PRIVATE, "emissive_groups"): SR.EMISSIVE_REGISTRY,
            },
        )
        self.assertEqual(sum(len(spec.renamed_from) for spec in SR.all()), 6)
        for scope, old in renamed:
            with self.subTest(key=old, scope=scope.value):
                self.assertIsNone(SR.by_key(old, scope))
        self.assertEqual(SR.RETIRED, ("fbx_takes", "emissive_groups"))
        # A retired record a reader still meets names its fix.
        self.assertLessEqual(set(SR.RETIRED_FIXES), set(SR.RETIRED))
        current = {spec.key for spec in SR.all()}
        for old in SR.RETIRED:
            with self.subTest(key=old):
                self.assertNotIn(old, current)
                self.assertIsNone(SR.by_key(old))

    def test_describe_is_one_row_per_record(self):
        rows = SR.describe()
        self.assertEqual(len(rows), len(SR.all()))
        self.assertEqual(rows[0]["key"], SR.MATERIALS.key)
        self.assertIn("description", rows[0])
        self.assertNotIn("fbx_takes", [r["key"] for r in rows], "retired, no record")
        self.assertNotIn("emissive_groups", [r["key"] for r in rows], "retired, no record")
        plain = next(r for r in rows if r["key"] == SR.LIGHTMAP_PROBE.key)
        self.assertEqual(plain["version"], SR.LIGHTMAP_PROBE.version)
        self.assertNotIn("version_key", plain, "every record versions alike")
        shots = next(r for r in rows if r["key"] == SR.SHOTS.key)
        self.assertEqual(shots["glb"], SR.SHOTS.version, "the record itself")
        self.assertIsNone(plain["glb"])


# -- the contracts other languages read ------------------------------------------


class TestRegisteredHooks(SceneRecordsCase):
    """A tool outside a library registers its record hooks as entry points,
    so an installed tool is found without being imported, and a batch export
    that imports only the library still writes its record."""

    @staticmethod
    def _points(*pairs):
        """Stand-ins for ``importlib.metadata.EntryPoint``: a name and a value."""
        return [type("EP", (), {"name": n, "value": v})() for n, v in pairs]

    def _hooks(self, *pairs, host="maya", role="producer"):
        SR.forget_registered_hooks()
        self.addCleanup(SR.forget_registered_hooks)
        with mock.patch(
            "pythontk.core_utils.package_manager.PackageManager.entry_points",
            return_value=self._points(*pairs),
        ) as points:
            found = SR.registered_hooks(host, role)
            again = SR.registered_hooks(host, role)
        points.assert_called_once_with(SR.HOOK_GROUP)  # read once a session
        self.assertEqual(found, again)
        return found

    def test_rows_are_read_from_metadata_alone(self):
        found = self._hooks(
            ("maya.producer.lightmaps", "tentacle.tools.lightmaps.maya:LightmapRecords.export_record"),
            ("blender.producer.lightmaps", "tentacle.tools.lightmaps.blender:Records.export_record"),
            ("maya.owner.shot_store", "tentacle.tools.shots.maya:ShotStore"),
            ("maya.stager.effects.prepare", "tentacle.fx:Effects.prepare_for_export [web]"),
            ("not-a-hook", "x:y"),
        )
        self.assertEqual(
            found,
            {"lightmaps": ("tentacle.tools.lightmaps.maya", "LightmapRecords", "export_record")},
        )

    def test_an_owner_names_a_class_and_a_stager_keeps_its_slot(self):
        self.assertEqual(
            self._hooks(("maya.owner.shot_store", "tentacle.tools.shots.maya:ShotStore"), role="owner"),
            {"shot_store": ("tentacle.tools.shots.maya", "ShotStore", None)},
        )
        self.assertEqual(
            self._hooks(("maya.stager.effects.prepare", "tentacle.fx:Effects.prepare_for_export [web]"), role="stager"),
            {"effects.prepare": ("tentacle.fx", "Effects", "prepare_for_export")},
        )

    def test_a_hosts_producers_take_registered_rows_under_its_own(self):
        table = {SR.MATERIALS: ("mayatk.mat", "MaterialRecords", "export_record")}
        hooks = {
            "materials": ("tentacle.mat", "Other", "export_record"),  # the library's wins
            "lightmaps": ("tentacle.lm", "LightmapRecords", "export_record"),
            "no_such_record": ("tentacle.x", "X", "export_record"),  # skipped
        }
        with mock.patch.object(SR, "registered_hooks", return_value=hooks) as reg:
            rows = SR.producer_rows(table, "maya")
        reg.assert_called_once_with("maya", "producer")
        self.assertEqual(
            rows,
            {
                SR.MATERIALS: ("mayatk.mat", "MaterialRecords", "export_record"),
                SR.LIGHTMAPS: ("tentacle.lm", "LightmapRecords", "export_record"),
            },
        )

    def test_a_registered_stager_pairs_its_two_slots_on_one_class(self):
        table = {"effects": ("mayatk.fx", "Effects", "prepare_for_export", "finish_export")}
        hooks = {
            "hints.prepare": ("tentacle.lm", "LightmapRecords", "migrate_folder_hints"),
            "proxies.prepare": ("tentacle.fx", "Proxies", "stage"),
            "proxies.finish": ("tentacle.fx", "Proxies", "unstage"),
            "split.prepare": ("tentacle.a", "A", "go"),
            "split.finish": ("tentacle.b", "B", "stop"),  # two classes: skipped
            "effects.prepare": ("tentacle.fx", "Effects", "other"),  # the library's wins
        }
        with mock.patch.object(SR, "registered_hooks", return_value=hooks):
            rows = SR.stager_rows(table, "maya")
        self.assertEqual(
            rows,
            {
                "effects": ("mayatk.fx", "Effects", "prepare_for_export", "finish_export"),
                "hints": ("tentacle.lm", "LightmapRecords", "migrate_folder_hints", None),
                "proxies": ("tentacle.fx", "Proxies", "stage", "unstage"),
            },
        )

    def test_a_single_hook_is_the_librarys_row_else_the_registered_one(self):
        own = ("mayatk.stash", "KeyStash", "active")
        hooks = {
            "key_stash": ("tentacle.stash", "KeyStash", "active"),
            "other": ("tentacle.other", "Other", "active"),  # second: warned, unused
        }
        with mock.patch.object(SR, "registered_hooks", return_value=hooks) as reg:
            self.assertEqual(SR.hook_row(own, "maya", "take_stash"), own)
            reg.assert_not_called()
            with self.assertLogs(level="WARNING"):
                self.assertEqual(
                    SR.hook_row(None, "maya", "take_stash"),
                    ("tentacle.stash", "KeyStash", "active"),
                )
            reg.assert_called_with("maya", "take_stash")
        with mock.patch.object(SR, "registered_hooks", return_value={}):
            self.assertIsNone(SR.hook_row(None, "maya", "take_stash"))

    def test_a_stager_slot_without_a_method_is_skipped(self):
        hooks = {
            "half.prepare": ("tentacle.fx", "Fx", None),  # module:Class, no method
            "half.finish": ("tentacle.fx", "Fx", "finish"),
        }
        with mock.patch.object(SR, "registered_hooks", return_value=hooks):
            with self.assertLogs(level="WARNING"):
                self.assertEqual(SR.stager_rows({}, "maya"), {})

    def test_one_entry_name_from_two_distributions_keeps_the_first_and_warns(self):
        with self.assertLogs(level="WARNING") as logs:
            found = self._hooks(
                ("maya.producer.lightmaps", "first.mod:Records.export_record"),
                ("maya.producer.lightmaps", "second.mod:Records.export_record"),
            )
        self.assertEqual(found, {"lightmaps": ("first.mod", "Records", "export_record")})
        self.assertIn("second.mod", "".join(logs.output))

    def test_the_session_read_is_redone_when_the_path_changes(self):
        SR.forget_registered_hooks()
        self.addCleanup(SR.forget_registered_hooks)
        target = "pythontk.core_utils.package_manager.PackageManager.entry_points"
        with mock.patch(target, return_value=[]) as points:
            SR.registered_hooks("maya", "producer")
            SR.registered_hooks("maya", "producer")
            self.assertEqual(points.call_count, 1)
            with mock.patch("sys.path", sys.path + ["O:/not/a/real/site"]):
                SR.registered_hooks("maya", "producer")
            self.assertEqual(points.call_count, 2)

    def test_a_store_merges_registered_owners_under_its_own(self):
        class HostStore(DictStore):
            HOST = "maya"
            OWNERS = {"shot_store": ("mayatk.shots", "ShotStore")}

        hooks = {
            "shot_store": ("tentacle.shots", "ShotStore", None),
            "key_stash": ("tentacle.stash", "KeyStash", None),
        }
        with mock.patch.object(SR, "registered_hooks", return_value=hooks) as reg:
            rows = HostStore.owner_rows()
        reg.assert_called_once_with("maya", "owner")
        self.assertEqual(
            rows,
            {
                "key_stash": ("tentacle.stash", "KeyStash"),
                "shot_store": ("mayatk.shots", "ShotStore"),  # the library's own row wins
            },
        )
        self.assertEqual(DictStore.owner_rows(), {})  # no host: nothing registered is read


class TestContracts(SceneRecordsCase):
    """Shapes and the records a GLB carries: what another language reads is
    declared here and everything that spelled it reads the declaration."""

    def test_a_glb_carries_a_records_block_under_the_records_key(self):
        """One name per record in every file (2026-10-20): the block a GLB
        carries is the record's, under its key -- ``render_effects`` among them
        since its v2 (2026-11-05), the record itself."""
        records = SR.glb_records()
        self.assertEqual(
            [s.key for s in records],
            ["animation", "audio", "lightmaps", "shadows", "articulation", "render_effects"],
        )
        self.assertEqual(
            records,
            [SR.SHOTS, SR.AUDIO, SR.LIGHTMAPS, SR.SHADOWS, SR.ARTICULATION, SR.VISIBILITY],
        )
        self.assertTrue(all(s.scope is Scope.DELIVERABLE for s in records))
        # Each as it stands (``glb_verbatim``): the one way a GLB carries one.
        self.assertTrue(all(s.glb_verbatim for s in records))

    def test_the_glb_block_machinery_is_retired_whole(self):
        """Articulation v2 (2026-11-08) was the last record a GLB carried as a
        block bound to its indices, so the machinery goes whole: no record
        declares a block (``RecordSpec.glb``, ``GlbBlock``, ``WebProjection``),
        a block version or shape of its own (``SceneRecords.glb_version``,
        ``glb_shape``) or problems apart from its record's
        (``block_problems``). Each is a moved name, its home one that
        resolves; ``glb_verbatim`` stays, since a record a GLB does not carry
        opts out: the deliverables it states otherwise -- ``materials`` (a
        GLB's own materials), the ``handoff`` (the GLB's own) -- and every
        private record. Added: 2026-11-08"""
        import dataclasses

        import pythontk

        self.assertNotIn("glb", {f.name for f in dataclasses.fields(RecordSpec)})
        for name in ("glb_shape", "glb_version", "block_problems"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(SR, name))
                self.assertIn(f"SceneRecords.{name}", pythontk.MOVED_NAMES)
        self.assertEqual(
            pythontk.MOVED_NAMES["RecordSpec.glb"], ("pythontk.RecordSpec.glb_verbatim", "0.21.0")
        )
        # The names v0.20.0 noticed for 0.18.0 stay, deprecated, until their
        # window closes (TestRetiredUntil022): held, then moved at 0.22.0.
        for name in ("GlbBlock", "WebProjection", "RecordSpec.web"):
            with self.subTest(held=name):
                self.assertNotIn(name, pythontk.MOVED_NAMES)
                home, version = pythontk.HELD_UNTIL_0_22[name]
                self.assertEqual((home, version), ("pythontk.RecordSpec.glb_verbatim", "0.22.0"))
        self.assertEqual(
            pythontk.HELD_UNTIL_0_22["SceneRecords.web_shape"][0], "pythontk.SceneRecords.shape"
        )
        with self.assertRaises(TypeError):
            RecordSpec("x", Scope.DELIVERABLE, 1, "t", "d", glb=object())
        self.assertFalse(SR.MATERIALS.glb_verbatim, "a deliverable a GLB does not carry")
        self.assertIn(SR.MATERIALS, SR.deliverable())
        self.assertNotIn(SR.MATERIALS, SR.glb_records())
        self.assertFalse(SR.HANDOFF.glb_verbatim, "the GLB states its own handoff")
        self.assertTrue(
            all(not s.glb_verbatim for s in SR.all() if s.scope is Scope.PRIVATE)
        )
        # Each moved name's home resolves (glb_version's: SceneRecords.resolve,
        # whose spec's version a block's version is).
        import importlib

        moved = {**pythontk.MOVED_NAMES, **pythontk.HELD_UNTIL_0_22}
        for name in ("GlbBlock", "WebProjection", "RecordSpec.glb", "RecordSpec.web") + tuple(
            f"SceneRecords.{n}" for n in ("glb_shape", "glb_version", "block_problems", "web_shape")
        ):
            with self.subTest(home_of=name):
                parts = moved[name][0].split(".")
                obj = importlib.import_module(parts[0])
                for part in parts[1:]:
                    obj = getattr(obj, part)

    def test_a_glb_carries_the_articulation_record_itself_and_a_v1_is_refused(self):
        """Articulation v2 (2026-11-08): one layout, two containers -- a GLB
        carries the record as it stands, in glTF's frame (metres), its
        joints and grabbed parts bound by name; the index-bound block
        retired. A v1 record or block is no current record: a reader refuses
        it, and its version's help names the fix, a re-export."""
        spec = SR.ARTICULATION
        self.assertTrue(spec.glb_verbatim)
        self.assertTrue(spec.in_frame)
        self.assertEqual(spec.version, 2)
        self.assertEqual(SR.shape(spec).__name__, "ArticulationRecord")
        self.assertIn("anchor", spec.assembly)
        self.assertIn("metres", spec.assembly)
        self.assertNotIn("unit the ratio", spec.assembly)
        v1_record = {"version": 1, "rigs": [{"name": "a", "joints": [], "grab": []}]}
        v1_block = {"version": 1, "rigs": [{"name": "a", "joints": [{"node": 0}]}]}
        for old in (v1_record, v1_block):
            with self.subTest(old=sorted(old["rigs"][0])):
                self.assertFalse(spec.is_current(old))
                (problem,) = SR.problems(spec, old)
                self.assertIn("v1 is not the v2", problem)
                self.assertIsNone(SR.glb_block({"extras": {spec.key: old}}, spec))
        (version,) = [f for f in SR.shape(spec).describe() if f.name == "version"]
        self.assertIn("version 1 record", version.help)
        self.assertIn("re-exported", version.help)

    def test_a_glb_carries_the_lightmaps_record_itself(self):
        """Lightmaps v3 (2026-11-07): one layout, two containers -- a GLB
        carries the record as it stands, under its key in the root extras,
        its version and shape the record's own; the index-bound block (a
        carrier slot, materials by name, the probe by bufferView) retired.
        The files it names ride the media transport (``GLB_MEDIA``)."""
        spec = SR.LIGHTMAPS
        self.assertTrue(spec.glb_verbatim)
        self.assertEqual(spec.version, 3)
        self.assertEqual(SR.shape(spec).__name__, "LightmapRecord")
        self.assertNotIn("uvIndex", spec.description)

    def test_a_glb_carries_the_shadows_record_itself_and_a_v2_is_refused(self):
        """Shadows v3 (2026-11-07): one layout, two containers -- a GLB carries
        the record as it stands, its version and shape the record's own, the
        index-bound block (planes, sources, contacts and maps by index)
        retired, and the files it names ride the media transport. A v2
        record or block is no current record: a reader refuses it, and its
        version's help names the fix, a re-export."""
        spec = SR.SHADOWS
        self.assertTrue(spec.glb_verbatim)
        self.assertEqual(spec.version, 3)
        self.assertEqual(SR.shape(spec).__name__, "ShadowRecord")
        self.assertIn("shadows record names", SR.GLB_MEDIA.text)
        v2_record = {"version": 2, "unit_scale": 0.01, "planes": [{"type": "projected"}]}
        v2_block = {"version": 2, "unit_scale": 0.01, "planes": [{"node": 0}]}
        for old in (v2_record, v2_block):
            with self.subTest(old=sorted(old["planes"][0])):
                self.assertFalse(spec.is_current(old))
                (problem,) = SR.problems(spec, old)
                self.assertIn("v2 is not the v3", problem)
                self.assertIsNone(SR.glb_block({"extras": {spec.key: old}}, spec))
        (version,) = [f for f in SR.shape(spec).describe() if f.name == "version"]
        self.assertIn("version 2 record", version.help)
        self.assertIn("re-exported", version.help)

    def test_glb_block_reads_the_current_block_alone(self):
        """``glb_block``: a record a GLB carries, in a document's root
        ``extras``, under the record's key, at the record's version -- the
        block itself, so a pass editing it edits the document. Another
        version, an older one too, is none: a reader refuses rather than
        guesses, and the cure is a re-export. Added: 2026-10-20"""
        spec = SR.ARTICULATION
        block = {"version": spec.version, "rigs": []}
        gltf = {"extras": {spec.key: block}}
        self.assertIs(SR.glb_block(gltf, spec), block)
        for version in (spec.version - 1, spec.version + 1, "2", None):
            with self.subTest(version=version):
                stale = {"extras": {spec.key: {"version": version, "rigs": []}}}
                self.assertIsNone(SR.glb_block(stale, spec))
        for elsewhere in (
            {"extras": {spec.key: json.dumps(block)}},
            {"scenes": [{"extras": {spec.key: block}}]},
            {"extras": None},
            {},
            None,
        ):
            with self.subTest(gltf=elsewhere):
                self.assertIsNone(SR.glb_block(elsewhere, spec))
        self.assertIsNone(
            SR.glb_block({"extras": {SR.HANDOFF.key: {"version": 3}}}, SR.HANDOFF),
            "no record a GLB carries as it stands",
        )
        materials = {"version": SR.MATERIALS.version, "materials": []}
        self.assertIsNone(
            SR.glb_block({"extras": {SR.MATERIALS.key: materials}}, SR.MATERIALS),
            "a GLB states its materials as glTF's own",
        )

    def test_put_glb_block_writes_what_glb_block_reads(self):
        """``put_glb_block``: a record written where ``glb_block`` reads it --
        the document's root ``extras``, under the record's key -- its
        record's version stamped (a writer's own wins) and pruned to its
        declared shape as ``make`` prunes a record, each field it cut warned
        at the address the file holds it. Added: 2026-10-20"""
        spec = SR.ARTICULATION
        gltf = {"extras": None}
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ) as logs:
            written = SR.put_glb_block(
                gltf,
                spec,
                {
                    "encoding": "srgb",
                    "rigs": [{"name": "Arm", "stray": True}],
                },
            )
        self.assertEqual(
            written,
            {"version": spec.version, "rigs": [{"name": "Arm"}]},
        )
        self.assertIs(SR.glb_block(gltf, spec), written)
        (line,) = logs.output
        self.assertIn(
            f"extras.{spec.key}: left out what its shape does not declare "
            "(encoding, rigs[].stray).",
            line,
        )
        # A writer's own version wins -- and its reader reads none of another.
        older = SR.put_glb_block(
            gltf, spec, {"version": spec.version - 1, "rigs": []}
        )
        self.assertEqual(older["version"], spec.version - 1)
        self.assertIsNone(SR.glb_block(gltf, spec))

    def test_the_retired_web_names_still_serve(self):
        """``SceneRecords.web_projected`` and ``SceneRecords.FBX_TAKES``
        (retired 2026-10-20, removed in 0.18.0) warn, and serve what replaced
        them; the block names beside them retired whole with the GLB block
        machinery (2026-11-08, ``test_the_glb_block_machinery_is_retired_whole``)."""
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(SR.web_projected(), SR.glb_records())
        with self.assertWarns(DeprecationWarning):
            takes = SR.FBX_TAKES
        self.assertIsInstance(takes, RecordSpec)
        self.assertIn(takes.key, SR.RETIRED)
        self.assertNotIn(takes, SR.all(), "no record of the registry's")

    def test_every_declared_shape_resolves_to_a_typed_schema(self):
        from pythontk import SchemaSpec

        for spec in SR.all():
            for shape in (SR.shape(spec),):
                if shape is None:
                    continue
                self.assertTrue(issubclass(shape, SchemaSpec), spec.key)
                self.assertTrue(shape.TYPED, f"{spec.key}: {shape.__name__}")
                shape.json_schema()  # one name per schema, annotations agree
        from pythontk.core_utils.engines.scene_export.handoff_record import (
            HandoffBlock,
        )
        from pythontk.file_utils.mesh_convert.render_effects_record import (
            RenderEffectsRecord,
        )

        self.assertIsNotNone(SR.shape(SR.ARTICULATION))
        self.assertIs(SR.shape("articulation"), SR.shape(SR.ARTICULATION))
        self.assertIs(SR.shape(SR.HANDOFF), HandoffBlock)
        self.assertIs(SR.shape(SR.VISIBILITY), RenderEffectsRecord)
        undeclared = RecordSpec("x", Scope.DELIVERABLE, 1, "t", "d")
        self.assertIsNone(SR.shape(undeclared))

    def test_problems_hold_a_payload_to_its_declaration(self):
        """``SceneRecords.problems``: what a payload breaks of its record's
        declaration -- the version a reader reads, then the declared shape --
        a line each, and nothing when a reader reads it. A payload no shape
        declares is held to its version alone. Added: 2026-10-20"""
        spec = SR.VISIBILITY
        current = {"version": spec.version, "tracks": [{"node": "a", "channels": ["opacity"]}]}
        self.assertEqual(SR.problems(spec, current), [])
        self.assertEqual(SR.problems(spec.key, current), [], "a key reads as its spec")
        self.assertEqual(SR.problems(spec, ["a"]), [f"{spec.key}: not an object"])
        for version in (spec.version - 1, spec.version + 1, None):
            with self.subTest(version=version):
                self.assertEqual(
                    SR.problems(spec, dict(current, version=version)),
                    [
                        f"{spec.key}: v{version} is not the v{spec.version} a reader reads"
                    ],
                )
        # A line names where the problem is, then what it is, behind a colon.
        self.assertEqual(
            SR.problems(spec, dict(current, tracks=[{"channels": ["visibility"]}])),
            [f"{spec.key}.tracks[0]: missing required key 'node'"],
        )
        self.assertEqual(
            SR.problems(spec, {"version": spec.version}),
            [f"{spec.key}: missing required key 'tracks'"],
        )
        self.assertEqual(
            SR.problems(spec, dict(current, tracks=["a"])),
            [f"{spec.key}.tracks[0]: expected a mapping, got str"],
        )
        undeclared = RecordSpec("x", Scope.DELIVERABLE, 1, "t", "d")
        self.assertEqual(SR.problems(undeclared, {"version": 1, "any": [1]}), [])

    def test_the_preview_activates_its_scripts_on_record_keys(self):
        """Every key a packaged script is activated on is a record's key, the
        one its block rides under."""
        from pythontk import PreviewServer

        declared = {s.key for s in SR.glb_records()}
        self.assertLessEqual(set(PreviewServer.AUTO_SCRIPTS.values()), declared)
        self.assertEqual(PreviewServer.AUTO_SCRIPTS["shadow_rig"], SR.SHADOWS.key)


# -- the codec ------------------------------------------------------------------


class TestCodec(SceneRecordsCase):
    def test_enveloped_record_stamps_its_version(self):
        rec = SR.LIGHTMAPS.make({"objects": []})
        self.assertEqual(rec.payload["version"], SR.LIGHTMAPS.version)
        self.assertEqual(json.loads(rec.text)["version"], SR.LIGHTMAPS.version)

    def test_a_producer_may_declare_a_version_itself(self):
        rec = SR.SHADOWS.make({"version": 1, "planes": []})
        self.assertEqual(rec.payload["version"], 1)

    def test_enveloped_record_needs_a_mapping(self):
        with self.assertRaises(TypeError):
            SR.LIGHTMAPS.make([1, 2])

    def test_a_record_ships_what_its_shape_declares_and_warns_what_it_cut(self):
        """A declared record's payload is pruned to its shape: what a
        producer adds beyond it is left out and warned once, each field named
        once however many entries carried it -- its drift said, never
        shipped. A record that declares no shape is taken as given.
        Added: 2026-10-20"""
        entry = {
            "name": "a",
            "map": "a.exr",
            "scaleOffset": [1, 1, 0, 0],
            "hierarchy": ["a"],
        }
        objects = [dict(entry, intensity=1.0), dict(entry, name="b", intensity=2.0)]
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ) as logs:
            rec = SR.LIGHTMAPS.make({"unit_scale": 1.0, "objects": objects, "x": 1})
        self.assertEqual(
            rec.payload,
            {
                "version": SR.LIGHTMAPS.version,
                "unit_scale": 1.0,
                "objects": [entry, dict(entry, name="b")],
            },
        )
        (line,) = logs.output
        self.assertIn(
            "lightmaps: left out what its shape does not declare "
            "(objects[].intensity, x).",
            line,
        )
        undeclared = RecordSpec("x", Scope.DELIVERABLE, 1, "t", "d")
        with self.assertNoLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ):
            rec = undeclared.make({"any": [1]})
        self.assertEqual(rec.payload, {"version": 1, "any": [1]})

    def test_every_record_versions_under_version(self):
        """Every enveloped record states its version under ``version``, so no
        record carries two versions; a newer one is refused."""
        for spec in SR.all():
            if spec.envelope:
                with self.subTest(record=spec.key):
                    self.assertEqual(spec.make({}).payload, {"version": spec.version})
        effects = SR.VISIBILITY
        rec = effects.make({"tracks": []})
        self.assertEqual(rec.payload, {"version": effects.version, "tracks": []})
        newer = json.dumps({"version": effects.version + 1, "tracks": []})
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ):
            self.assertIsNone(effects.decode(newer))

    def test_the_retired_version_key_is_version(self):
        """``RecordSpec.version_key`` (retired 2026-10-20, removed in 0.18.0)
        warns, and answers the one key every record versions under."""
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(SR.VISIBILITY.version_key, "version")

    def test_a_record_without_an_envelope_is_stored_as_given(self):
        rec = SR.SHOT_STORE.make([{"name": "a", "start": 1, "end": 2}])
        self.assertEqual(json.loads(rec.text), [{"name": "a", "start": 1, "end": 2}])

    def test_decode_tolerates_absent_cleared_and_corrupt(self):
        self.assertIsNone(SR.SHOTS.decode(None))
        self.assertEqual(SR.SHOTS.decode("", default=[]), [])
        self.assertEqual(SR.SHOTS.decode("{not json", default={}), {})

    def test_decode_refuses_a_newer_version(self):
        newer = json.dumps({"version": SR.SHOTS.version + 1, "shots": []})
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ):
            self.assertIsNone(SR.SHOTS.decode(newer))
        older = json.dumps({"version": 0, "shots": []})
        self.assertEqual(SR.SHOTS.decode(older)["shots"], [])

    def test_non_native_values_encode_as_their_string(self):
        rec = SR.LIGHTMAPS.make({"objects": [{"map": pathlib.PurePosixPath("a/b")}]})
        self.assertEqual(json.loads(rec.text)["objects"][0]["map"], "a/b")


# -- the store contract -----------------------------------------------------------


class TestStore(SceneRecordsCase):
    def test_save_load_round_trip(self):
        SR.LIGHTMAPS.save(DictStore, {"objects": [1]})
        self.assertEqual(SR.LIGHTMAPS.load(DictStore)["objects"], [1])
        self.assertTrue(SR.LIGHTMAPS.is_present(DictStore))

    def test_falsy_payload_clears_and_never_creates(self):
        self.assertIsNone(SR.LIGHTMAPS.save(DictStore, {}))
        self.assertEqual(DictStore.data, {}, "a clear must not create a carrier")
        SR.LIGHTMAPS.save(DictStore, {"objects": [1]})
        self.assertEqual(SR.LIGHTMAPS.save(DictStore, None), "data_export")
        self.assertIsNone(SR.LIGHTMAPS.load(DictStore))

    def test_scopes_do_not_mix(self):
        SR.EMISSIVE_REGISTRY.save(DictStore, {"slots": {}})
        self.assertIsNone(DictStore.read(Scope.DELIVERABLE, SR.EMISSIVE_REGISTRY.key))
        self.assertEqual(SR.EMISSIVE_REGISTRY.load(DictStore), {"slots": {}})

    def test_dump_groups_by_carrier_name_and_decodes(self):
        SR.SHOT_STORE.save(DictStore, {"shots": []})
        SR.LIGHTMAPS.save(DictStore, {"objects": []})
        DictStore.data[Scope.PRIVATE]["audio_clip_voice"] = 1  # a non-string attr
        SR.SHOT_STORE.clear(DictStore)  # cleared channel: skipped
        data = DictStore.dump()
        self.assertEqual(
            data,
            {
                "data_internal": {"audio_clip_voice": 1},
                "data_export": {
                    SR.LIGHTMAPS.key: {
                        "objects": [],
                        "version": SR.LIGHTMAPS.version,
                    }
                },
            },
        )
        self.assertEqual(
            DictStore.dump(decode=False)["data_export"][SR.LIGHTMAPS.key],
            json.dumps({"version": SR.LIGHTMAPS.version, "objects": []}),
        )
        self.assertEqual(
            json.loads(DictStore.format_dump())["data_export"][SR.LIGHTMAPS.key][
                "version"
            ],
            SR.LIGHTMAPS.version,
        )

    def test_format_dump_is_empty_for_an_empty_scene(self):
        self.assertEqual(DictStore.format_dump(), "")

    def test_channels_are_the_non_empty_strings_only(self):
        DictStore.data[Scope.DELIVERABLE] = {"a": "x", "b": "", "w": 0.5}
        self.assertEqual(DictStore.channels(Scope.DELIVERABLE), {"a": "x"})

    def test_remove_deletes_a_channel_and_says_whether_there_was_one(self):
        SR.SHOTS.save(DictStore, {"shots": []})
        self.assertTrue(DictStore.remove(Scope.DELIVERABLE, SR.SHOTS.key))
        self.assertFalse(SR.SHOTS.is_present(DictStore))
        self.assertFalse(DictStore.remove(Scope.DELIVERABLE, SR.SHOTS.key))

    def test_a_carrier_is_named_as_every_host_numbers_a_copy(self):
        """An import beside the scene's own carrier numbers the copy as its
        host does: Maya ``data_export1``, Blender ``data_export.001``. Both
        ride a written file, so a reader and the verifier both know them; the
        verifier was blind to Blender's. Added: 2026-10-28"""
        for name in (
            "data_export",
            "data_export1",
            "data_export.001",
            "|GRP|MOD:data_export",
            "MOD:data_export.002",
        ):
            with self.subTest(name=name):
                self.assertTrue(SceneStoreBase.is_carrier(name))
        for name in (
            "data_exportShape",
            "data_export_old",
            "data_export.",
            "data_export..1",
            "data_export.1a",
            "",
            None,
        ):
            with self.subTest(name=name):
                self.assertFalse(SceneStoreBase.is_carrier(name))
        self.assertTrue(SceneStoreBase.is_carrier("data_internal.001", Scope.PRIVATE))

    def test_the_scenes_carrier_is_the_root_level_data_export(self):
        """``SceneStoreBase.is_scene_carrier``: named exactly ``data_export``
        and at the root of the file -- a path through a parent, a namespace, a
        number, or a node its reader knows is not root level is a module's.
        Under the old rule (the name alone) a grouped module's
        ``Module|data_export`` read as the scene's. Added: 2026-11-08"""
        self.assertTrue(SceneStoreBase.is_scene_carrier("data_export"))
        self.assertTrue(SceneStoreBase.is_scene_carrier("|data_export"))
        for name, root_level in (
            ("Module|data_export", True),
            ("|GRP|data_export", True),
            ("data_export", False),
            ("MOD:data_export", True),
            ("data_export1", True),
            ("data_export.001", True),
            ("", True),
            (None, True),
        ):
            with self.subTest(name=name, root_level=root_level):
                self.assertFalse(SceneStoreBase.is_scene_carrier(name, root_level))
        self.assertTrue(SceneStoreBase.is_scene_carrier("data_internal", scope=Scope.PRIVATE))

    def test_a_reader_takes_the_scenes_own_else_the_first_modules(self):
        """``SceneStoreBase.reader_values``: the reader's precedence, once --
        each channel from the scene's own carrier, else the first module's in
        file order, a module's ``handoff`` never, a node no carrier never.
        Added: 2026-11-08"""
        handoff = SR.HANDOFF.key
        rows = [
            ("Module|data_export", True, "lightmaps", "module"),
            ("Module|data_export", True, handoff, "module"),
            ("Module|data_export", True, "audio", "module"),
            ("MOD:data_export", True, "audio", "second module"),
            ("MOD:data_export", True, "shadows", "second module"),
            ("Lamp_geo", True, "materials", "an artist's"),
            ("data_export", False, "animation", "under the converter's group"),
            ("data_export", True, "lightmaps", "scene"),
            ("data_export", True, "lightmaps", "scene, again"),
        ]
        self.assertEqual(
            SceneStoreBase.reader_values(rows),
            {
                "lightmaps": ("scene", True),
                "audio": ("module", False),
                "shadows": ("second module", False),
                "animation": ("under the converter's group", False),
            },
        )
        own = SceneStoreBase.reader_values(rows + [("data_export", True, handoff, "own")])
        self.assertEqual(own[handoff], ("own", True))


class TestMigrate(SceneRecordsCase):
    """``SceneStoreBase.migrate``: a scene's records brought to their current
    keys, once -- run when a scene opens and before every export commit.
    Added: 2026-10-20"""

    @staticmethod
    def _stored(scope, key, payload):
        DictStore.write(scope, key, json.dumps(payload))

    def test_a_record_under_a_key_it_was_renamed_from_moves_to_its_key(self):
        renamed = {old: spec for spec in SR.all() for old in spec.renamed_from}
        for old, spec in renamed.items():
            self._stored(spec.scope, old, {"version": spec.version, "was": old})
        self._stored(Scope.DELIVERABLE, "fbx_takes", [{"name": "A", "start": 1}])
        self._stored(Scope.DELIVERABLE, "emissive_groups", {"version": 2, "groups": []})
        self.assertEqual(sorted(DictStore.migrate()), sorted([*renamed, *SR.RETIRED]))
        for old, spec in renamed.items():
            with self.subTest(key=old):
                self.assertIsNone(DictStore.read(spec.scope, old))
                self.assertEqual(spec.load(DictStore)["was"], old)
        self.assertIsNone(DictStore.read(Scope.DELIVERABLE, "fbx_takes"), "retired")
        self.assertIsNone(DictStore.read(Scope.DELIVERABLE, "emissive_groups"), "retired")
        self.assertEqual(
            DictStore.migrate(), [], "a migrated scene has nothing to move"
        )

    def test_a_key_renamed_in_one_scope_moves_there_alone(self):
        """The scene's emissive registry took a key of its own (2026-10-20)
        from the private ``emissive_groups``, which the shipped record kept
        until it retired (2026-11-07): the private one moves to the
        registry's key, the deliverable one is cleared."""
        registry = SR.EMISSIVE_REGISTRY
        self.assertIn("emissive_groups", registry.renamed_from)
        self._stored(Scope.PRIVATE, "emissive_groups", {"groups": {"glow": {"slot": 0}}})
        self._stored(Scope.DELIVERABLE, "emissive_groups", {"version": 2, "groups": []})
        self.assertEqual(DictStore.migrate(), ["emissive_groups", "emissive_groups"])
        self.assertEqual(registry.load(DictStore), {"groups": {"glow": {"slot": 0}}})
        self.assertIsNone(DictStore.read(Scope.PRIVATE, "emissive_groups"))
        self.assertIsNone(DictStore.read(Scope.DELIVERABLE, "emissive_groups"))

    def test_an_emptied_old_channel_is_removed_too(self):
        """REGRESSION (2026-10-24): an old key whose channel a clear emptied
        (``""``, which both DCC stores keep) reads ``None``, so ``migrate``
        skipped it and the emptied channel stayed for good -- though each
        DCC store's ``remove`` deletes one outright. Every renamed and
        retired key goes through ``remove``."""

        class KeepingStore(DictStore):
            """Keeps an emptied channel, as the DCC stores do, and deletes
            one outright in ``remove``, as they do."""

            @classmethod
            def remove(cls, scope, key):
                return cls.data.get(Scope(scope), {}).pop(key, None) is not None

        old = SR.LIGHTMAPS.renamed_from[0]
        KeepingStore.data = {Scope.DELIVERABLE: {old: "", "fbx_takes": ""}}
        self.assertEqual(KeepingStore.migrate(), [old, "fbx_takes"])
        self.assertEqual(KeepingStore.data[Scope.DELIVERABLE], {})
        self.assertIsNone(SR.LIGHTMAPS.load(KeepingStore), "nothing to move")

    def test_a_value_already_under_the_key_wins(self):
        SR.LIGHTMAPS.save(DictStore, {"objects": ["current"]})
        self._stored(Scope.DELIVERABLE, "lightmap_metadata", {"objects": ["old"]})
        self.assertEqual(DictStore.migrate(), ["lightmap_metadata"])
        self.assertEqual(SR.LIGHTMAPS.load(DictStore)["objects"], ["current"])
        self.assertIsNone(DictStore.read(Scope.DELIVERABLE, "lightmap_metadata"))

    def test_an_export_commit_migrates_first(self):
        """The writes and the handoff see current keys alone: a scene written
        before the rename ships no record twice, nor a handoff naming a key no
        reader reads."""
        self._stored(
            Scope.DELIVERABLE, "visibility_tracks", {"version": 1, "tracks": [1]}
        )
        ExportSnapshot.assemble({SR.SHOTS: _shots}).commit(DictStore)
        self.assertIsNone(DictStore.read(Scope.DELIVERABLE, "visibility_tracks"))
        self.assertEqual(SR.VISIBILITY.load(DictStore)["tracks"], [1])
        self.assertEqual(
            _records(SR.HANDOFF.load(DictStore)),
            ["data_export.animation", "data_export.render_effects"],
        )

    def test_a_failed_migration_costs_the_export_nothing(self):
        class Unmigratable(DictStore):
            @classmethod
            def migrate(cls):
                raise RuntimeError("locked")

        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ):
            ExportSnapshot.assemble({SR.SHOTS: _shots}).commit(Unmigratable)
        self.assertTrue(SR.SHOTS.is_present(DictStore))


class _LeftoverStore(DictStore):
    """A scene holding carriers an import left beside its own: plain
    ``{key: value}`` dicts by name, each removal recorded."""

    leftovers: Dict[str, Any] = {}
    deleted: list = []

    @classmethod
    def reset(cls):
        cls.data = {}
        cls.leftovers = {}
        cls.deleted = []

    @classmethod
    def _leftover_carriers(cls):
        return dict(cls.leftovers)

    @classmethod
    def _carrier_values(cls, carrier):
        return dict(carrier)

    @classmethod
    def _delete_carrier(cls, carrier):
        cls.deleted.append(carrier)


class TestLeftoverCarriers(unittest.TestCase):
    """``SceneStoreBase.migrate`` settles a carrier an import left beside the
    scene's own -- a plain import never merges it, so nothing reads or
    refreshes it again, and only the written file still carries it.

    REGRESSION (2026-10-28): a production assembly imported a module whose
    ``data_export`` landed as ``data_export1`` holding a pre-rename
    ``lightmap_metadata`` and a version-1 handoff; every export since shipped
    both, and the deliverable verifier failed the file on them. Added:
    2026-10-28"""

    def setUp(self):
        _LeftoverStore.reset()

    @staticmethod
    def _carrier(**records):
        return {key: json.dumps(payload) for key, payload in records.items()}

    def test_one_holding_only_records_the_scene_derives_is_removed(self):
        stale = self._carrier(
            lightmap_metadata={"version": 1, "objects": [{"name": "A"}]},
            handoff={"version": 1, "reads": {}},
        )
        _LeftoverStore.leftovers = {
            "data_export1": (Scope.DELIVERABLE, stale)
        }
        SR.LIGHTMAPS.save(_LeftoverStore, {"objects": ["current"]})
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="INFO"
        ) as logs:
            _LeftoverStore.migrate()
        self.assertEqual(_LeftoverStore.deleted, [stale])
        self.assertTrue(any("data_export1" in line for line in logs.output), logs)
        self.assertEqual(
            SR.LIGHTMAPS.load(_LeftoverStore)["objects"],
            ["current"],
            "the scene's own records are untouched",
        )

    def test_one_holding_a_current_record_the_scenes_own_lacks_stays(self):
        """A reader takes a module carrier's record where the scene's own has
        none: removed, a raw-imported module's lightmaps would leave the
        deliverable unlit, though a merge re-derives them. The bridge test
        that shipped such a carrier failed on the first rule (2026-10-28)."""
        current = self._carrier(lightmaps={"version": SR.LIGHTMAPS.version, "objects": []})
        _LeftoverStore.leftovers = {"MOD:data_export": (Scope.DELIVERABLE, current)}
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ) as logs:
            _LeftoverStore.migrate()
        self.assertEqual(_LeftoverStore.deleted, [])
        self.assertIn(SR.LIGHTMAPS.owner, "".join(logs.output))

    def test_a_current_record_the_scenes_own_shadows_is_none_read(self):
        current = self._carrier(lightmaps={"version": SR.LIGHTMAPS.version, "objects": []})
        _LeftoverStore.leftovers = {"data_export1": (Scope.DELIVERABLE, current)}
        SR.LIGHTMAPS.save(_LeftoverStore, {"objects": ["own"]})
        _LeftoverStore.migrate()
        self.assertEqual(_LeftoverStore.deleted, [current])

    def test_one_holding_what_a_merge_keeps_stays_and_is_named(self):
        """Shots, a file map, a channel no record declares: what a merge
        would bring in is the user's call (``merge_carriers`` /
        ``discard_carriers``), never removed on its own."""
        kept = self._carrier(audio_file_map={"2": "b.wav"})
        _LeftoverStore.leftovers = {"data_internal1": (Scope.PRIVATE, kept)}
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ) as logs:
            _LeftoverStore.migrate()
        self.assertEqual(_LeftoverStore.deleted, [])
        (line,) = [line for line in logs.output if "data_internal1" in line]
        self.assertIn("merge", line)

    def test_one_holding_a_value_beside_its_records_stays(self):
        """A keyed weight drives the deliverable's animation: no record
        restates it, so the carrier is not the scene's to drop."""
        weighted = {**self._carrier(handoff={"version": 1}), "emissive_glow": 0.5}
        _LeftoverStore.leftovers = {"data_export1": (Scope.DELIVERABLE, weighted)}
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ):
            _LeftoverStore.migrate()
        self.assertEqual(_LeftoverStore.deleted, [])

    def test_a_removal_the_host_refuses_costs_the_migration_nothing(self):
        class Locked(_LeftoverStore):
            @classmethod
            def _delete_carrier(cls, carrier):
                raise RuntimeError("locked")

        Locked.leftovers = {
            "data_export1": (Scope.DELIVERABLE, self._carrier(handoff={"version": 1}))
        }
        Locked.data = {Scope.DELIVERABLE: {"fbx_takes": "[]"}}
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ):
            self.assertEqual(Locked.migrate(), ["fbx_takes"])

    def test_a_host_that_cannot_list_its_copies_still_moves_the_keys(self):
        class Unlistable(_LeftoverStore):
            @classmethod
            def _leftover_carriers(cls):
                raise RuntimeError("no scene")

        Unlistable.data = {Scope.DELIVERABLE: {"fbx_takes": "[]"}}
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ):
            self.assertEqual(Unlistable.migrate(), ["fbx_takes"])

    def test_a_store_with_no_copies_has_nothing_to_settle(self):
        self.assertEqual(DictStore._leftover_carriers(), {})
        self.assertEqual(DictStore.migrate(), [])


# -- the handoff --------------------------------------------------------------------


def _records(block):
    """The ``data_export`` records an FBX handoff block names, by address."""
    return sorted(
        {
            row["data"][0]
            for row in block["assembly"]
            if row["data"] and row["data"][0].startswith("data_export.")
        }
    )


class TestHandoff(SceneRecordsCase):
    """The handoff record. What its block says is ``HandoffRecord``'s
    (``test_handoff_record``); what each record contributes to it is declared
    here."""

    def test_every_shipping_record_declares_its_assembly(self):
        """A record added later rides into the block's ``assembly`` from its
        own declaration -- so it must have one, or a reader of the
        deliverable learns only that it exists."""
        for spec in SR.deliverable():
            if spec is SR.HANDOFF:
                continue
            with self.subTest(record=spec.key):
                self.assertTrue(spec.assembly.strip())

    def test_the_render_effects_are_described_channel_by_channel(self):
        """The render effects record carried a highlight ramp and its colour stops
        that the record's description never mentioned, so a reader of the
        deliverable had no way to know the scene pulses: every effect the
        glTF channel table knows must be named, stops included."""
        from pythontk.file_utils.mesh_convert.glb.fades import CHANNELS

        for name, channel in CHANNELS.items():
            keys = [name, *(channel.color_stops.keys if channel.color_stops else ())]
            for key in keys:
                with self.subTest(key=key):
                    self.assertIn(f"'{key}'", SR.VISIBILITY.description)
                    self.assertIn(f"'{key}'", SR.VISIBILITY.assembly)

    def test_the_lightmap_record_describes_the_probes_highlight_copies(self):
        """A finished FBX's record lists the probe's highlight copies
        (``MeshConvert.embed_fbx_highlight_probes``), while the text told a
        reader only to light the record's 'lights' itself: following it beside
        the copies, the highlights count twice. Every field an entry carries
        is named, and so is the rule -- for a record that holds them. Added:
        2026-10-19. Version 3 (2026-11-07): each copy is COMPLETE, used in the
        probe's place at its own size, and the probe's own map holds the
        lights as its capture point sees them, so a record with a probe has
        none of its 'lights' lit, copies or not -- said once, by the 'lights'
        clause, which every record holding lights carries (2026-11-07)."""
        text = SR.LIGHTMAPS.assembly_for({"probe": {"highlights": [{"map": "h"}]}})
        probe_alone = SR.LIGHTMAPS.assembly_for({"probe": {"map": "p.hdr"}})
        self.assertNotIn("'probe.highlights'", probe_alone)
        for key in ("probe.highlights", "map", "objects"):
            with self.subTest(key=key):
                self.assertIn(f"'{key}'", text)
        self.assertIn("COMPLETE", text)
        self.assertIn("in the probe's place", text)
        self.assertNotIn("box-filtered", text, "nothing is added to the probe")
        for record in (text, probe_alone):
            self.assertIn("as the capture point sees them", record)
        lit = SR.LIGHTMAPS.assembly_for(
            {"probe": {"map": "p.hdr"}, "lights": [{"type": "point"}]}
        )
        self.assertIn("while the record has one, a reader lights none of them", lit)
        self.assertEqual(lit.count("lights none of"), 1, "said once")
        lights_alone = SR.LIGHTMAPS.assembly_for({"lights": [{"type": "point"}]})
        self.assertIn("a delivery with no probe", lights_alone)

    def test_the_retired_handoff_names_still_serve_the_handoffs(self):
        """``rendering_policy``, ``HANDOFF_INSTRUCTIONS``, ``HANDOFF_ASSEMBLY``
        and ``handoff_block`` (retired 2026-10-20 for ``HandoffRecord``,
        removed in 0.18.0) warn, and answer with what ``HandoffRecord``
        states now."""
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(SR.rendering_policy(), HandoffRecord.rendering_policy())
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(SR.HANDOFF_INSTRUCTIONS, HandoffRecord.instructions("fbx"))
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(SR.HANDOFF_ASSEMBLY, HandoffRecord.CARRIERS["fbx"].own)
        channels = {SR.LIGHTMAPS.key: "{}", "emissiveGroup_Neon": 0.5}
        with self.assertWarns(DeprecationWarning):
            block = SR.handoff_block(channels, source={"application": "maya"})
        self.assertEqual(
            block, HandoffRecord.block("fbx", channels, {"application": "maya"})
        )
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(
                SR.handoff_block({"emissiveGroup_Neon": 1.0}),
                {},
                "a carrier holding no record has nothing to describe",
            )


# -- the context + snapshot ---------------------------------------------------------


def _shots(ctx: ExportContext):
    return SR.SHOTS.make({"shots": [{"clip": "A", "description": "The door."}]})


def _visibility(ctx: ExportContext):
    """A reader of the shots record: one track per shot clip it finds -- what
    the assembly hands it, else the store's."""
    shots = (ctx.record(SR.SHOTS, DictStore) or {}).get("shots") or []
    return SR.VISIBILITY.make(
        {"fps": 24.0, "tracks": [{"node": s["clip"]} for s in shots]}
    )


def _older_scene_takes():
    """The take list a scene published before 0.11.0 still holds -- a record
    retired whole (:attr:`SceneRecords.RETIRED`)."""
    DictStore.write(
        Scope.DELIVERABLE,
        "fbx_takes",
        json.dumps([{"name": "A", "start": 1, "end": 10}]),
    )


def _nothing(ctx):
    return None


def _boom(ctx):
    raise RuntimeError("subsystem down")


PRODUCERS = {SR.VISIBILITY: _visibility, SR.SHOTS: _shots, SR.LIGHTMAPS: _nothing}


def _scoped_lightmaps(ctx):
    """A lightmap producer over two marked nodes, narrowed to ``ctx.scope``
    the way a DCC's is: a root and its descendants."""
    nodes = ["|ROOM|wall", "|TEMP|old"]
    if ctx.scope is not None:
        nodes = [
            n for n in nodes if any(n == r or n.startswith(r + "|") for r in ctx.scope)
        ]
    if not nodes:
        return None
    return SR.LIGHTMAPS.make(
        {"objects": [{"name": n.rsplit("|", 1)[-1], "map": "room.exr"} for n in nodes]}
    )


def _materials(ctx):
    """A materials producer: one material, as the scene stages it."""
    return SR.MATERIALS.make(
        {
            "materials": [
                {
                    "name": "m",
                    "baseColorFactor": [1, 1, 1, 1],
                    "metallicFactor": 0,
                    "roughnessFactor": 1,
                    "emissiveFactor": [0, 0, 0],
                    "alphaMode": "OPAQUE",
                }
            ]
        }
    )


class TestTransientRecords(SceneRecordsCase):
    """A transient record describes one write: produced for it, carried by
    the file, never held by the scene. The materials record names the maps
    an export staged for its write (a texture template's converted copies,
    which the export's restore deletes), so a copy the scene kept would ship
    stale from the next File > Export. Added: 2026-10-20"""

    def test_the_materials_record_is_transient_and_produced_like_any_other(self):
        self.assertTrue(SR.MATERIALS.transient)
        self.assertIs(SR.MATERIALS.kind, Kind.DERIVED)
        self.assertEqual(SR.transients(), [SR.MATERIALS])
        self.assertEqual(SR.check_producers({SR.MATERIALS: _materials}), [SR.MATERIALS])

    def test_a_transient_record_is_declared_derived(self):
        """Produced for every write, hand-offs included: authored data that
        only some writes refresh could not be cleared after each."""
        with self.assertRaises(ValueError):
            RecordSpec(
                "x", Scope.DELIVERABLE, 1, owner="T", description="d", transient=True
            )

    def test_the_write_carries_it_and_the_restore_clears_it(self):
        snap = ExportSnapshot.assemble(
            {SR.MATERIALS: _materials, SR.SHOTS: _shots}, ExportContext()
        )
        self.assertEqual(snap.scene_wide, {"materials": None})
        snap.commit(DictStore)
        self.assertIsNotNone(SR.MATERIALS.load(DictStore), "the write carries it")
        self.assertIn("data_export.materials", _records(SR.HANDOFF.load(DictStore)))
        snap.restore_scene_wide(DictStore)
        self.assertIsNone(SR.MATERIALS.load(DictStore), "the scene never holds it")
        self.assertNotIn("data_export.materials", _records(SR.HANDOFF.load(DictStore)))
        self.assertIsNotNone(SR.SHOTS.load(DictStore), "the scene's own records stay")

    def test_a_carrier_the_write_alone_needed_is_discarded(self):
        """Every export states its materials, so the carrier is made for the
        write; a restore that leaves it holding nothing asks the store to
        remove it -- the scene left as the export found it -- and one that
        leaves the scene's own records asks nothing."""
        calls = []

        def discard(cls, scope):
            calls.append(scope)
            return True

        with mock.patch.object(DictStore, "discard_if_empty", classmethod(discard)):
            snap = ExportSnapshot.assemble({SR.MATERIALS: _materials}, ExportContext())
            snap.commit(DictStore)
            snap.restore_scene_wide(DictStore)
            self.assertEqual(calls, [Scope.DELIVERABLE])
            calls.clear()
            snap = ExportSnapshot.assemble(
                {SR.MATERIALS: _materials, SR.SHOTS: _shots}, ExportContext()
            )
            snap.commit(DictStore)
            snap.restore_scene_wide(DictStore)
            self.assertEqual(calls, [], "the shots stay, and the carrier with them")

    def test_a_hand_off_produces_it(self):
        snap = ExportSnapshot.assemble(
            {SR.MATERIALS: _materials}, ExportContext(mode=ExportContext.HANDOFF)
        )
        self.assertIsNotNone(snap.record(SR.MATERIALS))
        self.assertEqual(snap.scene_wide, {"materials": None})

    def test_an_authoring_publish_clears_one_it_finds(self):
        """No write: an import brought it, or a crash left it -- a merge's
        re-derivation publishes at authoring time, and clears it rather than
        producing it."""
        SR.MATERIALS.save(DictStore, {"materials": []})
        calls = []
        snap = ExportSnapshot.assemble(
            {SR.MATERIALS: lambda ctx: calls.append(ctx) or _materials(ctx)},
            ExportContext(mode=ExportContext.AUTHORING),
        )
        self.assertEqual(calls, [], "never produced")
        snap.commit(DictStore)
        self.assertIsNone(SR.MATERIALS.load(DictStore))


class TestSnapshot(SceneRecordsCase):
    def test_a_scoped_assembly_ships_the_export_set_and_keeps_the_scene_whole(self):
        """An export-scoped record names only what the write ships, and the
        scene's own copy is kept for the exporter to put back after it.
        Measured 2026-10-09: a production FBX's lightmap record named five
        objects of a hidden group the export left out, and the post-write
        embed shipped their stale maps (two it could not find failed the
        verifier). Added: 2026-10-09"""
        self.assertTrue(SR.LIGHTMAPS.export_scoped)
        self.assertTrue(SR.SHADOWS.export_scoped)
        # A shot names no member since 2026-10-20: every take ships whatever
        # the export set, so there is nothing to narrow.
        self.assertFalse(SR.SHOTS.export_scoped)
        producers = {SR.LIGHTMAPS: _scoped_lightmaps, SR.SHOTS: _shots}
        snap = ExportSnapshot.assemble(producers, ExportContext(scope=("|ROOM",)))
        self.assertEqual(
            [o["name"] for o in snap.record(SR.LIGHTMAPS)["objects"]], ["wall"]
        )
        self.assertEqual(
            list(snap.scene_wide), [SR.LIGHTMAPS.key], "only what the scope narrowed"
        )
        snap.commit(DictStore)
        self.assertEqual(
            len(SR.LIGHTMAPS.load(DictStore)["objects"]),
            1,
            "the write ships the scoped copy",
        )
        snap.restore_scene_wide(DictStore)
        self.assertEqual(
            len(SR.LIGHTMAPS.load(DictStore)["objects"]), 2, "the scene keeps it whole"
        )

    def test_an_assembly_the_scope_does_not_narrow_has_nothing_to_restore(self):
        producers = {SR.LIGHTMAPS: _scoped_lightmaps}
        for scope in (None, ("|ROOM", "|TEMP")):
            snap = ExportSnapshot.assemble(producers, ExportContext(scope=scope))
            self.assertEqual(snap.scene_wide, {}, scope)
            self.assertIsNone(snap.restore_scene_wide(DictStore), scope)

    def test_the_scene_wide_pass_says_nothing(self):
        """What a producer logs in the scene-wide pass is said already (a node
        the write ships) or about one it does not ship: the export log hears
        the scoped pass alone, and logging is handed back as it was.
        Added: 2026-10-09"""

        def loud(ctx):
            logging.getLogger("pythontk.test.loud").warning("scope=%s", ctx.scope)
            return _scoped_lightmaps(ctx)

        with self.assertLogs("pythontk.test.loud", "WARNING") as logs:
            snap = ExportSnapshot.assemble(
                {SR.LIGHTMAPS: loud}, ExportContext(scope=("|ROOM",))
            )
        self.assertEqual(logs.output, ["WARNING:pythontk.test.loud:scope=('|ROOM',)"])
        self.assertIn(SR.LIGHTMAPS.key, snap.scene_wide)
        self.assertEqual(logging.root.manager.disable, logging.NOTSET)

    def test_a_scope_that_ships_none_of_a_records_nodes_restores_the_whole(self):
        snap = ExportSnapshot.assemble(
            {SR.LIGHTMAPS: _scoped_lightmaps}, ExportContext(scope=("|PROPS",))
        )
        self.assertIsNone(snap.record(SR.LIGHTMAPS), "nothing lightmapped ships")
        snap.commit(DictStore)
        self.assertIsNone(SR.LIGHTMAPS.load(DictStore))
        snap.restore_scene_wide(DictStore)
        self.assertEqual(len(SR.LIGHTMAPS.load(DictStore)["objects"]), 2)

    def test_assemble_orders_by_dependency_and_hands_prior_records_over(self):
        ctx = ExportContext()
        snap = ExportSnapshot.assemble(PRODUCERS, ctx)
        # Dependency order, declaration order breaking ties: the two records
        # with no dependencies first (shots, then lightmaps), then the reader
        # of shots.
        self.assertEqual(
            list(snap.produced),
            [SR.SHOTS.key, SR.LIGHTMAPS.key, SR.VISIBILITY.key],
        )
        vis = snap.record(SR.VISIBILITY)
        self.assertEqual(
            vis["tracks"],
            [{"node": "A"}],
            "read from the record produced in this assembly",
        )
        self.assertEqual(
            snap.record(SR.SHOTS),
            {"version": 3, "shots": [{"clip": "A", "description": "The door."}]},
        )
        self.assertIsNone(snap.record(SR.LIGHTMAPS))

    def test_commit_writes_once_clears_the_empty_and_stamps_the_handoff(self):
        SR.LIGHTMAPS.save(DictStore, {"objects": [1]})  # stale, about to be cleared
        snap = ExportSnapshot.assemble(
            PRODUCERS, ExportContext(source={"application": "test"})
        )
        written = snap.commit(DictStore)
        self.assertIsNone(
            SR.LIGHTMAPS.load(DictStore), "a producer with nothing to say clears"
        )
        self.assertEqual(SR.SHOTS.load(DictStore)["shots"][0]["clip"], "A")
        handoff = SR.HANDOFF.load(DictStore)
        self.assertEqual(
            _records(handoff),
            ["data_export.animation", "data_export.render_effects"],
        )
        self.assertEqual(handoff["extras"]["source"], {"application": "test"})
        self.assertIsNone(written[SR.LIGHTMAPS.key])
        self.assertIn("handoff", written)

    def test_the_handoff_publishes_the_exports_lighting_choices(self):
        """What the export decided about the lighting recipe (a baked-reflection
        level) is an input: the stamped handoff carries it,
        merged the way the GLB's envelope merges it -- as the file's records
        use the recipe (a lightmapped material's level, where it carries
        lightmaps). Added: 2026-09-21"""
        rendering = {"lightmappedMaterials": {"envMapIntensity": 0.5}}
        producers = {**PRODUCERS, SR.LIGHTMAPS: _scoped_lightmaps}
        ExportSnapshot.assemble(producers, ExportContext(rendering=rendering)).commit(
            DictStore
        )
        published = SR.HANDOFF.load(DictStore)["rendering"]
        self.assertEqual(
            published,
            HandoffRecord.rendering_policy(
                rendering, DictStore.channels(Scope.DELIVERABLE)
            ),
        )
        self.assertEqual(published["lightmappedMaterials"]["envMapIntensity"], 0.5)

    def test_the_same_context_produces_the_same_records_twice(self):
        """Re-running an assembly is harmless by construction -- the defect the
        old patch-after-producers pipeline had (three exports logging the right
        clip origin and shipping the old one)."""
        ctx = ExportContext(source={"application": "test"})
        first = ExportSnapshot.assemble(PRODUCERS, ctx).commit(DictStore)
        second = ExportSnapshot.assemble(PRODUCERS, ctx).commit(DictStore)
        self.assertEqual(first, second)

    def test_a_reused_context_starts_each_assembly_empty(self):
        """A context carries ONE assembly's records: once the shots are
        cleared, a later assembly on the same context must read the store,
        not the shot record an earlier assembly left on the context."""
        ctx = ExportContext()
        ExportSnapshot.assemble(PRODUCERS, ctx).commit(DictStore)
        ExportSnapshot.assemble({SR.SHOTS: _nothing}).commit(DictStore)
        snap = ExportSnapshot.assemble({SR.VISIBILITY: _visibility}, ctx)
        self.assertEqual(snap.record(SR.VISIBILITY)["tracks"], [])

    def test_a_producers_notes_are_logged_and_kept_for_one_assembly(self):
        """What a producer leaves out of a deliverable is said, never silent:
        logged on every route, and kept on the context for an exporter's own
        report -- one assembly's worth, like its records."""

        def noting(ctx):
            ctx.note("2 shot(s) left out")
            return None

        ctx = ExportContext()
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", "WARNING"
        ) as log:
            snap = ExportSnapshot.assemble({SR.SHOTS: noting}, ctx)
        self.assertIn("2 shot(s) left out", log.output[0])
        self.assertEqual(snap.ctx.notes, ["2 shot(s) left out"])
        # By the record that noted it, so a host can offer its remedy.
        self.assertEqual(snap.noted, {SR.SHOTS.key: ["2 shot(s) left out"]})
        ExportSnapshot.assemble({SR.SHOTS: _nothing}, ctx)
        self.assertEqual(ctx.notes, [], "a reused context starts without notes")

    def test_a_producer_returning_a_bare_payload_is_isolated(self):
        """A producer that forgets ``spec.make`` fails ITS record, not the
        assembly -- the isolation contract covers a wrong return too."""
        table = {SR.LIGHTMAPS: lambda c: {"objects": [1]}, SR.SHOTS: _shots}
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ):
            snap = ExportSnapshot.assemble(table)
        self.assertEqual(snap.failed, {SR.LIGHTMAPS.key})
        self.assertIn(SR.SHOTS.key, snap.records)

    def test_a_private_record_is_refused_rather_than_clearing_its_deliverable_twin(
        self,
    ):
        """A snapshot commits the deliverable carrier, so a private record
        handed to it is refused rather than written there (or, cleared, a
        deliverable of the same key cleared)."""
        SR.VISIBILITY.save(DictStore, {"tracks": [1]})
        with self.assertRaises(ValueError):
            ExportSnapshot.publish(DictStore, {SR.EMISSIVE_REGISTRY: None})
        with self.assertRaises(ValueError):
            ExportSnapshot.assemble({SR.EMISSIVE_REGISTRY: _nothing})
        self.assertEqual(SR.VISIBILITY.load(DictStore)["tracks"], [1])

    def test_a_refused_write_is_isolated(self):
        """One record the store refuses (a locked attribute) is left as stored
        and reported; every other record, and the handoff, is still written."""

        class LockedStore(DictStore):
            @classmethod
            def write(cls, scope, key, text):
                if key == SR.LIGHTMAPS.key:
                    raise RuntimeError("locked")
                return super().write(scope, key, text)

        table = {
            SR.LIGHTMAPS: lambda c: SR.LIGHTMAPS.make({"objects": [1]}),
            SR.SHOTS: _shots,
        }
        snap = ExportSnapshot.assemble(table)
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ):
            written = snap.commit(LockedStore)
        self.assertIn(SR.LIGHTMAPS.key, snap.failed)
        self.assertNotIn(SR.LIGHTMAPS.key, written)
        self.assertTrue(SR.SHOTS.is_present(DictStore))
        self.assertTrue(SR.HANDOFF.is_present(DictStore))
        self.assertIn(f"{SR.LIGHTMAPS.key} (FAILED", snap.summary())
        self.assertNotIn(f"{SR.LIGHTMAPS.key} (1", snap.summary())

    def test_handoff_is_never_manufactured_onto_an_empty_carrier(self):
        snap = ExportSnapshot.assemble({SR.LIGHTMAPS: _nothing})
        snap.commit(DictStore)
        self.assertEqual(DictStore.data, {})

    def test_handoff_is_cleared_when_it_would_describe_nothing(self):
        SR.LIGHTMAPS.save(DictStore, {"objects": [1]})
        ExportSnapshot.assemble(
            {SR.LIGHTMAPS: lambda c: SR.LIGHTMAPS.make({"objects": [1]})}
        ).commit(DictStore)
        self.assertTrue(SR.HANDOFF.is_present(DictStore))
        ExportSnapshot.assemble({SR.LIGHTMAPS: _nothing}).commit(DictStore)
        self.assertFalse(
            SR.HANDOFF.is_present(DictStore),
            "a lone self-referential channel must not survive",
        )

    def test_a_failing_producer_is_isolated_and_leaves_its_record_as_stored(self):
        SR.LIGHTMAPS.save(DictStore, {"objects": [1]})
        table = {SR.LIGHTMAPS: _boom, SR.SHOTS: _shots}
        with self.assertLogs(
            "pythontk.core_utils.engines.scene_export", level="WARNING"
        ):
            snap = ExportSnapshot.assemble(table)
        self.assertEqual(snap.failed, {SR.LIGHTMAPS.key})
        snap.commit(DictStore)
        self.assertEqual(
            SR.LIGHTMAPS.load(DictStore)["objects"], [1], "left as stored, not cleared"
        )
        self.assertIn(SR.SHOTS.key, snap.records)
        self.assertIn("FAILED", snap.summary())

    def test_a_handoff_context_refreshes_only_derived_records(self):
        ran = []
        table = {
            SR.LIGHTMAPS: lambda c: ran.append("lightmaps"),
            SR.VISIBILITY: lambda c: ran.append("visibility"),
        }
        ExportSnapshot.assemble(table, ExportContext(mode=ExportContext.HANDOFF))
        self.assertEqual(ran, ["visibility"])
        ExportSnapshot.assemble(
            table, ExportContext(mode=ExportContext.HANDOFF), only=[SR.LIGHTMAPS]
        )
        self.assertEqual(
            ran,
            ["visibility"],
            "naming an authored record does not make a bridge its authority",
        )

    def test_a_handoff_ships_the_audio_its_scene_plays_now(self):
        """The audio events are read off the keyed tracks / sound strips an
        artist moves between one push and the next, like the visibility
        curves: a hand-off refreshes them. Authored, a Unity-bridge FBX
        shipped the last Scene Exporter run's events (measured: a track
        added since was missing) or none at all."""
        ran = []
        table = {SR.AUDIO: lambda c: ran.append("audio")}
        ExportSnapshot.assemble(table, ExportContext(mode=ExportContext.HANDOFF))
        self.assertEqual(ran, ["audio"])

    def test_a_handoff_ships_the_shots_its_scene_holds_now(self):
        """The shot store is the one authority on the shots record, and the
        record depends on the live scene (which shots are stale). Authored, a
        WebXR or Unity push shipped the shots as the last Scene Exporter run
        left them, or none at all: measured 2026-10-21, a fresh scene's Maya
        push and Blender send carried no shot record -- and the audio record,
        derived, scoped its events against that stale copy."""
        ran = []
        table = {
            SR.SHOTS: lambda c: ran.append("animation"),
            SR.AUDIO: lambda c: ran.append("audio"),
        }
        ExportSnapshot.assemble(table, ExportContext(mode=ExportContext.HANDOFF))
        self.assertEqual(ran, ["animation", "audio"], "shots first: audio reads them")

    def test_a_retired_record_never_rides_into_the_next_export(self):
        """``fbx_takes`` is no record (:attr:`SceneRecords.RETIRED`); a scene
        published before 0.11.0 still holds one, and the next commit removes
        it whatever the shots producer does -- returns the shots, nothing, or
        fails -- so the old takes never ride into the next export."""
        logger = "pythontk.core_utils.engines.scene_export"
        for producer in (_shots, _nothing, _boom):
            with self.subTest(producer=producer.__name__):
                _older_scene_takes()
                with self.assertLogs(logger, level="INFO"):
                    ExportSnapshot.assemble({SR.SHOTS: producer}).commit(DictStore)
                self.assertIsNone(DictStore.read(Scope.DELIVERABLE, "fbx_takes"))

    def test_a_reader_sees_a_record_cleared_in_this_assembly_as_absent(self):
        """The shots producer returns nothing (an emptied store): a reader later
        in the SAME assembly must not fall back to the stored shot record the
        commit is about to clear."""
        ExportSnapshot.assemble({SR.SHOTS: _shots}).commit(DictStore)
        seen = {}

        def reader(ctx):
            seen["shots"] = ctx.record(SR.SHOTS, DictStore)

        ExportSnapshot.assemble({SR.SHOTS: _nothing, SR.VISIBILITY: reader})
        self.assertEqual(seen, {"shots": None})

    def test_the_envelope_version_leads_and_a_producer_value_wins(self):
        self.assertEqual(
            list(SR.LIGHTMAPS.make({"objects": []}).payload), ["version", "objects"]
        )
        self.assertEqual(SR.SHADOWS.make({"version": 1}).payload["version"], 1)

    def test_only_narrows_a_pipeline_run(self):
        snap = ExportSnapshot.assemble(PRODUCERS, only=[SR.SHOTS.key])
        self.assertEqual(list(snap.produced), [SR.SHOTS.key])

    def test_publish_is_the_one_call_authoring_time_commit(self):
        """A tool republishing its own record hands payloads (or records, or a
        falsy value to clear) and gets the same commit an export makes -- the
        handoff restamped included."""
        snap = ExportSnapshot.publish(
            DictStore,
            {
                SR.LIGHTMAPS: {"objects": [1]},
                SR.SHADOWS: SR.SHADOWS.make({"planes": []}),
            },
        )
        self.assertEqual(snap.ctx.mode, ExportContext.AUTHORING)
        self.assertEqual(SR.LIGHTMAPS.load(DictStore)["objects"], [1])
        self.assertEqual(SR.SHADOWS.load(DictStore)["version"], SR.SHADOWS.version)
        self.assertEqual(
            _records(SR.HANDOFF.load(DictStore)),
            ["data_export.lightmaps", "data_export.shadows"],
        )
        ExportSnapshot.publish(DictStore, {SR.LIGHTMAPS: None})
        self.assertIsNone(SR.LIGHTMAPS.load(DictStore))
        self.assertEqual(
            _records(SR.HANDOFF.load(DictStore)),
            ["data_export.shadows"],
        )

    def test_channels_and_summary(self):
        snap = ExportSnapshot.assemble(PRODUCERS)
        self.assertEqual(sorted(snap.channels()), [SR.SHOTS.key, SR.VISIBILITY.key])
        self.assertIn(f"{SR.SHOTS.key} (1 entry)", snap.summary())
        self.assertNotIn(SR.LIGHTMAPS.key, snap.summary(), "nothing shipped")

    def test_context_record_falls_back_to_the_store_at_authoring_time(self):
        SR.SHOTS.save(DictStore, {"shots": [{"clip": "A"}]})
        ctx = ExportContext(mode=ExportContext.AUTHORING)
        self.assertEqual(ctx.record(SR.SHOTS, DictStore)["shots"], [{"clip": "A"}])
        self.assertIsNone(ctx.record(SR.SHOTS))


class TestRecordTransfer(SceneRecordsCase):
    """Another scene's carrier meeting this scene's (a referenced module whose
    reference is imported): planned by declaration, applied per rule."""

    @staticmethod
    def _other(private=None, deliverable=None):
        return {
            Scope.PRIVATE: {k: json.dumps(v) for k, v in (private or {}).items()},
            Scope.DELIVERABLE: {
                k: json.dumps(v) for k, v in (deliverable or {}).items()
            },
        }

    def _merge(self, other, ctx=None):
        merge = RecordTransfer.between(DictStore, other)
        return merge, merge.apply(DictStore, ctx)

    # ------------------------------------------------------------- the plan
    def test_a_carrier_of_derived_records_and_its_own_baseline_is_empty(self):
        """Nothing to decide: deliverables are re-derived, a baseline describes
        its own scene -- so no prompt, and the carrier can simply go."""
        other = self._other(
            private={"hierarchy_baseline": {"format": 1, "paths": ["a"]}},
            deliverable={
                SR.LIGHTMAPS.key: {"version": 2, "objects": [{"name": "a"}]},
                "handoff": {"version": 1},
            },
        )
        merge = RecordTransfer.between(DictStore, other)
        self.assertTrue(merge.is_empty)
        self.assertEqual(merge.summary(), [])
        self.assertEqual(merge.rederive, [SR.LIGHTMAPS, SR.HANDOFF])

    def test_an_empty_carrier_is_empty(self):
        self.assertTrue(RecordTransfer.between(DictStore, {}).is_empty)
        self.assertTrue(
            RecordTransfer.between(
                DictStore, {Scope.PRIVATE: {"shot_store": ""}}
            ).is_empty
        )

    def test_a_record_holding_no_entries_is_nothing_to_ask_about(self):
        """A module's key stash with no clip, its SmartBake record with no
        session: stored, yet a merge would keep nothing of them -- so they
        neither raise the question nor appear in it.  Settings beside an
        empty list are no entry; a list that holds one is."""
        stash = {
            "schema": 1,
            "scene_fps": 30.0,
            "clips": [],
            "preview": None,
            "next_id": 1,
        }
        empty = self._other(private={"key_stash": stash, "smart_bake_sessions": []})
        self.assertTrue(RecordTransfer.between(DictStore, empty).is_empty)
        merge = RecordTransfer.between(
            DictStore,
            self._other(
                private={
                    "key_stash": stash,
                    "smart_bake_sessions": [],
                    "audio_file_map": {"1": "a.wav"},
                }
            ),
        )
        self.assertFalse(merge.is_empty)
        self.assertEqual(merge.summary(), ["Audio Clips: 1 entry"])
        markers_only = self._other(
            private={
                "shot_store": {
                    "scene_fps": 30.0,
                    "shots": [],
                    "markers": [{"time": 1.0}],
                }
            }
        )
        markers = RecordTransfer.between(DictStore, markers_only)
        self.assertFalse(markers.is_empty)
        # Its first list is empty, so no count -- "0 entries" would say
        # it brings nothing.
        self.assertEqual(markers.summary(), ["Shots"])

    def test_the_summary_names_what_arrives_and_how_much(self):
        other = self._other(
            private={
                "audio_file_map": {"1": "a.wav", "2": "b.wav"},
                SR.EMISSIVE_REGISTRY.key: {
                    "schema": 1,
                    "groups": {"glow": {"slot": 0}},
                },
            }
        )
        merge = RecordTransfer.between(DictStore, other)
        self.assertFalse(merge.is_empty)
        self.assertEqual(
            merge.summary(), ["Emissive Groups: 1 entry", "Audio Clips: 2 entries"]
        )

    def test_every_private_record_declares_how_it_merges(self):
        """A private record has no producer, so it cannot be re-derived: each
        must say how another scene's copy combines with this one's, and a
        codec must have a merge to call."""
        for spec in SR.private():
            self.assertIsNot(spec.merge, Merge.DERIVE, spec.key)
            if spec.merge is Merge.CODEC:
                self.assertIsNotNone(SR.codec(spec), spec.key)
        for spec in SR.deliverable():
            self.assertIs(spec.merge, Merge.DERIVE, spec.key)

    def test_another_scenes_renamed_keys_are_read_at_their_current_keys(self):
        """A scene saved before a rename holds a record under the key it was
        renamed from (:attr:`RecordSpec.renamed_from`). Its records are read
        as ``SceneStoreBase.migrate`` reads a scene's own -- at the current
        key, a value already there winning -- so a module's emissive registry
        merges by its codec instead of crossing as a channel nothing declares
        (adopted under the old key, then dropped by this scene's next
        migration). Added: 2026-10-20"""
        registry = SR.EMISSIVE_REGISTRY
        (old,) = registry.renamed_from
        other = self._other(
            private={old: {"schema": 1, "groups": {"glow": {"slot": 0}}}}
        )
        merge = RecordTransfer.between(DictStore, other)
        self.assertEqual(merge.summary(), ["Emissive Groups: 1 entry"])
        self.assertEqual(merge.payloads()[registry]["groups"], {"glow": {"slot": 0}})
        self._merge(other)
        self.assertIn("glow", registry.load(DictStore)["groups"])
        self.assertIsNone(DictStore.read(Scope.PRIVATE, old))
        both = self._other(
            private={
                old: {"schema": 1, "groups": {"stale": {"slot": 0}}},
                registry.key: {"schema": 1, "groups": {"kept": {"slot": 1}}},
            }
        )
        payload = RecordTransfer.between(DictStore, both).payloads()[registry]
        self.assertEqual(payload["groups"], {"kept": {"slot": 1}}, "the key wins")

    def test_every_codec_record_resolves_its_codec(self):
        """Each :attr:`Merge.CODEC` record finds its codec, and the table holds
        exactly those records' keys: keyed by the declarations, a renamed
        record keeps its codec. (2026-10-20: the scene's emissive registry
        took its own key while the table still spelled ``"emissive_groups"``,
        and its merge raised LookupError.) Added: 2026-10-20"""
        codecs = [spec for spec in SR.all() if spec.merge is Merge.CODEC]
        self.assertTrue(codecs)
        for spec in codecs:
            with self.subTest(record=spec.key):
                self.assertIsNotNone(SR.codec(spec))
        self.assertEqual(set(SR.CODECS), {spec.key for spec in codecs})

    # ---------------------------------------------------------------- union
    def test_a_record_this_scene_lacks_is_adopted(self):
        self._merge(self._other(private={"audio_file_map": {"1": "a.wav"}}))
        self.assertEqual(SR.AUDIO_FILE_MAP.load(DictStore), {"1": "a.wav"})

    def test_a_mapping_unites_and_this_scenes_entry_wins(self):
        SR.AUDIO_FILE_MAP.save(DictStore, {"1": "mine.wav", "3": "c.wav"})
        _, ctx = self._merge(
            self._other(private={"audio_file_map": {"1": "theirs.wav", "2": "b.wav"}})
        )
        self.assertEqual(
            SR.AUDIO_FILE_MAP.load(DictStore),
            {"1": "mine.wav", "2": "b.wav", "3": "c.wav"},
        )
        self.assertEqual(len(ctx.notes), 1)
        self.assertIn("'1'", ctx.notes[0])

    def test_a_stack_keeps_this_scenes_newest_on_top(self):
        SR.SMART_BAKE_SESSIONS.save(DictStore, [{"id": "m1"}, {"id": "m2"}])
        self._merge(
            self._other(private={"smart_bake_sessions": [{"id": "t1"}, {"id": "m2"}]})
        )
        self.assertEqual(
            [s["id"] for s in SR.SMART_BAKE_SESSIONS.load(DictStore)],
            ["t1", "m1", "m2"],
        )

    def test_an_own_record_is_never_written(self):
        SR.HIERARCHY_BASELINE.save(DictStore, {"format": 1, "paths": ["mine"]})
        self._merge(
            self._other(private={"hierarchy_baseline": {"format": 1, "paths": ["x"]}})
        )
        self.assertEqual(SR.HIERARCHY_BASELINE.load(DictStore)["paths"], ["mine"])

    def test_a_derived_record_is_left_for_the_producers(self):
        _, ctx = self._merge(
            self._other(deliverable={SR.LIGHTMAPS.key: {"objects": []}})
        )
        self.assertIsNone(DictStore.read(Scope.DELIVERABLE, SR.LIGHTMAPS.key))
        self.assertEqual(ctx.notes, [])

    def test_an_undeclared_channel_is_adopted_only_when_absent(self):
        _, ctx = self._merge({Scope.PRIVATE: {"toolX": "a"}})
        self.assertEqual(DictStore.read(Scope.PRIVATE, "toolX"), "a")
        _, ctx = self._merge({Scope.PRIVATE: {"toolX": "b"}})
        self.assertEqual(DictStore.read(Scope.PRIVATE, "toolX"), "a")
        self.assertIn("toolX", ctx.notes[0])

    # --------------------------------------------------------------- respell
    def test_names_are_respelled_before_the_merge(self):
        rename = {"|MOD:cube": "|cube1"}.get
        ctx = TransferContext(rename=rename)
        self._merge(
            self._other(
                private={"smart_bake_sessions": [{"id": "t", "nodes": ["|MOD:cube"]}]}
            ),
            ctx,
        )
        self.assertEqual(SR.SMART_BAKE_SESSIONS.load(DictStore)[0]["nodes"], ["|cube1"])

    def test_mapping_keys_are_respelled_too(self):
        ctx = TransferContext(rename={"MOD:curve1": "curve7"}.get)
        self.assertEqual(
            ctx.respell({"MOD:curve1": [1, "MOD:curve1"]}),
            {"curve7": [1, "curve7"]},
        )

    def test_a_payload_of_group_names_is_not_respelled(self):
        """A group that happens to share a node's name keeps its name."""
        ctx = TransferContext(rename={"glow": "glow1"}.get)
        self._merge(
            self._other(
                private={
                    SR.EMISSIVE_REGISTRY.key: {
                        "schema": 1,
                        "groups": {"glow": {"slot": 0}},
                    }
                }
            ),
            ctx,
        )
        self.assertIn("glow", SR.EMISSIVE_REGISTRY.load(DictStore)["groups"])

    # ---------------------------------------------------------------- codecs
    def test_shots_arrive_renumbered_and_a_renamed_one_is_noted(self):
        mine = {
            "shots": [
                {"shot_id": 1, "name": "Intro", "start": 0, "end": 10, "objects": []}
            ]
        }
        SR.SHOT_STORE.save(DictStore, mine)
        theirs = {
            "shots": [
                {"shot_id": 1, "name": "Intro", "start": 0, "end": 5, "objects": []},
                {"shot_id": 2, "name": "Outro", "start": 5, "end": 9, "objects": []},
            ]
        }
        _, ctx = self._merge(self._other(private={"shot_store": theirs}))
        merged = SR.SHOT_STORE.load(DictStore)
        self.assertEqual(
            [(s["shot_id"], s["name"]) for s in merged["shots"]],
            [(1, "Intro"), (2, "Intro_2"), (3, "Outro")],
        )
        self.assertEqual(ctx.remaps["shot_id"], {1: 2, 2: 3})
        self.assertTrue(any("'Intro_2'" in n for n in ctx.notes), ctx.notes)

    def test_parked_clips_follow_their_shot_and_get_fresh_ids(self):
        SR.SHOT_STORE.save(
            DictStore,
            {
                "shots": [
                    {"shot_id": 1, "name": "A", "start": 0, "end": 1, "objects": []}
                ]
            },
        )
        SR.KEY_STASH.save(
            DictStore,
            {
                "schema": 1,
                "scene_fps": 24,
                "clips": [{"clip_id": 1, "label": "mine", "objects": [], "curves": []}],
                "next_id": 2,
            },
        )
        other = self._other(
            private={
                "shot_store": {
                    "shots": [
                        {"shot_id": 1, "name": "B", "start": 2, "end": 3, "objects": []}
                    ]
                },
                "key_stash": {
                    "schema": 1,
                    "scene_fps": 24,
                    "clips": [
                        {
                            "clip_id": 1,
                            "label": "theirs",
                            "objects": [],
                            "curves": [],
                            "source_shot_id": 1,
                        }
                    ],
                    "next_id": 2,
                },
            }
        )
        self._merge(other)
        clips = SR.KEY_STASH.load(DictStore)["clips"]
        self.assertEqual(
            [(c["clip_id"], c["label"]) for c in clips], [(1, "mine"), (2, "theirs")]
        )
        # Their shot 1 arrived as shot 2; the clip follows it.
        self.assertEqual(clips[1]["source_shot_id"], 2)

    def test_parked_clips_rescale_to_this_scenes_rate(self):
        SR.KEY_STASH.save(
            DictStore, {"schema": 1, "scene_fps": 30, "clips": [], "next_id": 1}
        )
        _, ctx = self._merge(
            self._other(
                private={
                    "key_stash": {
                        "schema": 1,
                        "scene_fps": 24,
                        "clips": [
                            {
                                "clip_id": 5,
                                "label": "x",
                                "objects": [],
                                "curves": [{"times": [24.0]}],
                            }
                        ],
                    }
                }
            )
        )
        clip = SR.KEY_STASH.load(DictStore)["clips"][0]
        self.assertAlmostEqual(clip["curves"][0]["times"][0], 30.0)
        self.assertTrue(any("30" in n for n in ctx.notes))

    def test_a_codec_respells_only_the_names_it_holds(self):
        """The shot store's codec respells members, lists and ledger curves; a
        shot named like a node the crossing renamed keeps its name."""
        ctx = TransferContext(
            rename={
                "door": "door1",
                "|grp|door": "|grp|door1",
                "door_translateX": "door_translateX1",
            }.get
        )
        theirs = {
            "shots": [
                {
                    "shot_id": 1,
                    "name": "door",
                    "start": 0,
                    "end": 5,
                    "objects": ["|grp|door"],
                }
            ],
            "hidden_objects": ["door"],
            "edit_ledger": {"keys": {"door_translateX": [[0.0, 1, "start"]]}},
        }
        self._merge(self._other(private={"shot_store": theirs}), ctx)
        merged = SR.SHOT_STORE.load(DictStore)
        self.assertEqual(merged["shots"][0]["name"], "door")
        self.assertEqual(merged["shots"][0]["objects"], ["|grp|door1"])
        self.assertEqual(merged["hidden_objects"], ["door1"])
        self.assertEqual(list(merged["edit_ledger"]["keys"]), ["door_translateX1"])

    def test_a_parked_clip_keeps_its_label(self):
        """The key stash's codec respells a clip's objects and curve refs, never
        its label."""
        ctx = TransferContext(rename={"cube": "cube1"}.get)
        clip = {
            "clip_id": 1,
            "label": "cube",
            "objects": ["cube"],
            "curves": [{"times": [1.0], "curve": {"name": "cube", "uuid": None}}],
        }
        self._merge(
            self._other(
                private={"key_stash": {"schema": 1, "scene_fps": 24, "clips": [clip]}}
            ),
            ctx,
        )
        landed = SR.KEY_STASH.load(DictStore)["clips"][0]
        self.assertEqual(landed["label"], "cube")
        self.assertEqual(landed["objects"], ["cube1"])
        self.assertEqual(landed["curves"][0]["curve"]["name"], "cube1")

    def test_an_owner_that_resolved_the_names_skips_the_respell(self):
        ctx = TransferContext(rename={"a": "b"}.get)
        RecordTransfer.merge_record(
            DictStore,
            SR.SMART_BAKE_SESSIONS,
            [{"id": "s", "node": "a"}],
            ctx,
            respelled=True,
        )
        self.assertEqual(SR.SMART_BAKE_SESSIONS.load(DictStore)[0]["node"], "a")

    def test_payloads_are_the_other_scenes_records_by_declaration(self):
        """Decoded, respelled through a context, and keyed by spec -- the
        private registry and a deliverable record each its own."""
        other = self._other(
            private={
                SR.EMISSIVE_REGISTRY.key: {"schema": 1, "groups": {"g": {"slot": 0}}},
                "key_stash": {
                    "schema": 1,
                    "clips": [
                        {"clip_id": 1, "label": "a", "objects": ["a"], "curves": []}
                    ],
                },
            },
            deliverable={
                SR.VISIBILITY.key: {
                    "version": SR.VISIBILITY.version,
                    "tracks": [],
                }
            },
        )
        merge = RecordTransfer.between(DictStore, other)
        payloads = merge.payloads(TransferContext(rename={"a": "a1"}.get))
        self.assertEqual(payloads[SR.EMISSIVE_REGISTRY]["groups"], {"g": {"slot": 0}})
        self.assertEqual(payloads[SR.VISIBILITY]["tracks"], [])
        self.assertEqual(payloads[SR.KEY_STASH]["clips"][0]["objects"], ["a1"])
        self.assertEqual(payloads[SR.KEY_STASH]["clips"][0]["label"], "a")
        self.assertEqual(merge.payloads()[SR.KEY_STASH]["clips"][0]["objects"], ["a"])

    def test_spell_keeps_a_name_without_a_counterpart(self):
        ctx = TransferContext(rename={"a": "b"}.get)
        self.assertEqual(ctx.spell("a"), "b")
        self.assertEqual(ctx.spell("zzz"), "zzz")
        self.assertEqual(TransferContext().spell("a"), "a")

    def test_one_failing_record_does_not_stop_the_rest(self):
        from unittest import mock

        with mock.patch.object(SR, "codec", side_effect=RuntimeError("boom")):
            _, ctx = self._merge(
                self._other(
                    private={
                        "shot_store": {
                            "shots": [{"shot_id": 1, "name": "A", "start": 0, "end": 1}]
                        },
                        "audio_file_map": {"1": "a.wav"},
                    }
                )
            )
        self.assertEqual(SR.AUDIO_FILE_MAP.load(DictStore), {"1": "a.wav"})
        self.assertTrue(any("boom" in n for n in ctx.notes), ctx.notes)


class TestRecordHandoff(SceneRecordsCase):
    """A DCC hand-off: the portable records ride the sidecar -- under their
    own section, or keyed in the generic one -- and land by the same merge."""

    def test_only_portable_records_ride_and_each_where_it_is_declared(self):
        SR.EMISSIVE_REGISTRY.save(
            DictStore, {"schema": 1, "groups": {"g": {"slot": 0}}}
        )
        SR.SMART_BAKE_SESSIONS.save(DictStore, [{"id": "s"}])  # Maya-bound
        SR.SHOT_STORE.save(DictStore, {"shots": [{"shot_id": 1, "name": "A"}]})
        out = RecordTransfer.sections(DictStore, TransferContext())
        self.assertEqual(sorted(out), ["records", "shots"])
        self.assertEqual(list(out["records"]), [SR.EMISSIVE_REGISTRY.key])
        # A record with a section of its own ships in that section's wire
        # shape -- the codec's (`ShotTransfer.encode`'s envelope, what every
        # consumer `decode`s), never the bare record.
        self.assertIn("version", out["shots"])
        self.assertEqual(out["shots"]["store"]["shots"][0]["name"], "A")

    def test_a_sectioned_record_lands_through_its_codec_without_an_owner(self):
        """No owner row: the codec that defines the section's shape both
        writes and reads it, so what `sections` ships is what `receive`
        lands -- the store, not its envelope."""
        SR.SHOT_STORE.save(
            DictStore, {"shots": [{"shot_id": 1, "name": "A", "start": 0, "end": 10}]}
        )
        out = RecordTransfer.sections(DictStore, TransferContext())
        DictStore.reset()
        ctx = RecordTransfer.receive(out, DictStore, TransferContext())
        landed = SR.SHOT_STORE.load(DictStore)
        self.assertNotIn("store", landed, ctx.notes)
        self.assertEqual([s["name"] for s in landed["shots"]], ["A"])

    def test_an_owner_takes_over_its_records_half(self):
        class Owner:
            sent = []

            @staticmethod
            def transfer_out(ctx):
                return {"from": "owner"}

            @classmethod
            def transfer_in(cls, payload, ctx):
                cls.sent.append(payload)

        out = RecordTransfer.sections(
            DictStore, TransferContext(), owners={"shot_store": Owner}
        )
        self.assertEqual(out["shots"], {"from": "owner"})
        RecordTransfer.receive(out, DictStore, TransferContext(), {"shot_store": Owner})
        self.assertEqual(Owner.sent, [{"from": "owner"}])

    def test_a_received_record_merges_by_its_rule(self):
        """A portable UNION record's entries combine by key, the scene's own
        winning a collision (noted)."""
        dirs = SR.LIGHTMAP_DIRS
        dirs.save(DictStore, {"a": "own/a"})
        manifest = {"records": {dirs.key: {"a": "theirs/a", "b": "theirs/b"}}}
        ctx = RecordTransfer.receive(manifest, DictStore, TransferContext())
        merged = dirs.load(DictStore)
        self.assertEqual(merged["a"], "own/a")
        self.assertIn("b", merged)
        self.assertTrue(ctx.notes)

    def test_a_bad_record_is_noted_and_never_costs_the_rest(self):
        class Broken:
            @staticmethod
            def transfer_in(payload, ctx):
                raise RuntimeError("nope")

        manifest = {
            "shots": {"shots": []},
            "records": {
                SR.EMISSIVE_REGISTRY.key: {"schema": 1, "groups": {"g": {"slot": 0}}}
            },
        }
        ctx = RecordTransfer.receive(
            manifest, DictStore, TransferContext(), {"shot_store": Broken}
        )
        self.assertIn("g", SR.EMISSIVE_REGISTRY.load(DictStore)["groups"])
        self.assertTrue(any("nope" in n for n in ctx.notes))

    def test_the_generic_section_is_the_manifests(self):
        """The engine names the section without importing the manifest class
        (the lighter module imports first); the two spellings are one."""
        from pythontk.core_utils.handoff.manifest import HandoffManifest

        self.assertEqual(RecordTransfer.RECORDS_SECTION, HandoffManifest.RECORDS)
        self.assertIn(SR.SHOT_STORE.section, HandoffManifest.SECTIONS)


class _CarrierStore(DictStore):
    """A store whose other-scene carriers are plain ``{key: text}`` dicts, each
    one's removal recorded -- the smallest DCC half a crossing needs."""

    deleted: list = []

    @classmethod
    def reset(cls):
        cls.data = {}
        cls.deleted = []

    @classmethod
    def _carrier_values(cls, carrier):
        return dict(carrier)

    @classmethod
    def _delete_carrier(cls, carrier):
        cls.deleted.append(carrier)


class TestStoreCrossings(unittest.TestCase):
    """``SceneStoreBase``'s crossing surface: written once, over the few carrier
    hooks a DCC store overrides."""

    def setUp(self):
        _CarrierStore.reset()

    @staticmethod
    def _private(**records):
        return {Scope.PRIVATE: {k: json.dumps(v) for k, v in records.items()}}

    @staticmethod
    def _owner(calls):
        class Owner:
            @staticmethod
            def merge_carrier(carriers, other, ctx):
                calls.append(("merge", dict(other)))

            @staticmethod
            def discard_carrier(carriers, other, ctx):
                calls.append(("discard", dict(other)))

        return Owner

    def test_a_merge_applies_runs_the_owners_and_removes_the_carriers(self):
        calls = []
        owners = {"audio_file_map": self._owner(calls)}
        with mock.patch.object(_CarrierStore, "owners", return_value=owners):
            _CarrierStore.merge_carriers(
                self._private(audio_file_map={"2": "b.wav"}), source="MOD"
            )
        self.assertEqual(SR.AUDIO_FILE_MAP.load(_CarrierStore), {"2": "b.wav"})
        self.assertEqual(len(_CarrierStore.deleted), 1)
        ((kind, other),) = calls
        self.assertEqual(kind, "merge")
        self.assertEqual(other[SR.AUDIO_FILE_MAP], {"2": "b.wav"})

    def test_a_discard_writes_nothing_and_says_what_went(self):
        calls = []
        owners = {"audio_file_map": self._owner(calls)}
        with mock.patch.object(_CarrierStore, "owners", return_value=owners):
            ctx = _CarrierStore.discard_carriers(
                self._private(audio_file_map={"2": "b.wav"}), source="MOD"
            )
        self.assertIsNone(SR.AUDIO_FILE_MAP.load(_CarrierStore))
        self.assertEqual(len(_CarrierStore.deleted), 1)
        self.assertEqual([kind for kind, _other in calls], ["discard"])
        self.assertTrue(any("MOD: not merged" in n for n in ctx.notes), ctx.notes)

    def test_a_carrier_the_store_calls_its_own_is_merged_into_nothing(self):
        """``_foreign_carriers`` decides: an adopted carrier is not merged into
        itself -- but a discard still removes it."""
        with mock.patch.object(_CarrierStore, "_foreign_carriers", return_value={}):
            _CarrierStore.merge_carriers(self._private(audio_file_map={"2": "b"}))
            self.assertEqual(_CarrierStore.deleted, [])
            _CarrierStore.discard_carriers(self._private(audio_file_map={"2": "b"}))
        self.assertEqual(len(_CarrierStore.deleted), 1)

    def test_nothing_given_is_nothing_done(self):
        ctx = _CarrierStore.merge_carriers({})
        self.assertEqual((ctx.notes, _CarrierStore.deleted), ([], []))

    def test_deliverables_are_produced_again_once_the_carriers_are_gone(self):
        """They are produced from what the scene keeps.  Produced while the
        other carriers stood, a discard derived them from the records it
        drops -- a carrier the import adopted as the scene's own is still
        the scene's then -- and wrote them into that carrier, which went
        next."""
        order = []
        carriers = {
            **self._private(audio_file_map={"2": "b.wav"}),
            Scope.DELIVERABLE: {SR.LIGHTMAPS.key: json.dumps({"objects": [1]})},
        }
        with (
            mock.patch.object(
                _CarrierStore,
                "_delete_carrier",
                side_effect=lambda c: order.append("x"),
            ),
            mock.patch.object(
                _CarrierStore,
                "_rederive",
                side_effect=lambda s, c: order.append("derive"),
            ),
        ):
            _CarrierStore.discard_carriers(carriers)
            self.assertEqual(order, ["x", "x", "derive"])
            del order[:]
            _CarrierStore.merge_carriers(carriers)
            self.assertEqual(order, ["x", "x", "derive"])

    def test_owners_store_what_they_hold_unsaved_before_a_crossing_reads(self):
        """An owner's store may write on idle (Maya coalesces a mutation into
        one deferred write), so a record can be an edit behind.  The hand-off
        routes ask the owners to store it first, and an importer does through
        ``flush_owners`` before the other scene's nodes land.  A settle does
        not: by then the import may have adopted the other scene's carrier as
        this scene's own, and the pending state would be written into it."""
        flushed = []

        class Owner:
            @staticmethod
            def flush_pending():
                flushed.append(True)
                SR.AUDIO_FILE_MAP.save(_CarrierStore, {"1": "mine.wav"})

        owners = {"audio_file_map": Owner}
        with mock.patch.object(_CarrierStore, "owners", return_value=owners):
            _CarrierStore.transfer_sections()
            _CarrierStore.receive_sections({})
            _CarrierStore.flush_owners()
            self.assertEqual(len(flushed), 3)
            _CarrierStore.merge_carriers(self._private(audio_file_map={"2": "b.wav"}))
            _CarrierStore.discard_carriers(self._private(audio_file_map={"3": "c"}))
        self.assertEqual(len(flushed), 3, "a settle wrote pending state")
        self.assertEqual(
            SR.AUDIO_FILE_MAP.load(_CarrierStore), {"1": "mine.wav", "2": "b.wav"}
        )

    def test_the_owners_table_resolves_lazily_and_skips_the_missing(self):
        table = {
            "a": (
                "pythontk.core_utils.engines.scene_export.record_transfer",
                "RecordTransfer",
            ),
            "b": ("no_such_module_anywhere", "Owner"),
        }
        with mock.patch.object(_CarrierStore, "OWNERS", table):
            self.assertEqual(_CarrierStore.owners(), {"a": RecordTransfer})
            # One owner, by key: what a part ranked below it resolves.
            self.assertIs(_CarrierStore.owner("a"), RecordTransfer)
            self.assertIsNone(_CarrierStore.owner("b"))
            self.assertIsNone(_CarrierStore.owner("not_declared"))

    def test_a_handoff_round_trips_through_the_store(self):
        SR.EMISSIVE_REGISTRY.save(
            _CarrierStore, {"schema": 1, "groups": {"g": {"slot": 0, "default": 1.0}}}
        )
        sections = _CarrierStore.transfer_sections()
        _CarrierStore.reset()
        _CarrierStore.receive_sections(sections, source="send")
        self.assertIn("g", SR.EMISSIVE_REGISTRY.load(_CarrierStore)["groups"])


class TestLightmapDirs(SceneRecordsCase):
    """The folder each baked lightmap was written to (BACKLOG 2026-09-19).
    It rode every per-object marker, and a marker rides the FBX -- so the
    hint moved to a PRIVATE record: never on a deliverable, still crossing a
    hand-off so a pulled scene finds its maps."""

    def test_it_never_ships_and_it_crosses_a_handoff(self):
        for spec in (SR.LIGHTMAP_DIRS, SR.LIGHTMAP_WRITERS):
            self.assertIs(spec.scope, Scope.PRIVATE)
            self.assertNotIn(spec, SR.deliverable())
            self.assertIn(spec, SR.portable())
        SR.LIGHTMAP_DIRS.save(DictStore, {"floor_lightmap.exr": "sourceimages/lm"})
        out = RecordTransfer.sections(DictStore, TransferContext())
        self.assertEqual(
            out["records"]["lightmap_dirs"], {"floor_lightmap.exr": "sourceimages/lm"}
        )

    def test_a_crossing_keeps_this_scenes_folder_and_adopts_the_rest(self):
        """File names, not scene names: a rename context must not touch a key,
        and this scene's own hint for a map wins over the arriving one."""
        SR.LIGHTMAP_DIRS.save(DictStore, {"a.exr": "mine", "c.exr": "mine"})
        manifest = {"records": {"lightmap_dirs": {"a.exr": "theirs", "b.exr": "x"}}}
        RecordTransfer.receive(
            manifest,
            DictStore,
            TransferContext(rename=lambda n: "renamed.exr" if n == "a.exr" else n),
        )
        self.assertEqual(
            SR.LIGHTMAP_DIRS.load(DictStore),
            {"a.exr": "mine", "b.exr": "x", "c.exr": "mine"},
        )


class TestLightmapBakeState(SceneRecordsCase):
    """What each lightmap was baked against (``BakeBaseline``): build-time
    state keyed by this scene's own node names. Never on a deliverable (the
    marker and ``lightmaps`` ship), never across a hand-off, and
    another scene's copy is dropped."""

    def test_it_is_private_its_own_and_never_crosses(self):
        spec = SR.LIGHTMAP_BAKE_STATE
        self.assertEqual(spec.key, "lightmap_bake_state")
        self.assertIs(spec.scope, Scope.PRIVATE)
        self.assertIs(spec.merge, Merge.OWN)
        self.assertFalse(spec.portable)
        self.assertFalse(spec.respell)
        self.assertNotIn(spec, SR.deliverable())
        self.assertNotIn(spec, SR.portable())
        self.assertIs(SR.by_key("lightmap_bake_state", Scope.PRIVATE), spec)

    def test_its_payload_carries_its_version_and_another_scenes_is_dropped(self):
        SR.LIGHTMAP_BAKE_STATE.save(DictStore, {"serial": 3, "objects": {}})
        self.assertEqual(SR.LIGHTMAP_BAKE_STATE.load(DictStore)["version"], 1)
        other = {
            Scope.PRIVATE: {
                "lightmap_bake_state": json.dumps({"version": 1, "serial": 9})
            },
            Scope.DELIVERABLE: {},
        }
        merge = RecordTransfer.between(DictStore, other)
        self.assertTrue(merge.is_empty)
        merge.apply(DictStore, None)
        self.assertEqual(SR.LIGHTMAP_BAKE_STATE.load(DictStore)["serial"], 3)


class TestProjectRelativePaths(SceneRecordsCase):
    """BACKLOG 2026-09-22 (decided 2026-09-23): a path record stores its files
    relative to the SCENE'S OWN project, ``../`` chains included, so it has to
    follow the scene -- a Save As into another project re-spells it, a hand-off
    ships it absolute, and a module's copy arrives spelled from the host."""

    OLD = os.path.abspath("/work/proj")
    NEW = os.path.abspath("/work/shows/copy")
    LIB = os.path.abspath("/work/library")

    @staticmethod
    def _abs(base, spelled):
        return os.path.normpath(os.path.join(base, spelled)).replace("\\", "/")

    def test_the_path_records(self):
        """The audio and lightmap records, and the hierarchy baseline's writer
        stamp -- the one path in a record that is otherwise data."""
        self.assertEqual(
            SR.with_paths(),
            [
                SR.HIERARCHY_BASELINE,
                SR.AUDIO_FILE_MAP,
                SR.LIGHTMAP_DIRS,
                SR.LIGHTMAP_WRITERS,
            ],
        )
        self.assertEqual(SR.HIERARCHY_BASELINE.path_keys, ("scene",))
        self.assertIsNone(SR.LIGHTMAP_DIRS.path_keys, "every value is a path")

    def test_a_key_scoped_record_respells_its_path_keys_alone(self):
        """2026-09-24: the hierarchy baseline names the scene that recorded it,
        and a copy saved into another project resolved that stamp from the
        wrong project -- to a file not there, read as "renamed" -- and trusted
        its source's baseline. Its hash is a string too, and must not move."""
        spec = SR.HIERARCHY_BASELINE
        record = {
            "format": 1,
            "paths": ["GRP", "GRP|part"],
            "hash": "e3b0c442",
            "object_count": 2,
            "scene": "scenes/source.ma",
        }
        spec.save(DictStore, record)
        self.assertEqual(SR.rebase_paths(DictStore, self.OLD, self.NEW), 1)
        moved = spec.load(DictStore)
        self.assertEqual(
            self._abs(self.NEW, moved["scene"]),
            self._abs(self.OLD, "scenes/source.ma"),
        )
        self.assertEqual(
            {k: v for k, v in moved.items() if k != "scene"},
            {k: v for k, v in record.items() if k != "scene"},
        )

    def test_a_copy_saved_into_another_project_still_names_the_writer(self):
        """What keeps a Save As copy's re-bake from deleting the maps its
        source still reads: the writer is the source file wherever the copy
        lands."""
        SR.LIGHTMAP_WRITERS.save(DictStore, {"a.exr": "scenes/room.ma"})
        self.assertEqual(SR.rebase_paths(DictStore, self.OLD, self.NEW), 1)
        landed = SR.LIGHTMAP_WRITERS.load(DictStore)["a.exr"]
        self.assertEqual(
            self._abs(self.NEW, landed), self._abs(self.OLD, "scenes/room.ma")
        )

    def test_a_save_into_another_project_respells_every_path(self):
        SR.LIGHTMAP_DIRS.save(DictStore, {"a.exr": "sourceimages/lm", "b.exr": "."})
        SR.AUDIO_FILE_MAP.save(DictStore, {"t1": "../library/hit.wav"})
        self.assertEqual(SR.rebase_paths(DictStore, self.OLD, self.NEW), 3)
        dirs = SR.LIGHTMAP_DIRS.load(DictStore)
        audio = SR.AUDIO_FILE_MAP.load(DictStore)
        self.assertEqual(
            self._abs(self.NEW, dirs["a.exr"]), self._abs(self.OLD, "sourceimages/lm")
        )
        self.assertEqual(self._abs(self.NEW, dirs["b.exr"]), self._abs(self.OLD, "."))
        self.assertEqual(audio["t1"], "../../library/hit.wav")
        # Saved again where it already lives: nothing moves, nothing is written.
        with mock.patch.object(DictStore, "write", wraps=DictStore.write) as write:
            self.assertEqual(SR.rebase_paths(DictStore, self.NEW, self.NEW), 0)
        write.assert_not_called()

    def test_the_same_project_normalizes_an_absolute_entry(self):
        """Written before the rule, or arrived across a hand-off."""
        legacy = os.path.join(self.LIB, "hit.wav")
        SR.AUDIO_FILE_MAP.save(DictStore, {"t1": legacy, "t2": "sound/vo.wav"})
        self.assertEqual(SR.rebase_paths(DictStore, self.OLD, self.OLD), 1)
        self.assertEqual(
            SR.AUDIO_FILE_MAP.load(DictStore),
            {"t1": "../library/hit.wav", "t2": "sound/vo.wav"},
        )

    def test_a_handoff_ships_absolute_and_lands_spelled_from_this_scene(self):
        SR.LIGHTMAP_DIRS.save(DictStore, {"a.exr": "sourceimages/lm"})
        out = RecordTransfer.sections(DictStore, TransferContext(path_base=self.OLD))
        shipped = out["records"]["lightmap_dirs"]["a.exr"]
        self.assertEqual(shipped, self._abs(self.OLD, "sourceimages/lm"))

        DictStore.reset()
        RecordTransfer.receive(out, DictStore, TransferContext(path_base=self.NEW))
        landed = SR.LIGHTMAP_DIRS.load(DictStore)["a.exr"]
        self.assertEqual(landed, "../../proj/sourceimages/lm")
        self.assertEqual(self._abs(self.NEW, landed), shipped)
        # An unsaved receiving scene has no project yet: it keeps them absolute
        # (its first save re-spells them).
        DictStore.reset()
        RecordTransfer.receive(out, DictStore, TransferContext())
        self.assertEqual(SR.LIGHTMAP_DIRS.load(DictStore)["a.exr"], shipped)

    def test_the_store_crossings_spell_paths_from_its_own_project(self):
        """``SceneStoreBase`` supplies the bases: the store's
        :meth:`project_root` for a hand-off either way, and a module merge's
        own project passed by the importer (the DCC knows the file)."""
        module = os.path.abspath("/work/modules/office")
        _CarrierStore.reset()
        self.addCleanup(_CarrierStore.reset)
        with mock.patch.object(
            _CarrierStore, "project_root", classmethod(lambda cls: self.OLD)
        ):
            SR.LIGHTMAP_DIRS.save(_CarrierStore, {"a.exr": "sourceimages/lm"})
            shipped = _CarrierStore.transfer_sections()["records"]["lightmap_dirs"]
            self.assertEqual(shipped["a.exr"], self._abs(self.OLD, "sourceimages/lm"))
            _CarrierStore.reset()
            _CarrierStore.receive_sections(
                {"records": {"lightmap_dirs": {"b.exr": shipped["a.exr"]}}}
            )
            self.assertEqual(
                SR.LIGHTMAP_DIRS.load(_CarrierStore), {"b.exr": "sourceimages/lm"}
            )
            _CarrierStore.merge_carriers(
                {
                    Scope.PRIVATE: {
                        "lightmap_dirs": json.dumps({"m.exr": "sourceimages/lm"})
                    }
                },
                source="MOD",
                source_path_base=module,
            )
        self.assertEqual(
            SR.LIGHTMAP_DIRS.load(_CarrierStore)["m.exr"],
            "../modules/office/sourceimages/lm",
        )

    def test_an_entry_written_while_unsaved_never_crosses(self):
        """A writer entry of ``""`` is "this scene, while unsaved": it names no
        file, so it cannot leave its scene -- landed, it would claim the maps
        for the scene it lands in (``FileDependencies.written_here``), whose
        re-bake then deletes what the other scene still reads."""
        room = self._abs(self.OLD, "scenes/room.ma")
        SR.LIGHTMAP_WRITERS.save(DictStore, {"a.exr": "", "b.exr": "scenes/room.ma"})
        out = RecordTransfer.sections(DictStore, TransferContext(path_base=self.OLD))
        self.assertEqual(out["records"]["lightmap_writers"], {"b.exr": room})

        DictStore.reset()
        arriving = {"records": {"lightmap_writers": {"a.exr": "", "b.exr": room}}}
        RecordTransfer.receive(arriving, DictStore, TransferContext())
        self.assertEqual(SR.LIGHTMAP_WRITERS.load(DictStore), {"b.exr": room})

        DictStore.reset()
        RecordTransfer.merge_record(
            DictStore,
            SR.LIGHTMAP_WRITERS,
            {"m.exr": ""},
            TransferContext(source_path_base=self.NEW, path_base=self.OLD),
        )
        self.assertIsNone(SR.LIGHTMAP_WRITERS.load(DictStore))

    def test_a_modules_copy_arrives_spelled_from_the_host(self):
        """A referenced module's paths are spelled from ITS project."""
        module = os.path.abspath("/work/modules/office")
        RecordTransfer.merge_record(
            DictStore,
            SR.LIGHTMAP_DIRS,
            {"m.exr": "sourceimages/lm"},
            TransferContext(source_path_base=module, path_base=self.OLD),
        )
        landed = SR.LIGHTMAP_DIRS.load(DictStore)["m.exr"]
        self.assertEqual(landed, "../modules/office/sourceimages/lm")
        # normcase: OLD is taken at import and *module* here, and a test between
        # them can leave the working directory's drive spelled the other case
        # (o: / O:), the same folder to Windows.
        self.assertEqual(
            os.path.normcase(self._abs(self.OLD, landed)),
            os.path.normcase(self._abs(module, "sourceimages/lm")),
        )


class TestWriterStamp(SceneRecordsCase):
    """A store's writer stamp (2026-09-24): what a record stores to say which
    scene file made it -- a Save As copy carries every record verbatim -- and
    whether a stamp is the open scene's. A DCC store answers ``scene_path``;
    the rest is decided once, here."""

    def setUp(self):
        super().setUp()
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        temp = TempArtifacts("ptk_writer_stamp", policy="scoped")
        self.addCleanup(temp.cleanup)
        self.proj = os.path.join(temp.dir_path(), "proj")
        os.makedirs(os.path.join(self.proj, "scenes"))
        pathlib.Path(self.proj, "workspace.mel").write_text("")
        self.source = os.path.join(self.proj, "scenes", "source.ma")
        self.copy = os.path.join(self.proj, "scenes", "copy.ma")
        for path in (self.source, self.copy):
            pathlib.Path(path).write_text("")
        test = self
        self.scene = ""

        class Store(DictStore):
            @classmethod
            def scene_path(cls):
                return test.scene

        self.store = Store

    def test_an_unsaved_scene_stamps_empty_and_owns_only_that(self):
        self.assertEqual(self.store.writer_stamp(), "")
        self.assertIsNone(self.store.project_root())
        self.assertTrue(self.store.written_here(""))
        self.assertFalse(self.store.written_here(None), "no stamp is nobody's")

    def test_a_saved_scene_stamps_itself_from_its_own_project(self):
        self.scene = self.source
        self.assertEqual(self.store.writer_stamp(), "scenes/source.ma")
        self.assertTrue(self.store.written_here("scenes/source.ma"))

    def test_a_save_as_copy_does_not_own_its_sources_stamp(self):
        self.scene = self.copy
        self.assertFalse(self.store.written_here("scenes/source.ma"))
        os.remove(self.source)  # ...until the source is gone: a rename
        self.assertTrue(self.store.written_here("scenes/source.ma"))

    # -- the first save (BACKLOG 2026-09-23) --------------------------------------

    def _unsaved_records(self):
        """What a scene holds after a lightmap bake and a baseline while
        unsaved: every stamp ``""``, beside one a copied record brought."""
        SR.LIGHTMAP_WRITERS.save(
            self.store, {"a.exr": "", "b.exr": "scenes/elsewhere.ma"}
        )
        SR.HIERARCHY_BASELINE.save(
            self.store, {"format": 1, "paths": ["|GRP"], "hash": "h", "scene": ""}
        )

    def test_the_stamp_records(self):
        """Every record holding a writer stamp; each is a path record too."""
        self.assertEqual(SR.with_stamps(), [SR.HIERARCHY_BASELINE, SR.LIGHTMAP_WRITERS])
        self.assertEqual(SR.HIERARCHY_BASELINE.stamp_keys, ("scene",))
        self.assertIsNone(SR.LIGHTMAP_WRITERS.stamp_keys, "every value")
        for spec in SR.with_stamps():
            self.assertTrue(spec.paths, spec.key)
            if spec.path_keys is not None:  # every stamp key re-spelled too
                self.assertLessEqual(set(spec.stamp_keys or ()), set(spec.path_keys))
                self.assertIsNotNone(spec.stamp_keys, spec.key)

    def test_a_first_save_stamps_what_was_made_unsaved(self):
        """``""`` is the scene's own only while it is unsaved: left as it was,
        whatever the bake stamped was nobody's once saved, so a re-bake never
        retired those maps."""
        self._unsaved_records()
        self.store.respell_for_write(self.source, None, first_save=True)
        self.assertEqual(
            SR.LIGHTMAP_WRITERS.load(self.store),
            {"a.exr": "scenes/source.ma", "b.exr": "scenes/elsewhere.ma"},
        )
        self.assertEqual(
            SR.HIERARCHY_BASELINE.load(self.store)["scene"], "scenes/source.ma"
        )
        self.scene = self.source  # the save done: the scene is that file
        self.assertTrue(self.store.written_here("scenes/source.ma"))

    def test_a_copy_is_not_a_first_save_and_its_snapshot_puts_all_back(self):
        """A copy of the scene -- an export, an autosave -- is written spelled
        for ITS project, then the open scene gets its own spelling back,
        exactly; an unsaved scene's copy keeps its ``""`` stamps."""
        self._unsaved_records()
        SR.LIGHTMAP_DIRS.save(self.store, {"a.exr": "sourceimages/lm"})
        before = self.store.path_records()
        other = os.path.join(os.path.dirname(self.proj), "other")
        os.makedirs(os.path.join(other, "scenes"))
        pathlib.Path(other, "workspace.mel").write_text("")
        snapshot = self.store.respell_for_write(
            os.path.join(other, "scenes", "copy.ma"), self.proj
        )
        self.assertEqual(snapshot, before)
        self.assertEqual(
            SR.LIGHTMAP_DIRS.load(self.store), {"a.exr": "../proj/sourceimages/lm"}
        )
        self.assertEqual(SR.LIGHTMAP_WRITERS.load(self.store)["a.exr"], "")
        self.assertEqual(self.store.restore_path_records(snapshot), 2)
        self.assertEqual(self.store.path_records(), before)

    # -- a renamed scene's stamps (2026-10-21) ------------------------------------

    def _renamed_records(self):
        """The open scene is ``copy.ma``, renamed from ``gone.ma``: its records
        still name the old file, beside one its Save As source wrote."""
        self.scene = self.copy
        SR.LIGHTMAP_WRITERS.save(
            self.store,
            {
                "a.exr": "scenes/gone.ma",
                "b.exr": "scenes/source.ma",
                "c.exr": "scenes/copy.ma",
                "d.exr": "",
            },
        )
        SR.HIERARCHY_BASELINE.save(
            self.store,
            {"format": 1, "paths": ["|GRP"], "hash": "h", "scene": "scenes/gone.ma"},
        )

    def test_a_save_gives_a_gone_writers_stamps_to_the_scene_saved(self):
        """Gone, the old name read as this scene's own -- until a file came
        back there (a Save As back to it, a restored backup): that file then
        owned the maps, and its re-bake wrote over what this scene reads.
        The save writes the ownership down."""
        self._renamed_records()
        self.store.respell_for_write(self.copy, self.proj, saved_from=self.copy)
        self.assertEqual(
            SR.LIGHTMAP_WRITERS.load(self.store),
            {
                "a.exr": "scenes/copy.ma",
                "b.exr": "scenes/source.ma",
                "c.exr": "scenes/copy.ma",
                "d.exr": "",
            },
        )
        self.assertEqual(SR.HIERARCHY_BASELINE.load(self.store)["scene"], "scenes/copy.ma")
        pathlib.Path(self.proj, "scenes", "gone.ma").write_text("")  # back again
        self.assertTrue(self.store.written_here("scenes/copy.ma"))

    def test_a_save_as_gives_them_to_the_file_it_was_saved_from(self):
        """That file, still on disk, reads them as its own (gone): the new
        file must name it, or both would write in place."""
        self._renamed_records()
        new = os.path.join(self.proj, "scenes", "next.ma")
        self.store.respell_for_write(new, self.proj, saved_from=self.copy)
        self.assertEqual(SR.LIGHTMAP_WRITERS.load(self.store)["a.exr"], "scenes/copy.ma")
        self.scene = new
        self.assertFalse(self.store.written_here("scenes/copy.ma"))

    def test_a_copy_adopts_nothing(self):
        """An export or autosave (no *saved_from*) leaves the stamps alone."""
        self._renamed_records()
        self.store.respell_for_write(os.path.join(self.proj, "scenes", "x.ma"), self.proj)
        self.assertEqual(SR.LIGHTMAP_WRITERS.load(self.store)["a.exr"], "scenes/gone.ma")

    def test_a_save_as_into_another_project_spells_the_adopted_stamp_from_it(self):
        self._renamed_records()
        other = os.path.join(os.path.dirname(self.proj), "other")
        os.makedirs(os.path.join(other, "scenes"))
        pathlib.Path(other, "workspace.mel").write_text("")
        new = os.path.join(other, "scenes", "moved.ma")
        self.store.respell_for_write(new, self.proj, saved_from=self.copy)
        self.assertEqual(
            SR.LIGHTMAP_WRITERS.load(self.store)["a.exr"], "../proj/scenes/copy.ma"
        )

    def test_no_stamp_stamps_nothing(self):
        self._unsaved_records()
        self.assertEqual(SR.stamp_unsaved(self.store, ""), 0)
        self.assertEqual(SR.LIGHTMAP_WRITERS.load(self.store)["a.exr"], "")


class TestClause(unittest.TestCase):
    """A sentence of an assembly that only some payloads call for."""

    PLANES = {"planes": [{"type": "projected"}, {"type": "horizon", "on": True}]}

    def test_a_value_somewhere_down_a_list(self):
        self.assertTrue(Clause("x", ("planes", "type"), "horizon").applies(self.PLANES))
        self.assertFalse(Clause("x", ("planes", "type"), "baked").applies(self.PLANES))

    def test_a_value_matches_its_type_too(self):
        """``True`` is not ``1``: a flag never matches a count."""
        self.assertTrue(Clause("x", ("planes", "on"), True).applies(self.PLANES))
        self.assertFalse(
            Clause("x", ("n",), True).applies({"n": 1}),
        )

    def test_present_is_anything_but_null_or_empty(self):
        clause = Clause("x", ("probe",))
        self.assertTrue(clause.applies({"probe": {"map": "p.exr"}}))
        self.assertTrue(clause.applies({"probe": 0}))
        for absent in (
            {},
            {"probe": None},
            {"probe": {}},
            {"probe": []},
            {"probe": ""},
        ):
            with self.subTest(payload=absent):
                self.assertFalse(clause.applies(absent))

    def test_an_unknown_payload_and_an_empty_path_always_apply(self):
        self.assertTrue(Clause("x", ("probe", "highlights")).applies(Clause.UNKNOWN))
        self.assertTrue(Clause("x").applies({}))

    def test_join_states_every_clause_without_a_payload(self):
        clauses = (Clause("B.", ("b",)), Clause("C.", ("c",)))
        self.assertEqual(Clause.join("A.", clauses), "A. B. C.")
        self.assertEqual(Clause.join("A.", clauses, {"c": 1}), "A. C.")
        self.assertEqual(Clause.join("", clauses, {}), "")


class _TextShadows:
    """The shadows record's runtime written from its assembly text ALONE --
    what a reader holding only the file builds -- to score the text against
    pythontk's own model and march. Reads the map from its RGBA image."""

    @staticmethod
    def _terms(plane, C, L):
        """The model's terms as the text spells them, Y-up, in metres."""
        ms, height = plane["max_stretch"], plane["height"]
        ground = plane["ground"]
        d = (C[0] - L[0], C[2] - L[2])
        dist = math.hypot(*d)
        k_max = 1 + ms
        k_base = min(max((L[1] - ground) / max(1e-4, L[1] - C[1]), 0), k_max)
        k_top = min(
            max((L[1] - ground) / max(1e-4, L[1] - C[1] - height), 0),
            min(k_max, k_base + ms * height / max(dist, 1e-6)),
        )
        reach = max(0.0, dist * (k_top - k_base))
        return k_base, k_top, reach

    @classmethod
    def fade(cls, plane, C, L):
        """The fade, from the text: *plane*'s lengths and *C*, *L* in metres."""
        k_base, k_top, reach = cls._terms(plane, C, L)
        radius, ground = plane["radius"], plane["ground"]
        stretch = max(1, (reach + radius * (k_base + k_top)) / max(1e-4, 2 * radius))
        clamp = lambda x: min(max(x, 0.0), 1.0)  # noqa: E731
        return clamp(
            plane["intensity"]
            / max(0.001, stretch ** plane["falloff_power"])
            * clamp((L[1] - C[1]) / plane["source_ramp"])
            * clamp(1 - max(0, C[1] - ground) / max(0.001, plane["fade_height"]))
        )

    @classmethod
    def place(cls, plane, C, L):
        """``(centre (x, z), extent along u, extent along +X)`` of a Y-up quad."""
        ms, height, radius = plane["max_stretch"], plane["height"], plane["radius"]
        ground = plane["ground"]
        canvas = plane["canvas"]
        d = (C[0] - L[0], C[2] - L[2])
        dist = math.hypot(*d)
        u = (d[0] / dist, d[1] / dist) if dist >= 1e-6 else (0.0, 1.0)
        k_max = 1 + ms
        k_base = min(max((L[1] - ground) / max(1e-4, L[1] - C[1]), 0), k_max)
        k_top = min(
            max((L[1] - ground) / max(1e-4, L[1] - C[1] - height), 0),
            min(k_max, k_base + ms * height / max(dist, 1e-6)),
        )
        reach = max(0.0, dist * (k_top - k_base))
        width = 2 * radius * max(k_top, k_base)
        anchor = (L[0] + d[0] * k_base, L[2] + d[1] * k_base)
        far = max(reach + canvas[1] * radius * k_top, 0)
        x = (u[1], -u[0])  # +X of a quad along u, normal up: up x u
        cu = (canvas[0] * far + far) / 2
        cx = (canvas[2] + canvas[3]) * width / 2
        centre = (
            anchor[0] + u[0] * cu + x[0] * cx,
            anchor[1] + u[1] * cu + x[1] * cx,
        )
        return centre, far - canvas[0] * far, (canvas[3] - canvas[2]) * width

    @staticmethod
    def shadow(img, horizon, P, g, source=None, size=0.0, shines=None, angle=0.0):
        S, K = horizon["size"], horizon["spans"]
        a0, a1, b0, b1 = horizon["bounds"]
        hs = horizon["height_scale"]
        pa, pb = (a1 - a0) / S, (b1 - b0) / S
        p = min(pa, pb)
        aspect = max(pa, pb) / p
        px = img.astype(int)

        def w(x, y):
            return 256 * x + y

        def tile(k):
            return px[:S, k * S : (k + 1) * S]

        spans = [
            (w(tile(k)[..., 0], tile(k)[..., 1]), w(tile(k)[..., 2], tile(k)[..., 3]))
            for k in range(K)
        ]
        aux = tile(K)
        D = w(aux[..., 0], aux[..., 1]) / 64.0
        top, bottom = aux[..., 2] / 255.0 * hs, aux[..., 3] / 255.0 * hs
        pyramid = tile(K + 1)
        interior = None
        if tuple(pyramid[S // 2, S // 2]) == (83, 72, 1, 165):
            interior = [[0.0] * S for _ in range(S)]
            for j in range(S):
                for i in range(S):
                    interior[j][i] = (
                        pyramid[j // 2, S // 2 + i // 2][(i % 2) + 2 * (j % 2)] / 4.0
                    )
        if shines is not None:
            n = math.sqrt(sum(v * v for v in shines))
            Lv = [-v / n for v in shines]
            rho = angle / 2
        else:
            v = (source[0] - P[0], source[1] - g, source[2] - P[1])
            dist = math.sqrt(sum(c * c for c in v))
            Lv = [c / dist for c in v]
            rho = math.asin(min(1.0, size / 2 / dist))
        c = math.hypot(Lv[0], Lv[2])
        if Lv[1] <= 0 or c < 1e-9:
            return 0.0
        ox, oy = (P[0] - a0) / pa, (P[1] - b0) / pb
        dx, dy = Lv[0] / c / pa, Lv[2] / c / pb

        def entry_exit(o, dv):
            if abs(dv) < 1e-12:
                return (-math.inf, math.inf) if 0 <= o < S else (math.inf, -math.inf)
            return sorted(((0 - o) / dv, (S - o) / dv))

        (tx0, tx1), (ty0, ty1) = entry_exit(ox, dx), entry_exit(oy, dy)
        t, t_end = max(tx0, ty0, 0.0), min(tx1, ty1)

        def bilinear(field, tm):
            mx, my = ox + dx * tm - 0.5, oy + dy * tm - 0.5
            x0, y0 = math.floor(mx), math.floor(my)
            fx, fy = mx - x0, my - y0

            def at(x, y):
                return float(field[min(max(y, 0), S - 1)][min(max(x, 0), S - 1)])

            upper = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx
            lower = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx
            return upper * (1 - fy) + lower * fy

        rise, side, clear, hit = 0.0, 0.0, math.inf, False
        while t < t_end - 1e-12:
            nudge = 1e-9 * (1 + t)
            ix, iy = (
                math.floor(ox + dx * (t + nudge)),
                math.floor(oy + dy * (t + nudge)),
            )
            exits = [t_end]
            if abs(dx) > 1e-12:
                exits.append((ix + (dx > 0) - ox) / dx)
            if abs(dy) > 1e-12:
                exits.append((iy + (dy > 0) - oy) / dy)
            t1 = min(exits)
            if 0 <= ix < S and 0 <= iy < S:
                h0, h1, tm = t * Lv[1] / c, t1 * Lv[1] / c, (t + t1) / 2
                blocked, solid, gap = False, False, math.inf
                for lo_w, hi_w in spans:
                    if hi_w[iy, ix] == 0:
                        continue
                    solid = True
                    lo, hi = lo_w[iy, ix] / 65535 * hs, hi_w[iy, ix] / 65535 * hs
                    if h1 >= lo and h0 <= hi:
                        blocked = True
                    else:
                        gap = min(gap, max(lo - h1, h0 - hi))
                    if hi >= h0:
                        rise = max(rise, (hi - h0) * c * c / tm)
                if blocked:
                    hit = True
                    inner = (
                        math.inf
                        if interior is None
                        else max(bilinear(interior, tm) - aspect / 2, 0) * p * c / tm
                    )
                    side = max(side, inner)
                elif not hit:
                    if not solid:
                        b, tp = bilinear(bottom, tm), bilinear(top, tm)
                        vgap = h0 - tp if h0 > tp else (b - h1 if h1 < b else 0.0)
                        gap = math.hypot(
                            max(bilinear(D, tm) - aspect / 2, 0) * p * c, c * c * vgap
                        )
                    else:
                        gap *= c * c
                    clear = min(clear, gap / tm)
                if hit and min(rise, side) >= rho:
                    break
            t = t1
        if rho == 0:
            return 1.0 if hit else 0.0
        s = min(max(-min(rise, side) / rho if hit else clear / rho, -1.0), 1.0)
        return (math.acos(s) - s * math.sqrt(1 - s * s)) / math.pi


class TestShadowsText(unittest.TestCase):
    """The shadows record states its runtime in the file: the projection
    model, the placement and the horizon march -- each only where a plane
    uses it, exactly as pythontk computes it."""

    def test_a_file_is_told_only_what_its_planes_use(self):
        """v3 (2026-11-07): every plane is marched from its horizon map and
        fades, so the model, the fade and the march are always stated; the
        re-placement only where a plane follows its source, the silhouette
        only where a plane carries one. One statement for both containers:
        a GLB carries the record itself."""
        spec = SR.SHADOWS
        self.assertTrue(spec.glb_verbatim)
        still = {"planes": [{"follow_source": False}]}
        following = {"planes": [{"follow_source": True}]}
        drawn = {"planes": [{"follow_source": False, "silhouette": {"draw": True}}]}
        text = spec.assembly_for(still)
        for always in ("k_base", "Every plane fades", "disc(x)"):
            self.assertIn(always, text)
        self.assertNotIn("Turn the quad", text)
        self.assertNotIn("'silhouette' is", text)
        self.assertIn("Turn the quad", spec.assembly_for(following))
        self.assertIn("'silhouette' is", spec.assembly_for(drawn))
        for retired in ("'type'", "texture_index", "source_node", "A projected plane"):
            self.assertNotIn(retired, spec.assembly_for())

    def test_the_text_quotes_the_codes_constants(self):
        from pythontk.geo_utils import shadow_horizon as sh
        from pythontk.geo_utils.shadow_projection import ShadowProjection

        text = SR.SHADOWS.assembly_for()
        self.assertEqual(ShadowProjection.FAR_FACTOR, 1e6)
        self.assertIn("s*1e6*max(height, 2*radius)", text)
        self.assertEqual(ShadowProjection.EPS, 1e-6)
        self.assertIn("dist < 1e-6", text)
        self.assertEqual(ShadowProjection.OVERHEAD_BEARING, (0.0, 1.0))
        self.assertIn("((0, 1) when dist < 1e-6)", text)
        self.assertIn(str(sh.INTERIOR_MARKER), text)
        self.assertIn(f"w(R, G)/{int(sh.DIST_SCALE)} in p", text)
        self.assertIn(f"byte/{int(sh.INTERIOR_SCALE)} in p", text)
        self.assertIn(f"w/{int(sh.MAX16)}*height_scale", text)

    def test_the_placement_reads_the_planes_own_canvas(self):
        """``canvas`` is the plane's (it decides the placement every reader
        makes), never the silhouette extension's: a reader without the
        extension placed every plane by the default and clipped the
        penumbra a soft source spreads past it. Added: 2026-11-07."""
        import dataclasses

        from pythontk.geo_utils.shadow_record import ShadowPlane, ShadowPlaneSilhouette

        text = SR.SHADOWS.assembly_for()
        self.assertIn("from its 'canvas'", text)
        self.assertNotIn("silhouette.canvas", text)
        names = lambda spec: {f.name for f in dataclasses.fields(spec)}  # noqa: E731
        self.assertIn("canvas", names(ShadowPlane))
        self.assertNotIn("canvas", names(ShadowPlaneSilhouette))

    def test_the_text_rebuilds_the_fade(self):
        """The fade a reader builds from the text alone is
        ``ShadowProjection.fade`` -- a point source and a sun, a metre and a
        centimetre rig's ramp, every term in play.
        Added: 2026-11-07."""
        from pythontk.geo_utils.shadow_projection import ShadowProjection

        rng = __import__("random").Random(7)
        for i in range(300):
            ramp = (1.0, 0.01)[i % 2]
            C = (rng.uniform(-2, 2), rng.uniform(0, 1.2), rng.uniform(-2, 2))
            sun = i % 3 == 0
            if sun:
                s = (rng.uniform(-1, 1), rng.uniform(-1, -0.05), rng.uniform(-1, 1))
            else:
                L = (rng.uniform(-5, 5), C[1] + rng.uniform(-0.05, 5), rng.uniform(-5, 5))
            plane = {
                "max_stretch": rng.uniform(2, 8),
                "height": rng.uniform(0.3, 2),
                "radius": rng.uniform(0.2, 1),
                "ground": 0.0,
                "intensity": rng.uniform(0.2, 1),
                "falloff_power": rng.uniform(0, 3),
                "fade_height": rng.uniform(0.5, 3),
                "source_ramp": ramp,
            }
            if sun:
                L = ShadowProjection.far_point(
                    C, s, max(plane["height"], 2 * plane["radius"])
                )
            want = ShadowProjection.fade(
                C,
                L,
                plane["ground"],
                plane["radius"],
                plane["height"],
                max_stretch=plane["max_stretch"],
                intensity=plane["intensity"],
                falloff_power=plane["falloff_power"],
                fade_height=plane["fade_height"],
                source_ramp=plane["source_ramp"],
            )
            self.assertAlmostEqual(_TextShadows.fade(plane, C, L), want, places=12)

    def test_the_text_rebuilds_the_projection_and_its_placement(self):
        from pythontk.geo_utils.shadow_projection import ShadowProjection

        rng = __import__("random").Random(11)
        for _ in range(100):
            C = (rng.uniform(-2, 2), rng.uniform(0, 0.5), rng.uniform(-2, 2))
            L = (rng.uniform(-5, 5), rng.uniform(0.2, 6), rng.uniform(-5, 5))
            plane = {
                "max_stretch": rng.uniform(2, 8),
                "height": rng.uniform(0.3, 2),
                "radius": rng.uniform(0.2, 1),
                "ground": 0.0,
                "canvas": [rng.uniform(-1.2, -0.2), rng.uniform(0.5, 1.5), -0.5, 0.6],
            }
            model = ShadowProjection.model(
                C,
                L,
                0.0,
                plane["radius"],
                plane["height"],
                max_stretch=plane["max_stretch"],
            )
            want = model.placement(plane["canvas"])
            got = _TextShadows.place(plane, C, L)
            for a, b in zip((*want[0], want[1], want[2]), (*got[0], got[1], got[2])):
                self.assertAlmostEqual(a, b, places=9)

    def test_the_text_rebuilds_the_horizon_march(self):
        """Every fragment as HeightFieldMap.alpha has it -- the edge of the
        footprint included, where a ray leaving it reads the outermost
        texels (a read there once jumped to the next texel: three of these
        fragments, 0.10 off)."""
        import numpy as np

        from pythontk.geo_utils.shadow_horizon import (
            HeightFieldMap,
            ShadowHorizon,
            ShadowHorizonConformance,
        )

        box = ShadowHorizonConformance._box

        table = [
            box(-0.5, 0.5, 0.70, 0.75, -0.3, 0.3),
            box(-0.45, 0.45, 0.25, 0.28, -0.02, 0.02),
        ]
        table += [
            box(x, x + 0.05, 0.0, 0.70, z, z + 0.05)
            for x in (-0.45, 0.4)
            for z in (-0.25, 0.2)
        ]
        rng = np.random.default_rng(3)
        cases = [
            # the edge: rays leaving the footprint through its low edges
            (
                32,
                3,
                dict(shines=(0.601898, -0.48793, 0.632174), angle=0.1),
                [(0.264806, 0.658219), (0.200901, 0.611152), (0.859856, 0.478384)],
            ),
            (16, 2, dict(source=(2.0, 1.2, -1.5), size=0.3), None),
            (16, 2, dict(shines=(-0.3, -0.5, 0.8), angle=0.08), None),
            (16, 2, dict(source=(-1.0, 3.0, 2.0), size=0.0), None),
        ]
        for size, spans, light, points in cases:
            hmap = ShadowHorizon.bake(table, ground=0.0, up=1, size=size, spans=spans)
            img = hmap.to_rgba()
            horizon = {
                "size": hmap.size,
                "spans": hmap.spans,
                "bounds": list(hmap.bounds),
                "height_scale": hmap.height_scale,
            }
            reference = HeightFieldMap.from_rgba(
                img,
                size=hmap.size,
                spans=hmap.spans,
                bounds=hmap.bounds,
                ground=hmap.ground,
                up=hmap.up,
                height_scale=hmap.height_scale,
            )
            a0, a1, b0, b1 = hmap.bounds
            if points is None:
                points = list(
                    zip(
                        rng.uniform(a0 - 0.3, a1 + 0.3, 60),
                        rng.uniform(b0 - 0.3, b1 + 0.3, 60),
                    )
                )
            frags = np.array([(x, 0.01, z) for x, z in points])
            if "shines" in light:
                want = reference.alpha(
                    frags, direction=light["shines"], source_angle=light["angle"]
                )
            else:
                want = reference.alpha(
                    frags, light=light["source"], source_size=light["size"]
                )
            for (x, z), expected in zip(points, want):
                got = _TextShadows.shadow(img, horizon, (x, z), 0.0, **light)
                with self.subTest(size=size, light=light, at=(x, z)):
                    self.assertAlmostEqual(got, float(expected), places=4)


class TestPublishedContract(SceneRecordsCase):
    """``SceneRecords.json_schema``: the contract published as one JSON Schema
    document, which a reader in any language validates against or generates
    its types from. Added: 2026-11-04"""

    def test_every_deliverable_record_is_a_property_by_its_key(self):
        doc = SR.json_schema()
        for spec in SR.deliverable():
            with self.subTest(record=spec.key):
                prop = doc["properties"][spec.key]
                self.assertEqual(prop["x-version"], spec.version)
                shape = SR.shape(spec)
                if shape is not None:
                    self.assertEqual(prop["$ref"], f"#/$defs/{shape.__name__}")

    def test_every_glb_block_is_a_property_of_the_glb_extras(self):
        extras = SR.json_schema()["$defs"]["GlbExtras"]["properties"]
        for spec in SR.glb_records():
            with self.subTest(record=spec.key):
                self.assertEqual(extras[spec.key]["x-version"], spec.version)

    def test_every_reference_resolves(self):
        doc = SR.json_schema()

        def refs(node):
            if isinstance(node, dict):
                if "$ref" in node:
                    yield node["$ref"]
                for value in node.values():
                    yield from refs(value)
            elif isinstance(node, list):
                for value in node:
                    yield from refs(value)

        found = list(refs(doc))
        self.assertTrue(found)
        for ref in found:
            self.assertTrue(ref.startswith("#/$defs/"), ref)
            self.assertIn(ref[len("#/$defs/"):], doc["$defs"])

    def test_it_states_the_frame_and_the_transport(self):
        doc = SR.json_schema()
        self.assertEqual(doc["x-frame"]["length"], "metre")
        self.assertEqual(doc["x-frame"]["up"], "+Y")
        self.assertEqual(doc["x-frame"]["time"], "second")
        self.assertEqual(doc["x-transport"]["marker"], "curveProxy")
        json.dumps(doc)

    def test_the_handoff_is_a_glb_extras_property(self):
        extras = SR.json_schema()["$defs"]["GlbExtras"]["properties"]
        self.assertEqual(extras[SR.HANDOFF.key]["x-version"], SR.HANDOFF.version)

    def test_each_record_says_whether_its_values_are_in_the_frame(self):
        props = SR.json_schema()["properties"]
        for spec in SR.deliverable():
            with self.subTest(record=spec.key):
                self.assertIs(props[spec.key]["x-in-frame"], spec.in_frame)

    def test_an_extension_names_only_readers_of_its_record(self):
        def readers(node):
            if isinstance(node, dict):
                for prop in (node.get("properties") or {}).values():
                    if isinstance(prop, dict) and "x-readers" in prop:
                        yield from prop["x-readers"]
                for value in node.values():
                    yield from readers(value)
            elif isinstance(node, list):
                for value in node:
                    yield from readers(value)

        for spec in SR.deliverable():
            for shape in filter(None, [SR.shape(spec)]):
                with self.subTest(record=spec.key, shape=shape.__name__):
                    self.assertLessEqual(set(readers(shape.json_schema())), set(spec.consumers))

    def _contract_of(self, shapes):
        """SR.json_schema over records whose shapes are *shapes*, by key."""
        specs = [
            RecordSpec(key, Scope.DELIVERABLE, 1, owner="Test", description=key)
            for key in shapes
        ]
        real_shape = SR.shape.__func__

        def shape(cls, item):
            spec = cls.resolve(item) if not isinstance(item, RecordSpec) else item
            return shapes.get(spec.key) or real_shape(cls, item)

        with mock.patch.object(SR, "deliverable", classmethod(lambda cls: specs)), \
                mock.patch.object(SR, "glb_records", classmethod(lambda cls: [])), \
                mock.patch.object(SR, "shape", classmethod(shape)):
            return SR.json_schema()

    def test_a_shape_that_is_one_records_whole_and_nested_in_anothers_is_defined_once(self):
        from dataclasses import dataclass
        from typing import Optional

        from pythontk.core_utils.schema_spec import SchemaSpec

        @dataclass
        class Inner(SchemaSpec):
            """Inner."""

            TYPED = True
            n: Optional[int] = SchemaSpec.spec_field(help="A number.")

        @dataclass
        class Outer(SchemaSpec):
            """Outer."""

            TYPED = True
            inner: Optional[Inner] = SchemaSpec.spec_field(help="The inner.")

        doc = self._contract_of({"a": Inner, "b": Outer})
        self.assertEqual(doc["properties"]["a"]["$ref"], "#/$defs/Inner")
        self.assertIn("Inner", doc["$defs"])

    def test_two_shapes_defining_one_name_differently_are_refused(self):
        from dataclasses import dataclass
        from typing import Optional

        from pythontk.core_utils.schema_spec import SchemaSpec

        @dataclass
        class One(SchemaSpec):
            """One."""

            TYPED = True
            n: Optional[int] = SchemaSpec.spec_field(help="A number.")

        @dataclass
        class Two(SchemaSpec):
            """Two."""

            TYPED = True
            s: Optional[str] = SchemaSpec.spec_field(help="A word.")

        Two.__name__ = "One"
        with self.assertRaises(ValueError):
            self._contract_of({"a": One, "b": Two})

    def test_the_producers_and_the_converter_mark_proxies_by_the_declared_transport(self):
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        self.assertEqual(MeshConvert.CURVE_PROXY_MARKER, SR.TRANSPORT.marker)


class TestGlbMedia(unittest.TestCase):
    """``SceneRecords.GLB_MEDIA``: how each file a record a GLB carries
    itself names rides that GLB, declared once (2026-11-07) -- the bufferView
    named by the file's name, its extras stating its bytes."""

    def test_a_file_rides_as_the_buffer_view_named_by_its_file_name(self):
        """Joined by file name, lower-cased on both sides; a name with a folder
        joins by its file name, as a record names a file."""
        media = SR.GLB_MEDIA
        gltf = {
            "bufferViews": [{"name": "other.png"}, {"name": "Room_Lightmap.EXR"}, {}]
        }
        self.assertEqual(media.view(gltf, "room_lightmap.exr"), 1)
        self.assertEqual(media.view(gltf, "Room_Lightmap.EXR"), 1)
        self.assertEqual(media.view(gltf, "maps/Room_Lightmap.exr"), 1)
        for name in ("missing.exr", "", None):
            with self.subTest(name=name):
                self.assertIsNone(media.view(gltf, name))
        self.assertIsNone(media.view({}, "other.png"))
        self.assertIsNone(media.view(None, "other.png"))

    def test_a_held_name_joins_by_its_file_name_too(self):
        """Both sides are joined the same way: a view whose name carries a
        folder still holds that file. Added: 2026-11-07"""
        gltf = {"bufferViews": [{"name": "maps\\Room_Lightmap.exr"}]}
        self.assertEqual(SR.GLB_MEDIA.view(gltf, "room_lightmap.EXR"), 0)

    def test_two_views_holding_one_name_hold_neither(self):
        """A tie is never guessed, as a node join leaves one unresolved: two
        views whose names fold alike hold neither file, and ``views`` says
        which they are. Added: 2026-11-07"""
        gltf = {"bufferViews": [{"name": "a.exr"}, {"name": "dir/A.EXR"}, {"name": "b.exr"}]}
        self.assertIsNone(SR.GLB_MEDIA.view(gltf, "A.exr"))
        self.assertEqual(SR.GLB_MEDIA.views(gltf, "A.exr"), [0, 1])
        self.assertEqual(SR.GLB_MEDIA.view(gltf, "B.EXR"), 2)

    def test_the_fold_is_lower_case_alike_in_both_languages(self):
        """``str.lower``, which JavaScript's ``toLowerCase`` spells alike --
        not ``casefold``, which folds what JavaScript does not (the German
        sharp s to ``ss``). Added: 2026-11-07"""
        self.assertEqual(SR.GLB_MEDIA.fold("dir/Stra\u00dfe.EXR"), "stra\u00dfe.exr")
        gltf = {"bufferViews": [{"name": "strasse.exr"}]}
        self.assertIsNone(SR.GLB_MEDIA.view(gltf, "Stra\u00dfe.exr"))

    def test_its_extras_state_the_bytes(self):
        """``mimeType`` always; ``scale`` only for an 8-bit web encoding of
        an HDR map, whose decoded linear texels times it are the map."""
        media = SR.GLB_MEDIA
        self.assertEqual(media.extras(media.radiance), {"mimeType": "image/vnd.radiance"})
        self.assertEqual(
            media.extras("image/png", scale=3.25), {"mimeType": "image/png", "scale": 3.25}
        )
        self.assertEqual(media.hdr_web, ("image/png", "image/webp", "image/ktx2"))
        with self.assertRaises(ValueError):
            media.extras("image/vnd.radiance", scale=2.0)

    def test_a_scale_is_a_finite_positive_number(self):
        """glTF JSON holds no NaN or infinity, and no map is scaled by zero or
        less: each is refused before it is written. Added: 2026-11-07"""
        for scale in (float("nan"), float("inf"), 0.0, -1.0):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                SR.GLB_MEDIA.extras("image/png", scale=scale)

    def test_its_text_is_the_rule_a_reader_holds(self):
        text = SR.GLB_MEDIA.text
        for part in (
            "bufferView",
            "`name`",
            "lower-cased (str.lower, toLowerCase)",
            "hold neither",
            "the audio record's sounds ride none yet",
            "`extras.mimeType`",
            "`extras.scale`",
            "`image/vnd.radiance`",
            "`image/png`",
            "`image/ktx2`",
            "An FBX embeds each file itself",
        ):
            with self.subTest(part=part):
                self.assertIn(part, text)

    def test_the_published_schema_states_it(self):
        import dataclasses

        doc = SR.json_schema()
        self.assertEqual(doc["x-media"], dataclasses.asdict(SR.GLB_MEDIA))
        self.assertIn(SR.GLB_MEDIA.text, doc["description"])



class TestRetiredUntil022(unittest.TestCase):
    """The nine names v0.20.0 noticed for removal in 0.18.0 (2026-10-24),
    whose window closes 2026-11-23, after 0.21.0 ships: each still works and
    warns until then (CODE_STANDARD section 5). Delete with the aliases at
    0.22.0 (``pythontk/__init__.py``, "Held until 0.22.0").
    Restored: 2026-11-08"""

    def test_the_scene_records_names_warn_and_answer(self):
        import pythontk as ptk

        with self.assertWarns(DeprecationWarning):
            old = ptk.WebProjection("crate", version=3)
        self.assertIsInstance(old, ptk.GlbBlock)
        self.assertEqual(old.version, 3)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(old.key, "crate")
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(ptk.GlbBlock(version=2).key, "")
        with self.assertWarns(DeprecationWarning):
            block = SR.SHOTS.web
        self.assertEqual(block.version, SR.SHOTS.version)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(block.key, SR.SHOTS.key)
        with self.assertWarns(DeprecationWarning):
            self.assertIsNone(SR.MATERIALS.web)
        with self.assertWarns(DeprecationWarning):
            shape = SR.web_shape(SR.ARTICULATION)
        self.assertIs(shape, SR.shape(SR.ARTICULATION))

    def test_the_articulation_block_names_warn_and_validate_a_block(self):
        import pythontk as ptk

        block = {
            "version": 1,
            "rigs": [
                {
                    "name": "door",
                    "joints": [
                        {
                            "parent": None,
                            "t": [0.0, 0.0, 0.0],
                            "q": [0.0, 0.0, 0.0, 1.0],
                            "rotate_order": "xyz",
                            "channels": [],
                            "node": 3,
                        }
                    ],
                    "grab": [{"node": 4, "joint": 0}],
                }
            ],
        }
        self.assertTrue(ptk.ArticulationWeb.validate(block).ok)
        with self.assertWarns(DeprecationWarning):
            ptk.ArticulationWeb(version=1, rigs=[])
        with self.assertWarns(DeprecationWarning):
            ptk.ArticulationWebRig(name="door", joints=[], grab=[])
        with self.assertWarns(DeprecationWarning):
            ptk.ArticulationWebJoint(
                parent=None, t=(0, 0, 0), q=(0, 0, 0, 1), rotate_order="xyz", node=3
            )
        with self.assertWarns(DeprecationWarning):
            ptk.ArticulationWebGrab(node=4, joint=0)

    def test_the_mesh_convert_block_attributes_warn_and_answer(self):
        """A GLB carries each record as it stands: its version is the
        record's, and no field of it binds by node index."""
        import pythontk as ptk

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(ptk.MeshConvert.ANIMATION_WEB_VERSION, SR.SHOTS.version)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(ptk.MeshConvert.SHADOW_WEB_NODE_FIELDS, ())
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(ptk.MeshConvert.ARTICULATION_WEB_NODE_FIELDS, ())

    def test_the_region_mask_manifest_states_its_version_under_schema(self):
        import pythontk as ptk

        manifest = ptk.RegionMaskManifest.vertex_color([{"name": "sign", "slot": 0}])
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(manifest.schema, manifest.version)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(ptk.RegionMaskManifest.SCHEMA_VERSION, ptk.RegionMaskManifest.VERSION)
        self.assertEqual(manifest.to_dict()["groups"], [{"name": "sign", "slot": 0, "default": 1.0}])




class TestNamedNodes(unittest.TestCase):
    """``named_nodes``: the scene names a record holds -- a task regrouping or
    renaming one would leave the record naming nothing (2026-11-10: the Scene
    Exporter's LOD generation skips them)."""

    def test_each_record_names_its_entries_nodes(self):
        lightmaps = {"version": 1, "objects": [{"name": "ns:floor"}, {"name": "wall"}, {"name": "wall"}]}
        self.assertEqual(SR.LIGHTMAPS.named_nodes(lightmaps), ["ns:floor", "wall"])
        rigs = {"rigs": [{"joints": [{"name": "hinge"}, {"name": "lid"}]}]}
        self.assertEqual(SR.ARTICULATION.named_nodes(rigs), ["hinge", "lid"])
        self.assertEqual(SR.LIGHTMAPS.named_nodes({"objects": "bogus"}), [])
        self.assertEqual(SR.SHOTS.named_nodes({"shots": []}), [], "a record naming no node")

    def test_a_store_dump_answers_every_name_and_its_leaf(self):
        dump = {
            "data_export": {
                "lightmaps": {"objects": [{"name": "ns:floor"}]},
                "shadows": {"planes": [{"name": "|grp|plane"}]},
                "unknown_key": {"objects": [{"name": "x"}]},
            },
            "data_internal": {},
        }
        self.assertEqual(
            SR.named_nodes(dump),
            {"ns:floor", "floor", "|grp|plane", "plane"},
        )
        dump["data_export"]["lods"] = {"groups": [{"name": "rock"}]}
        self.assertIn("rock", SR.named_nodes(dump))
        self.assertNotIn("rock", SR.named_nodes(dump, skip=(SR.LODS.key,)))
        skipped = SR.named_nodes(dump, skip=(k for k in (SR.LODS.key, SR.SHADOWS.key)))
        self.assertNotIn("rock", skipped, "a generator is read once, not spent on the first record")
        self.assertNotIn("plane", skipped)


if __name__ == "__main__":
    unittest.main(verbosity=2)
