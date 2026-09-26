# !/usr/bin/python
# coding=utf-8
"""SceneRecords -- the one declaration of scene metadata, and the machinery
that stores, assembles and commits it.

Runs against an in-memory store: the DCC stores are thin and pinned by their
own suites; everything about records, ordering, contexts and the commit is
decided here, once.
"""

import json
import os
import pathlib
import unittest
from typing import Any, Dict
from unittest import mock

from pythontk.core_utils.scene_records import (
    ExportContext,
    ExportSnapshot,
    Merge,
    RecordSpec,
    RecordTransfer,
    SceneRecords as SR,
    SceneStoreBase,
    Scope,
    TransferContext,
)


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
        """``emissive_groups`` is a manifest on the carrier AND a registry in
        the scene; the bare key means the shipped one."""
        self.assertIs(SR.by_key("emissive_groups"), SR.EMISSIVE_GROUPS)
        self.assertIs(SR.by_key("emissive_groups", Scope.PRIVATE), SR.EMISSIVE_REGISTRY)
        self.assertIsNone(SR.by_key("nope"))
        with self.assertRaises(KeyError):
            SR.resolve("nope")

    def test_ordered_puts_a_reader_after_what_it_reads(self):
        order = [s.key for s in SR.ordered([SR.VISIBILITY, SR.AUDIO, SR.SHOTS])]
        self.assertLess(order.index("shot_metadata"), order.index("audio_manifest"))
        self.assertLess(order.index("shot_metadata"), order.index("visibility_tracks"))

    def test_ordered_follows_dependencies_against_declaration_order(self):
        """The registry's own records are DECLARED in dependency order, so they
        cannot prove ``after`` is honoured; two unregistered specs handed in
        reader-first can."""
        d = RecordSpec("d", Scope.DELIVERABLE, 1, "t", "d")
        r = RecordSpec("r", Scope.DELIVERABLE, 1, "t", "d", after=("d",))
        self.assertEqual(SR.ordered([r, d]), [d, r])

    def test_declared_takes_come_from_the_clips_then_the_legacy_list(self):
        """Every file since 0.11.0: each clip carries its range.  Before it,
        the clips carried none and the take list rode ``fbx_takes``."""
        legacy = [{"name": "old", "start": 1, "end": 2}]
        clips = {"shots": [{"clip": "A", "start": 1, "end": 2, "objects": []}]}
        read = {SR.SHOTS.key: clips, SR.FBX_TAKES.key: legacy}.get
        self.assertEqual(SR.declared_takes(read), [{"name": "A", "start": 1, "end": 2}])
        # An older file: clips without ranges, or no shot record at all.
        read = {SR.SHOTS.key: {"shots": [{"clip": "A"}]}, SR.FBX_TAKES.key: legacy}.get
        self.assertEqual(SR.declared_takes(read), legacy)
        self.assertEqual(
            SR.declared_takes({SR.FBX_TAKES.key: legacy + ["x"]}.get), legacy
        )
        self.assertEqual(SR.declared_takes({}.get), [])

    def test_ordered_does_not_wait_for_a_record_that_is_not_in_the_set(self):
        self.assertEqual(
            [s.key for s in SR.ordered([SR.VISIBILITY])], ["visibility_tracks"]
        )

    def test_ordered_refuses_a_cycle(self):
        a = RecordSpec("a", Scope.DELIVERABLE, 1, "t", "d", after=("b",))
        b = RecordSpec("b", Scope.DELIVERABLE, 1, "t", "d", after=("a",))
        with self.assertRaises(ValueError):
            SR.ordered([a, b])

    def test_check_producers_rejects_private_keys_and_the_handoff(self):
        self.assertEqual(
            SR.check_producers({SR.SHOTS: None, "lightmap_metadata": None}),
            [SR.SHOTS, SR.LIGHTMAPS],
        )
        with self.assertRaises(ValueError):
            SR.check_producers({SR.SHOT_STORE: None})
        with self.assertRaises(ValueError):
            SR.check_producers({SR.HANDOFF: None})
        with self.assertRaises(KeyError):
            SR.check_producers({"shot_metadat": None})

    def test_a_retired_record_is_gone_by_its_removal_version(self):
        """A deprecated record names the release it stops being written in;
        reaching it without deleting the record fails here, the same way
        ``Deprecation`` gates a retired name -- a removal nobody enforces
        never happens."""
        import pythontk

        def parse(v):
            return tuple(int(p) for p in str(v).split(".")[:3])

        for spec in SR.all():
            if spec.remove_in:
                self.assertIsNotNone(spec.deprecated_by, spec.key)
                self.assertLess(
                    parse(pythontk.__version__),
                    parse(spec.remove_in),
                    f"{spec.key} was to be removed in {spec.remove_in}",
                )

    def test_describe_is_one_row_per_record(self):
        rows = SR.describe()
        self.assertEqual(len(rows), len(SR.all()))
        self.assertEqual(rows[0]["key"], SR.SHOTS.key)
        self.assertIn("description", rows[0])
        legacy = next(r for r in rows if r["key"] == SR.FBX_TAKES.key)
        self.assertEqual(legacy["remove_in"], SR.FBX_TAKES.remove_in)
        emissive = next(r for r in rows if r["key"] == SR.EMISSIVE_GROUPS.key)
        self.assertEqual(emissive["version_key"], "schema")


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

    def test_a_record_versions_under_its_declared_key(self):
        """The emissive manifest is a document that versions itself under
        ``schema`` (what its Unity reader gates on): the declaration stamps
        and checks THAT key, so the record never carries two versions."""
        from pythontk.core_utils.engines.textures.region_masks import (
            RegionMaskManifest,
        )

        self.assertEqual(SR.EMISSIVE_GROUPS.version, RegionMaskManifest.SCHEMA_VERSION)
        rec = SR.EMISSIVE_GROUPS.make({"groups": []})
        self.assertEqual(rec.payload, {"schema": 1, "groups": []})
        newer = json.dumps({"schema": 2, "groups": []})
        with self.assertLogs("pythontk.core_utils.scene_records", level="WARNING"):
            self.assertIsNone(SR.EMISSIVE_GROUPS.decode(newer))

    def test_legacy_shape_is_stored_as_given(self):
        rec = SR.FBX_TAKES.make([{"name": "a", "start": 1, "end": 2}])
        self.assertEqual(json.loads(rec.text), [{"name": "a", "start": 1, "end": 2}])

    def test_decode_tolerates_absent_cleared_and_corrupt(self):
        self.assertIsNone(SR.SHOTS.decode(None))
        self.assertEqual(SR.SHOTS.decode("", default=[]), [])
        self.assertEqual(SR.SHOTS.decode("{not json", default={}), {})

    def test_decode_refuses_a_newer_version(self):
        newer = json.dumps({"version": SR.SHOTS.version + 1, "shots": []})
        with self.assertLogs("pythontk.core_utils.scene_records", level="WARNING"):
            self.assertIsNone(SR.SHOTS.decode(newer))
        older = json.dumps({"version": 0, "shots": []})
        self.assertEqual(SR.SHOTS.decode(older)["shots"], [])

    def test_non_native_values_encode_as_their_string(self):
        rec = SR.LIGHTMAPS.make({"dir": pathlib.PurePosixPath("a/b")})
        self.assertEqual(json.loads(rec.text)["dir"], "a/b")


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
        self.assertIsNone(SR.EMISSIVE_GROUPS.load(DictStore))
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
                "data_export": {"lightmap_metadata": {"objects": [], "version": 1}},
            },
        )
        self.assertEqual(
            DictStore.dump(decode=False)["data_export"]["lightmap_metadata"],
            json.dumps({"version": 1, "objects": []}),
        )
        self.assertEqual(
            json.loads(DictStore.format_dump())["data_export"]["lightmap_metadata"][
                "version"
            ],
            1,
        )

    def test_format_dump_is_empty_for_an_empty_scene(self):
        self.assertEqual(DictStore.format_dump(), "")

    def test_channels_are_the_non_empty_strings_only(self):
        DictStore.data[Scope.DELIVERABLE] = {"a": "x", "b": "", "w": 0.5}
        self.assertEqual(DictStore.channels(Scope.DELIVERABLE), {"a": "x"})


# -- the handoff --------------------------------------------------------------------


class TestHandoff(SceneRecordsCase):
    def test_describes_exactly_the_string_channels_present(self):
        block = SR.handoff_block(
            {
                "lightmap_metadata": "{}",
                "emissiveGroup_Neon": 0.5,
                "future": "{}",
                "handoff": "{}",
            }
        )
        self.assertEqual(
            sorted(block["reads"]),
            ["data_export.future", "data_export.lightmap_metadata"],
        )
        self.assertEqual(block["reads"]["data_export.future"], "tool-authored channel")
        self.assertEqual(
            block["reads"]["data_export.lightmap_metadata"], SR.LIGHTMAPS.description
        )
        self.assertEqual(block["version"], SR.HANDOFF.version)
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        self.assertEqual(
            block["rendering"],
            MeshConvert.RENDERING_POLICY,
            "the FBX publishes the viewer's own policy, from the one constant",
        )

    def test_nothing_to_describe_means_no_block(self):
        self.assertEqual(SR.handoff_block({}), {})
        self.assertEqual(SR.handoff_block({"handoff": "{}"}), {})
        self.assertEqual(SR.handoff_block({"emissiveGroup_Neon": 1.0}), {})

    def test_source_drops_empty_entries(self):
        block = SR.handoff_block(
            ["lightmap_metadata"], source={"application": "maya", "scene": None}
        )
        self.assertEqual(block["source"], {"application": "maya"})
        self.assertIsNone(SR.handoff_block(["lightmap_metadata"])["source"])


# -- the context + snapshot ---------------------------------------------------------


def _shots(ctx: ExportContext):
    meta = {"fps": 24.0, "shots": [{"clip": "A", "start": 1, "end": 10}]}
    if ctx.clip_mode:
        meta["clip_mode"] = ctx.clip_mode
    return SR.SHOTS.make(meta)


def _visibility(ctx: ExportContext):
    shots = ctx.record(SR.SHOTS, DictStore) or {}
    takes = SR.declared_takes(lambda key: ctx.record(key, DictStore))
    span = list(ctx.clip_span) if ctx.clip_span else [0, 100]
    return SR.VISIBILITY.make(
        {
            "fps": shots.get("fps"),
            "tracks": [1],
            "clip_span": {
                "*": span,
                **{t["name"]: [t["start"], t["end"]] for t in takes},
            },
        }
    )


def _older_scene_takes():
    """The legacy take list a scene published before 0.11.0 still holds."""
    SR.FBX_TAKES.save(DictStore, [{"name": "A", "start": 1, "end": 10}])


def _nothing(ctx):
    return None


def _boom(ctx):
    raise RuntimeError("subsystem down")


PRODUCERS = {SR.VISIBILITY: _visibility, SR.SHOTS: _shots, SR.LIGHTMAPS: _nothing}


class TestSnapshot(SceneRecordsCase):
    def test_assemble_orders_by_dependency_and_hands_prior_records_over(self):
        ctx = ExportContext(clip_mode="both", clip_span=(5.0, 50.0))
        snap = ExportSnapshot.assemble(PRODUCERS, ctx)
        # Dependency order, declaration order breaking ties: the two records
        # with no dependencies first (shots, then lightmaps), then the reader
        # of shots.
        self.assertEqual(
            list(snap.produced),
            ["shot_metadata", "lightmap_metadata", "visibility_tracks"],
        )
        vis = snap.record(SR.VISIBILITY)
        self.assertEqual(
            vis["fps"], 24.0, "read from the record produced in this assembly"
        )
        self.assertEqual(
            vis["clip_span"]["*"], [5.0, 50.0], "the exporter's decision is an INPUT"
        )
        self.assertEqual(vis["clip_span"]["A"], [1, 10], "the clips' own ranges")
        self.assertEqual(snap.record(SR.SHOTS)["clip_mode"], "both")
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
            sorted(handoff["reads"]),
            ["data_export.shot_metadata", "data_export.visibility_tracks"],
        )
        self.assertEqual(handoff["source"], {"application": "test"})
        self.assertIsNone(written["lightmap_metadata"])
        self.assertIn("handoff", written)

    def test_the_handoff_publishes_the_exports_lighting_choices(self):
        """What the export decided about the lighting recipe (a baked-reflection
        level) is an input like the clip mode: the stamped handoff carries it,
        merged the way the GLB's envelope merges it. Added: 2026-09-21"""
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        rendering = {"lightmappedMaterials": {"envMapIntensity": 1.0}}
        ExportSnapshot.assemble(PRODUCERS, ExportContext(rendering=rendering)).commit(
            DictStore
        )
        self.assertEqual(
            SR.HANDOFF.load(DictStore)["rendering"],
            MeshConvert.rendering_policy(rendering),
        )

    def test_the_same_context_produces_the_same_records_twice(self):
        """Re-running an assembly is harmless by construction -- the defect the
        old patch-after-producers pipeline had (three exports logging the right
        clip origin and shipping the old one)."""
        ctx = ExportContext(clip_mode="full", clip_span=(1.0, 9.0))
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
        self.assertEqual(list(snap.record(SR.VISIBILITY)["clip_span"]), ["*"])

    def test_a_producers_notes_are_logged_and_kept_for_one_assembly(self):
        """What a producer leaves out of a deliverable is said, never silent:
        logged on every route, and kept on the context for an exporter's own
        report -- one assembly's worth, like its records."""

        def noting(ctx):
            ctx.note("2 shot(s) left out")
            return None

        ctx = ExportContext()
        with self.assertLogs("pythontk.core_utils.scene_records", "WARNING") as log:
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
        with self.assertLogs("pythontk.core_utils.scene_records", level="WARNING"):
            snap = ExportSnapshot.assemble(table)
        self.assertEqual(snap.failed, {"lightmap_metadata"})
        self.assertIn("shot_metadata", snap.records)

    def test_a_private_record_is_refused_rather_than_clearing_its_deliverable_twin(
        self,
    ):
        """The emissive registry (private) shares its key with the emissive
        manifest (deliverable); a snapshot commits the deliverable carrier, so
        a private record handed to it would clear the manifest."""
        SR.EMISSIVE_GROUPS.save(DictStore, {"groups": [1]})
        with self.assertRaises(ValueError):
            ExportSnapshot.publish(DictStore, {SR.EMISSIVE_REGISTRY: None})
        with self.assertRaises(ValueError):
            ExportSnapshot.assemble({SR.EMISSIVE_REGISTRY: _nothing})
        self.assertEqual(SR.EMISSIVE_GROUPS.load(DictStore)["groups"], [1])

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
        with self.assertLogs("pythontk.core_utils.scene_records", level="WARNING"):
            written = snap.commit(LockedStore)
        self.assertIn("lightmap_metadata", snap.failed)
        self.assertNotIn("lightmap_metadata", written)
        self.assertTrue(SR.SHOTS.is_present(DictStore))
        self.assertTrue(SR.HANDOFF.is_present(DictStore))
        self.assertIn("lightmap_metadata (FAILED", snap.summary())
        self.assertNotIn("lightmap_metadata (1", snap.summary())

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
        with self.assertLogs("pythontk.core_utils.scene_records", level="WARNING"):
            snap = ExportSnapshot.assemble(table)
        self.assertEqual(snap.failed, {"lightmap_metadata"})
        snap.commit(DictStore)
        self.assertEqual(
            SR.LIGHTMAPS.load(DictStore)["objects"], [1], "left as stored, not cleared"
        )
        self.assertIn("shot_metadata", snap.records)
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

    def test_a_legacy_record_is_cleared_with_its_successor(self):
        """``fbx_takes`` is no longer written; a scene published before 0.11.0
        still holds one, and the next shots publish clears it -- whether the
        producer returns the shots or nothing (an emptied store must not leave
        the old takes riding into the next export)."""
        _older_scene_takes()
        written = ExportSnapshot.assemble({SR.SHOTS: _shots}).commit(DictStore)
        self.assertTrue(SR.SHOTS.is_present(DictStore))
        self.assertFalse(SR.FBX_TAKES.is_present(DictStore))
        self.assertIsNone(written["fbx_takes"])
        _older_scene_takes()
        ExportSnapshot.assemble({SR.SHOTS: _nothing}).commit(DictStore)
        self.assertFalse(SR.SHOTS.is_present(DictStore))
        self.assertFalse(SR.FBX_TAKES.is_present(DictStore))

    def test_a_failed_successor_leaves_its_legacy_record_alone(self):
        ExportSnapshot.assemble({SR.SHOTS: _shots}).commit(DictStore)
        _older_scene_takes()
        with self.assertLogs("pythontk.core_utils.scene_records", level="WARNING"):
            ExportSnapshot.assemble({SR.SHOTS: _boom}).commit(DictStore)
        self.assertTrue(SR.SHOTS.is_present(DictStore))
        self.assertTrue(SR.FBX_TAKES.is_present(DictStore))

    def test_a_reader_sees_a_record_cleared_in_this_assembly_as_absent(self):
        """The shots producer returns nothing (an emptied store): a reader later
        in the SAME assembly must not fall back to the stored shot record or
        take list the commit is about to clear."""
        ExportSnapshot.assemble({SR.SHOTS: _shots}).commit(DictStore)
        _older_scene_takes()
        seen = {}

        def reader(ctx):
            seen["shots"] = ctx.record(SR.SHOTS, DictStore)
            seen["takes"] = ctx.record(SR.FBX_TAKES, DictStore)

        ExportSnapshot.assemble({SR.SHOTS: _nothing, SR.VISIBILITY: reader})
        self.assertEqual(seen, {"shots": None, "takes": None})

    def test_the_envelope_version_leads_and_a_producer_value_wins(self):
        self.assertEqual(
            list(SR.LIGHTMAPS.make({"objects": []}).payload), ["version", "objects"]
        )
        self.assertEqual(SR.SHADOWS.make({"version": 1}).payload["version"], 1)

    def test_only_narrows_a_pipeline_run(self):
        snap = ExportSnapshot.assemble(PRODUCERS, only=["shot_metadata"])
        self.assertEqual(list(snap.produced), ["shot_metadata"])

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
            sorted(SR.HANDOFF.load(DictStore)["reads"]),
            ["data_export.lightmap_metadata", "data_export.shadow_metadata"],
        )
        ExportSnapshot.publish(DictStore, {SR.LIGHTMAPS: None})
        self.assertIsNone(SR.LIGHTMAPS.load(DictStore))
        self.assertEqual(
            list(SR.HANDOFF.load(DictStore)["reads"]), ["data_export.shadow_metadata"]
        )

    def test_the_shot_record_declares_the_clip_mode_under_the_readers_key(self):
        from pythontk.core_utils.engines.shots.shot_model import ShotStore
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        self.assertEqual(ShotStore.CLIP_MODE_KEY, MeshConvert.SHOT_CLIP_MODE_KEY)

    def test_channels_and_summary(self):
        snap = ExportSnapshot.assemble(PRODUCERS)
        self.assertEqual(
            sorted(snap.channels()), ["shot_metadata", "visibility_tracks"]
        )
        self.assertIn("shot_metadata (1 entry)", snap.summary())
        self.assertNotIn("lightmap_metadata", snap.summary(), "nothing shipped")

    def test_context_record_falls_back_to_the_store_at_authoring_time(self):
        SR.SHOTS.save(DictStore, {"fps": 30.0, "shots": []})
        ctx = ExportContext(mode=ExportContext.AUTHORING)
        self.assertEqual(ctx.record(SR.SHOTS, DictStore)["fps"], 30.0)
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
                "lightmap_metadata": {"version": 1, "objects": [{"name": "a"}]},
                "handoff": {"version": 1},
            },
        )
        merge = RecordTransfer.between(DictStore, other)
        self.assertTrue(merge.is_empty)
        self.assertEqual(merge.summary(), [])
        self.assertEqual(
            [s.key for s in merge.rederive], ["lightmap_metadata", "handoff"]
        )

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
                "emissive_groups": {"schema": 1, "groups": {"glow": {"slot": 0}}},
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
            self._other(deliverable={"lightmap_metadata": {"objects": []}})
        )
        self.assertIsNone(DictStore.read(Scope.DELIVERABLE, "lightmap_metadata"))
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
                    "emissive_groups": {"schema": 1, "groups": {"glow": {"slot": 0}}}
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

    def test_an_emissive_group_whose_slot_is_taken_moves_and_says_so(self):
        SR.EMISSIVE_REGISTRY.save(
            DictStore,
            {
                "schema": 1,
                "groups": {"rim": {"slot": 0, "default": 1.0}},
                "retired_slots": [1],
            },
        )
        _, ctx = self._merge(
            self._other(
                private={
                    "emissive_groups": {
                        "schema": 1,
                        "groups": {
                            "glow": {"slot": 0, "default": 0.5},
                            "rim": {"slot": 1, "default": 0.2},
                        },
                    }
                }
            )
        )
        groups = SR.EMISSIVE_REGISTRY.load(DictStore)["groups"]
        self.assertEqual(groups["rim"], {"slot": 0, "default": 1.0})  # this scene's
        self.assertEqual(groups["glow"]["slot"], 2)  # 0 used, 1 retired
        self.assertEqual(groups["glow"]["default"], 0.5)
        self.assertTrue(
            any("moves from slot 0 to 2" in n for n in ctx.notes), ctx.notes
        )
        self.assertTrue(any("'rim'" in n for n in ctx.notes), ctx.notes)

    def test_an_emissive_group_with_no_usable_slot_still_merges(self):
        """A group whose slot is missing, null or not a number is re-slotted
        (the merge already tolerates a bad slot) -- it must not trip the
        ordering that runs first and take the whole record down with it."""
        SR.EMISSIVE_REGISTRY.save(DictStore, {"schema": 1, "groups": {}})
        _, ctx = self._merge(
            self._other(
                private={
                    "emissive_groups": {
                        "schema": 1,
                        "groups": {
                            "late": {"slot": None, "default": 0.5},
                            "early": {"slot": 3, "default": 0.2},
                            "odd": {"slot": "x", "default": 0.1},
                        },
                    }
                }
            )
        )
        groups = SR.EMISSIVE_REGISTRY.load(DictStore)["groups"]
        self.assertEqual(sorted(groups), ["early", "late", "odd"], ctx.notes)
        self.assertEqual(groups["early"]["slot"], 3)
        self.assertEqual(sorted(groups[g]["slot"] for g in ("late", "odd")), [0, 1])
        self.assertFalse(any("not merged" in n for n in ctx.notes), ctx.notes)

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
        private registry and the deliverable manifest share one key."""
        other = self._other(
            private={
                "emissive_groups": {"schema": 1, "groups": {"g": {"slot": 0}}},
                "key_stash": {
                    "schema": 1,
                    "clips": [
                        {"clip_id": 1, "label": "a", "objects": ["a"], "curves": []}
                    ],
                },
            },
            deliverable={"emissive_groups": {"schema": 1, "groups": {}}},
        )
        merge = RecordTransfer.between(DictStore, other)
        payloads = merge.payloads(TransferContext(rename={"a": "a1"}.get))
        self.assertEqual(payloads[SR.EMISSIVE_REGISTRY]["groups"], {"g": {"slot": 0}})
        self.assertEqual(payloads[SR.EMISSIVE_GROUPS]["groups"], {})
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
        self.assertEqual(list(out["records"]), ["emissive_groups"])
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
        SR.EMISSIVE_REGISTRY.save(
            DictStore, {"schema": 1, "groups": {"rim": {"slot": 0}}}
        )
        manifest = {
            "records": {
                "emissive_groups": {"schema": 1, "groups": {"glow": {"slot": 0}}}
            }
        }
        ctx = RecordTransfer.receive(manifest, DictStore, TransferContext())
        groups = SR.EMISSIVE_REGISTRY.load(DictStore)["groups"]
        self.assertEqual(groups["glow"]["slot"], 1)
        self.assertTrue(any("slot 0 to 1" in n for n in ctx.notes))

    def test_a_bad_record_is_noted_and_never_costs_the_rest(self):
        class Broken:
            @staticmethod
            def transfer_in(payload, ctx):
                raise RuntimeError("nope")

        manifest = {
            "shots": {"shots": []},
            "records": {"emissive_groups": {"schema": 1, "groups": {"g": {"slot": 0}}}},
        }
        ctx = RecordTransfer.receive(
            manifest, DictStore, TransferContext(), {"shot_store": Broken}
        )
        self.assertIn("g", SR.EMISSIVE_REGISTRY.load(DictStore)["groups"])
        self.assertTrue(any("nope" in n for n in ctx.notes))

    def test_the_generic_section_is_the_manifests(self):
        """The engine names the section without importing the manifest class
        (the lighter module imports first); the two spellings are one."""
        from pythontk.core_utils.handoff_manifest import HandoffManifest

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
            Scope.DELIVERABLE: {"lightmap_metadata": json.dumps({"objects": [1]})},
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
            "a": ("pythontk.core_utils.scene_records", "RecordTransfer"),
            "b": ("no_such_module_anywhere", "Owner"),
        }
        with mock.patch.object(_CarrierStore, "OWNERS", table):
            self.assertEqual(_CarrierStore.owners(), {"a": RecordTransfer})

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
        self.assertEqual(
            self._abs(self.OLD, landed), self._abs(module, "sourceimages/lm")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
