#!/usr/bin/python
# coding=utf-8
"""``HandoffRecord``: how a deliverable states itself, declared once for every carrier.

The block rides every carrier under the same keys -- ``version``,
``instructions``, ``assembly``, ``rendering`` and ``extras`` -- and names the
approved scene's parts alike on each, each record by its one key, so an FBX
and the GLB built from it line up row by row, the address alone differing. A
carrier's writer says only which records its file holds
(:meth:`HandoffRecord.block`).
"""

import copy
import json
import unittest

from pythontk import HandoffBlock, HandoffCarrier, HandoffRecord, SceneRecords
from pythontk.core_utils.engines.scene_export.scene_records import Clause

LIGHTING = ["environment", "key_light", "lightmapped_materials", "renderer"]
#: The lighting rows of a file without lightmaps: no lightmapped material.
UNLIT = ["environment", "key_light", "renderer"]


def _parts(rows):
    return [row["part"] for row in rows]


class TestIdentity(unittest.TestCase):
    def test_the_block_is_its_records(self):
        """The key it rides under and its schema version are the handoff
        record's, on every carrier."""
        self.assertEqual(HandoffRecord.KEY, SceneRecords.HANDOFF.key)
        self.assertEqual(HandoffRecord.KEY, "handoff")
        self.assertEqual(HandoffRecord.VERSION, SceneRecords.HANDOFF.version)
        self.assertEqual(HandoffRecord.VERSION, 3)

    def test_each_carrier_is_declared_once(self):
        self.assertEqual(set(HandoffRecord.CARRIERS), {"fbx", "glb"})
        for name, carrier in HandoffRecord.CARRIERS.items():
            with self.subTest(carrier=name):
                self.assertIsInstance(carrier, HandoffCarrier)
                self.assertIs(HandoffRecord.carrier(name), carrier)
        with self.assertRaises(ValueError):
            HandoffRecord.carrier("usdz")
        self.assertFalse(HandoffRecord.CARRIERS["fbx"].blocks, "the records themselves")
        self.assertTrue(HandoffRecord.CARRIERS["glb"].blocks, "each record's block")

    def test_a_record_has_one_key_on_every_carrier(self):
        """One name per record across both files (2026-10-20): each carrier
        addresses a record by its key after its own prefix -- the FBX's
        ``data_export`` node, the GLB's root ``extras``."""
        fbx, glb = HandoffRecord.CARRIERS["fbx"], HandoffRecord.CARRIERS["glb"]
        for spec in SceneRecords.deliverable():
            with self.subTest(record=spec.key):
                self.assertEqual(fbx.address(spec.key), f"data_export.{spec.key}")
                self.assertEqual(glb.address(spec.key), f"extras.{spec.key}")
        self.assertEqual(glb.address(SceneRecords.LIGHTMAPS.key), "extras.lightmaps")
        self.assertEqual(fbx.address(SceneRecords.SHOTS.key), "data_export.animation")


class TestBlock(unittest.TestCase):
    def test_the_fundamentals_at_the_top_and_the_rest_under_extras(self):
        """Version 3: what a reader must read at the top, ``extras`` holding
        only where the file came from -- empty entries dropped, and no
        ``extras`` at all for no source."""
        for carrier in HandoffRecord.CARRIERS:
            with self.subTest(carrier=carrier):
                block = HandoffRecord.block(
                    carrier, [], source={"application": "maya", "scene": None}
                )
                self.assertEqual(
                    set(block),
                    {"version", "instructions", "assembly", "rendering", "extras"},
                )
                self.assertEqual(block["version"], HandoffRecord.VERSION)
                self.assertEqual(block["extras"], {"source": {"application": "maya"}})
                self.assertEqual(
                    block["instructions"], HandoffRecord.instructions(carrier)
                )
                # The recipe as a file holding no record uses it.
                self.assertEqual(
                    block["rendering"], HandoffRecord.rendering_policy(None, [])
                )
                self.assertNotIn("extras", HandoffRecord.block(carrier, []))

    def test_the_block_carries_the_exports_lighting_choices(self):
        chosen = {"lightmappedMaterials": {"envMapIntensity": 0.5}}
        held = [SceneRecords.LIGHTMAPS.key]
        block = HandoffRecord.block("glb", held, rendering=chosen)
        self.assertEqual(
            block["rendering"], HandoffRecord.rendering_policy(chosen, held)
        )
        self.assertEqual(
            block["rendering"]["lightmappedMaterials"]["envMapIntensity"], 0.5
        )

    def test_a_block_whose_records_are_not_known_yet_keeps_them_whole(self):
        """``held`` ``None``: the records are not known yet -- a GLB's handoff is
        written before its passes add their blocks. The block names no record
        row, states the lighting rows whole and keeps the export's whole
        recipe, so the step that knows the blocks
        (``MeshConvert._finish_glb_handoff``) gates them: a lightmap-only
        choice gated out first could never be read back. Added: 2026-10-20"""
        chosen = {"lightmappedMaterials": {"envMapIntensity": 0.5}}
        for carrier in HandoffRecord.CARRIERS:
            with self.subTest(carrier=carrier):
                first = HandoffRecord.block(
                    carrier, None, {"application": "maya"}, chosen
                )
                self.assertEqual(
                    first["rendering"], HandoffRecord.rendering_policy(chosen)
                )
                # Each of the carrier's own rows whole: its spec with every
                # clause, and only the keys a row is written with.
                own = [
                    {
                        "part": row["part"],
                        "data": list(row["data"]),
                        "spec": Clause.join(row["spec"], row.get("clauses", ())),
                    }
                    for row in HandoffRecord.carrier(carrier).own
                ]
                self.assertEqual(
                    first["assembly"], own + HandoffRecord.lighting(carrier, None)
                )
                self.assertEqual(_parts(first["assembly"])[len(own) :], LIGHTING)
        first = HandoffRecord.block("glb", None, {"application": "maya"}, chosen)
        lit = {SceneRecords.LIGHTMAPS.key: {"carrier": "occlusion", "materials": {}}}
        final = HandoffRecord.rendering_policy(
            HandoffRecord.overrides(first["rendering"]) or None, lit
        )
        self.assertEqual(final["lightmappedMaterials"]["envMapIntensity"], 0.5)

    def test_the_block_is_json(self):
        block = HandoffRecord.block(
            "fbx", [SceneRecords.LIGHTMAPS.key], source={"application": "maya"}
        )
        self.assertEqual(json.loads(json.dumps(block)), block)

    def test_a_block_whose_records_are_not_known_yet_is_json(self):
        """REGRESSION (2026-10-24): ``held`` ``None`` returned the carrier's
        own rows raw, so the GLB's ``materials`` row kept its ``clauses`` --
        ``Clause`` objects -- and ``json.dumps`` of the block raised
        TypeError. Each own row is filled as a known file's is, every clause
        joined into its ``spec``."""
        for carrier in HandoffRecord.CARRIERS:
            with self.subTest(carrier=carrier):
                block = HandoffRecord.block(carrier, None)
                self.assertEqual(json.loads(json.dumps(block)), block)
                for row in block["assembly"]:
                    self.assertEqual(set(row), {"part", "data", "spec"})

    def test_a_source_is_written_as_its_declared_shape(self):
        """``extras.source`` states only what ``HandoffSource`` declares, each
        value a string, empties left out -- and no source at all without the
        application it is stated by: whatever a writer hands over, the block
        is one its shape admits. Added: 2026-10-20"""
        odd = {
            "application": "maya",
            "version": 2025,
            "scene": None,
            "user": "me",
            "build": "",
        }
        for carrier in HandoffRecord.CARRIERS:
            with self.subTest(carrier=carrier):
                block = HandoffRecord.block(carrier, [], odd)
                self.assertEqual(
                    block["extras"],
                    {"source": {"application": "maya", "version": "2025"}},
                )
                self.assertEqual(SceneRecords.problems(SceneRecords.HANDOFF, block), [])
                nameless = HandoffRecord.block(carrier, [], {"scene": "room.ma"})
                self.assertNotIn("extras", nameless)
                self.assertEqual(
                    SceneRecords.problems(SceneRecords.HANDOFF, nameless), []
                )

    def test_every_block_written_is_its_declared_shape(self):
        """What :meth:`HandoffRecord.block` writes is the handoff record's
        declared shape (``HandoffBlock``) on every carrier -- holding records
        or none, with a source or without -- the shape every reader holds it
        to (``SceneRecords.problems``). Added: 2026-10-20"""
        self.assertIs(SceneRecords.shape(SceneRecords.HANDOFF), HandoffBlock)
        held = [SceneRecords.LIGHTMAPS.key, SceneRecords.SHOTS.key]
        source = {"application": "maya", "version": "2025", "scene": "room.ma"}
        for carrier in HandoffRecord.CARRIERS:
            for records, given in (([], None), (held, source)):
                with self.subTest(carrier=carrier, held=bool(records)):
                    block = HandoffRecord.block(carrier, records, given)
                    self.assertEqual(
                        SceneRecords.problems(SceneRecords.HANDOFF, block), []
                    )


class TestInstructions(unittest.TestCase):
    """The reading contract, composed per carrier from the shared text."""

    def test_composed_from_the_carriers_opening_the_shared_rules_and_the_check(self):
        for name, carrier in HandoffRecord.CARRIERS.items():
            with self.subTest(carrier=name):
                text = HandoffRecord.instructions(name)
                self.assertTrue(text.startswith(carrier.intro))
                self.assertTrue(
                    text.endswith("Check 'version' against the schema you expect.")
                )
                self.assertIn(carrier.addresses, text)
                self.assertIn("under 'extras'", text)
                self.assertIn("'source'", text)
                if carrier.rules:
                    self.assertIn(carrier.rules, text)
                # Each part's meaning is its row's: the text names no record,
                # so one added later is described with no edit here.
                self.assertNotIn("'reads'", text)
                self.assertNotIn("lightmapInfo", text)
                for spec in SceneRecords.deliverable():
                    if spec is not SceneRecords.HANDOFF:
                        self.assertNotIn(carrier.address(spec.key), text)

    def test_the_fbx_says_which_carrier_wins_and_how_names_join(self):
        """An assembly ships a referenced module's carrier beside the scene's
        own, as that module was last exported alone; both readers here prefer
        the scene's own node per record and join file names without case."""
        text = HandoffRecord.instructions("fbx")
        self.assertIn("named exactly 'data_export'", text)
        self.assertIn("counts only where the scene's own node has none", text)
        self.assertIn(
            f"a '{HandoffRecord.KEY}' there describes that module, not this file", text
        )
        self.assertIn("FILE NAME", text)
        self.assertIn("without regard to case", text)
        self.assertIn("where the file holds no copy", text)

    def test_the_fbx_says_the_files_its_records_name_are_embedded(self):
        """The load-bearing sentence. An FBX embeds what its MATERIALS
        reference, and a lighting-only bake leaves its map unwired so the
        authored material survives: the maps a record names were the one part
        of a deliverable that did not travel with it (measured on a delivered
        room, 23.0 MB of a 23.5 MB file was material textures and none of it
        the lightmap), and the block said they were "supplied separately".
        Since 2026-10-05 every deliverable embeds each file a record names, so
        the text says a name joins to the copy INSIDE the file -- and, as it
        rides every carrier-bearing write while only a delivering writer
        embeds, what it joins to otherwise."""
        text = HandoffRecord.instructions("fbx")
        self.assertIn("a delivered file embeds each as media of its own", text)
        self.assertIn("join token", text)
        # What each named file is -- a lightmap wired to no material, the
        # probe's image media no material uses -- is its own row's, so a file
        # holding neither is told of neither.
        self.assertNotIn("probe", text)
        self.assertIn("wired to no material", SceneRecords.LIGHTMAPS.assembly)
        probe = HandoffRecord.lighting(
            "fbx", {SceneRecords.LIGHTMAPS.key: {"probe": {"map": "p.exr"}}}
        )
        self.assertIn("media no material uses", " ".join(r["spec"] for r in probe))
        self.assertNotIn("supplied separately", text)
        self.assertNotIn("NOT embedded", text)
        self.assertNotIn("named by 'asset'", text, "no asset key is carried")
        # The bake commits a non-1.0 intensity INTO the texels -- both readers
        # apply the map as stored -- so a reader told it is a multiplier
        # "restoring the bake's range" would apply it twice.
        row = SceneRecords.LIGHTMAPS.assembly
        for field in ("'scaleOffset'", "'uvIndex'", "applied as stored"):
            with self.subTest(field=field):
                self.assertIn(field, row)
        self.assertNotIn("restoring the bake's range", text + row)

    def test_the_glb_names_its_own_block(self):
        text = HandoffRecord.instructions("glb")
        self.assertIn(f"'extras.{HandoffRecord.KEY}'", text)
        self.assertNotIn("data_export", text)


class TestAssembly(unittest.TestCase):
    """The approved scene part by part, in each carrier's addresses: the
    file's own content, a row per record it holds -- under the record's key,
    in declaration order -- then the lighting."""

    def test_both_carriers_name_a_record_by_its_key(self):
        """One name per record, whatever holds it, so the two deliverables of
        one export line up row by row; the GLB states its materials in glTF's
        own terms, a row of its own."""
        held = [spec.key for spec in SceneRecords.glb_records()]
        self.assertEqual(
            held,
            [
                SceneRecords.SHOTS.key,
                SceneRecords.LIGHTMAPS.key,
                SceneRecords.SHADOWS.key,
                SceneRecords.ARTICULATION.key,
            ],
        )
        fbx = HandoffRecord.assembly("fbx", reversed(held))
        glb = HandoffRecord.assembly("glb", held)
        self.assertEqual(_parts(fbx), ["geometry", *held, *LIGHTING])
        self.assertEqual(
            _parts(glb), ["geometry", SceneRecords.MATERIALS.key, *held, *LIGHTING]
        )

    def test_the_fbx_states_the_record_and_the_glb_its_block(self):
        """A carrier's row says what IT holds, at the same key: the FBX the
        record itself (:meth:`RecordSpec.assembly_for`), the GLB the block
        bound to the file (:meth:`GlbBlock.assembly_for`) -- whole, when only
        the keys are known."""
        held = [SceneRecords.LIGHTMAPS.key, SceneRecords.SHOTS.key]
        fbx = {row["part"]: row for row in HandoffRecord.assembly("fbx", held)}
        glb = {row["part"]: row for row in HandoffRecord.assembly("glb", held)}
        for spec in (SceneRecords.LIGHTMAPS, SceneRecords.SHOTS):
            with self.subTest(record=spec.key):
                self.assertEqual(fbx[spec.key]["data"], [f"data_export.{spec.key}"])
                self.assertEqual(fbx[spec.key]["spec"], spec.assembly_for())
                self.assertEqual(glb[spec.key]["data"], [f"extras.{spec.key}"])
                self.assertEqual(glb[spec.key]["spec"], spec.glb.assembly_for())

    def test_a_records_row_states_what_its_payload_uses(self):
        """A shadows record of still projected planes is told nothing of the
        placement or the horizon march; one with a horizon plane is told the
        march -- in each carrier's own addresses."""
        shadows = SceneRecords.SHADOWS.key
        still = {"planes": [{"type": "projected", "follow_source": False}]}
        marched = {"planes": [{"type": "horizon", "follow_source": False}]}
        for carrier in HandoffRecord.CARRIERS:
            with self.subTest(carrier=carrier):
                rows = {
                    r["part"]: r["spec"]
                    for r in HandoffRecord.assembly(carrier, {shadows: still})
                }
                self.assertNotIn("k_base", rows[shadows])
                self.assertNotIn("disc(x)", rows[shadows])
                text = {
                    r["part"]: r["spec"]
                    for r in HandoffRecord.assembly(
                        carrier, {shadows: json.dumps(marched)}
                    )
                }[shadows]
                self.assertIn("disc(x)", text)
                self.assertIn(
                    "'horizon.texture_index'"
                    if carrier == "glb"
                    else "'horizon.texture' names",
                    text,
                )

    def test_a_glb_names_only_the_records_it_carries_a_block_for(self):
        """A record a GLB carries no block for -- the audio events, the
        materials it states as glTF's own -- is no part of it; the FBX, which
        holds the records themselves, names each."""
        audio, materials = SceneRecords.AUDIO, SceneRecords.MATERIALS
        self.assertIsNone(audio.glb)
        self.assertIsNone(materials.glb)
        held = [audio.key, materials.key]
        self.assertEqual(
            _parts(HandoffRecord.assembly("glb", held)),
            ["geometry", materials.key, *UNLIT],
        )
        self.assertEqual(
            _parts(HandoffRecord.assembly("fbx", held)),
            ["geometry", materials.key, audio.key, *UNLIT],
        )

    def test_the_fbx_block_names_exactly_the_records_its_carrier_holds(self):
        """A reader holding the file and nothing else rebuilds the scene from
        the block: every record the carrier holds -- another tool's included,
        listed rather than lost -- and none it lacks; a keyable float or the
        handoff itself is no record. Each row says what its data means, and
        every rendering address points at a field the block publishes."""
        channels = {
            SceneRecords.LIGHTMAPS.key: "{}",
            SceneRecords.VISIBILITY.key: "{}",
            "future": "{}",
            "emissiveGroup_Neon": 0.5,
            HandoffRecord.KEY: "{}",
        }
        block = HandoffRecord.block("fbx", channels)
        rows = {row["part"]: row for row in block["assembly"]}
        addressed = sorted(
            {
                row["data"][0]
                for row in block["assembly"]
                if row["data"] and row["data"][0].startswith("data_export.")
            }
        )
        self.assertEqual(
            addressed,
            [
                "data_export.future",
                "data_export.lightmaps",
                "data_export.render_effects",
            ],
        )
        # its row as ITS record calls for: one with no probe is told of none
        self.assertEqual(
            rows[SceneRecords.LIGHTMAPS.key]["spec"],
            SceneRecords.LIGHTMAPS.assembly_for({}),
        )
        self.assertEqual(rows["future"]["spec"], HandoffRecord.UNDECLARED)
        self.assertNotIn(SceneRecords.SHOTS.key, rows, "never a record the file lacks")
        for row in block["assembly"]:
            with self.subTest(part=row["part"]):
                self.assertTrue(row["spec"].strip())
                for address in row["data"]:
                    if address.startswith("rendering."):
                        self.assertIn(address.split(".", 1)[1], block["rendering"])

    def test_the_files_own_content_has_no_address(self):
        """The GLB's materials are complete glTF materials, its own content;
        the FBX's, without a record, are only its own to go by."""
        glb = HandoffRecord.assembly("glb", [])
        self.assertEqual(_parts(glb), ["geometry", SceneRecords.MATERIALS.key, *UNLIT])
        self.assertEqual([row["data"] for row in glb[:2]], [[], []])
        fbx = HandoffRecord.assembly("fbx", [])
        self.assertEqual(_parts(fbx), ["geometry", *UNLIT])
        self.assertEqual(fbx[0]["data"], [])

    def test_a_record_no_declaration_knows_is_listed_not_lost(self):
        for carrier, prefix in (("fbx", "data_export."), ("glb", "extras.")):
            with self.subTest(carrier=carrier):
                rows = {
                    row["part"]: row
                    for row in HandoffRecord.assembly(carrier, ["future"])
                }
                self.assertEqual(rows["future"]["data"], [prefix + "future"])
                self.assertEqual(rows["future"]["spec"], HandoffRecord.UNDECLARED)

    def test_the_block_never_describes_itself(self):
        rows = HandoffRecord.assembly("fbx", [HandoffRecord.KEY])
        self.assertNotIn(HandoffRecord.KEY, _parts(rows))

    def test_a_caller_editing_rows_edits_no_declaration(self):
        rows = HandoffRecord.assembly("glb", [])
        rows[0]["spec"] = "edited"
        rows[0]["data"].append("edited")
        self.assertNotEqual(HandoffRecord.CARRIERS["glb"].own[0]["spec"], "edited")
        self.assertEqual(HandoffRecord.CARRIERS["glb"].own[0]["data"], [])


class TestLighting(unittest.TestCase):
    """The lighting rows: what each part of the policy means, the carrier's
    addresses filled in."""

    def test_every_row_points_into_rendering(self):
        policy = HandoffRecord.rendering_policy()
        for carrier in HandoffRecord.CARRIERS:
            rows = HandoffRecord.lighting(carrier)
            with self.subTest(carrier=carrier):
                self.assertEqual(_parts(rows), LIGHTING)
                for row in rows:
                    (address,) = row["data"]
                    self.assertTrue(address.startswith("rendering."), address)
                    self.assertIn(address.split(".", 1)[1], policy)

    def test_the_lightmaps_address_joins_the_rows_that_read_them(self):
        """The environment reads the probe there and a lightmapped material
        its map; the key light and the renderer read none."""
        rows = {
            r["part"]: r["data"]
            for r in HandoffRecord.lighting("fbx", [SceneRecords.LIGHTMAPS.key])
        }
        self.assertEqual(
            rows["environment"], ["rendering.environment", "data_export.lightmaps"]
        )
        self.assertEqual(
            rows["lightmapped_materials"],
            ["rendering.lightmappedMaterials", "data_export.lightmaps"],
        )
        self.assertEqual(rows["key_light"], ["rendering.keyLight"])
        self.assertEqual(rows["renderer"], ["rendering.renderer"])

    def test_an_assembly_reads_the_lightmaps_only_where_the_file_holds_them(self):
        lightmaps = SceneRecords.LIGHTMAPS.key
        for carrier, prefix in (("fbx", "data_export."), ("glb", "extras.")):
            with self.subTest(carrier=carrier):
                held = {
                    r["part"]: r["data"]
                    for r in HandoffRecord.assembly(carrier, [lightmaps])
                }
                self.assertEqual(
                    held["environment"],
                    ["rendering.environment", prefix + lightmaps],
                )
                bare = {
                    r["part"]: r["data"] for r in HandoffRecord.assembly(carrier, [])
                }
                self.assertEqual(bare["environment"], ["rendering.environment"])

    def test_a_file_is_told_only_what_its_records_use(self):
        """No lightmaps: no lightmapped-material rule, no probe, no key-light
        switch. Lightmaps without a probe: the rule, still no probe. A probe:
        its text, and on a GLB its highlight copies only where it has them."""
        lightmaps = SceneRecords.LIGHTMAPS.key
        cases = {
            "nothing": {},
            "lightmaps": {lightmaps: {"objects": []}},
            "probe": {lightmaps: {"probe": {"map": "p"}}},
            "highlights": {lightmaps: {"probe": {"highlights": [{"map": "h"}]}}},
        }
        for carrier in HandoffRecord.CARRIERS:
            seen = {
                name: {
                    r["part"]: r["spec"] for r in HandoffRecord.lighting(carrier, held)
                }
                for name, held in cases.items()
            }
            with self.subTest(carrier=carrier):
                self.assertEqual(list(seen["nothing"]), UNLIT)
                self.assertNotIn("'disabledWhen'", seen["nothing"]["key_light"])
                self.assertNotIn("equirectangular", seen["nothing"]["environment"])
                self.assertEqual(list(seen["lightmaps"]), LIGHTING)
                self.assertIn("'disabledWhen'", seen["lightmaps"]["key_light"])
                self.assertNotIn("equirectangular", seen["lightmaps"]["environment"])
                self.assertIn("equirectangular", seen["probe"]["environment"])
                self.assertNotIn("COPIES", seen["probe"]["environment"])
                if carrier == "glb":
                    self.assertIn("COPIES", seen["highlights"]["environment"])
                for spec in seen["highlights"].values():
                    self.assertNotIn("{", spec.replace("{min, max}", ""))

    def test_each_carrier_says_where_it_holds_the_probe(self):
        """An FBX reader sent after a GLB address found no probe and dropped
        the environment (2026-10-14): each carrier names its own."""
        fbx = {r["part"]: r["spec"] for r in HandoffRecord.lighting("fbx")}
        glb = {r["part"]: r["spec"] for r in HandoffRecord.lighting("glb")}
        self.assertIn("data_export.lightmaps.probe", fbx["environment"])
        self.assertIn("unit_scale", fbx["environment"])
        self.assertNotIn("extras.", " ".join(fbx.values()))
        self.assertNotIn("bufferView", " ".join(fbx.values()))
        self.assertIn("extras.lightmaps.probe", glb["environment"])
        self.assertIn("'bufferView'", glb["environment"])
        self.assertIn("'highlights'", glb["environment"])
        self.assertIn(
            "that material's 'intensity' in extras.lightmaps",
            glb["lightmapped_materials"],
        )
        self.assertIn("1, the map applied as stored", fbx["lightmapped_materials"])

    def test_the_probe_level_is_stated_for_both_lightmap_conventions(self):
        """``probeIntensity`` is three.js'; a renderer compositing the texel
        as albedo x texel keeps the balance at 1 -- a bare number sends a
        Unity build to a third of the approved reflections -- but lights the
        scene pi times brighter than approved, so the text says how it
        matches: its lightmaps at 1/pi and the probe at ``probeIntensity``
        (a cold reader found the absolute level unstated, 2026-10-20)."""
        for carrier in HandoffRecord.CARRIERS:
            spec = {r["part"]: r["spec"] for r in HandoffRecord.lighting(carrier)}[
                "environment"
            ]
            with self.subTest(carrier=carrier):
                self.assertIn("'probeIntensity' (1/pi)", spec)
                self.assertIn("with its lightmaps scaled by 1/pi", spec)
                self.assertIn("pi times brighter than approved", spec)

    def test_the_formulas_are_the_rows_not_the_values(self):
        """The policy is values alone; what each means is its row's."""
        spec = {r["part"]: r["spec"] for r in HandoffRecord.lighting("glb")}
        baked = spec["lightmapped_materials"]
        self.assertIn("(dot(N, L) * 0.5 + 0.5) / (dot(Ng, L) * 0.5 + 0.5)", baked)
        self.assertIn("clamp(luma(E_bake) / luma(E_env), 0, 1)", baked)
        self.assertIn("iblIrradiance", baked)
        self.assertIn("'disabledWhen'", spec["key_light"])


class TestCarried(unittest.TestCase):
    def test_the_records_a_carrier_holds_are_its_text_channels(self):
        """A carrier also holds keyable float attributes the records beside
        them describe; the handoff itself is never one of its records."""
        self.assertEqual(
            HandoffRecord.carried(
                {
                    SceneRecords.SHOTS.key: "{}",
                    SceneRecords.LIGHTMAPS.key: "{}",
                    "emissiveGroup_Neon": 0.5,
                    "cleared": "",
                    HandoffRecord.KEY: "{}",
                }
            ),
            [SceneRecords.SHOTS.key, SceneRecords.LIGHTMAPS.key],
        )
        self.assertEqual(
            HandoffRecord.carried(["b", "a", HandoffRecord.KEY]), ["a", "b"]
        )

    def test_the_payloads_are_each_records_content(self):
        """A carrier's JSON text is its payload -- an object or a list -- and a
        GLB's block is one already; a text that is not JSON, or holds a JSON
        string (nothing to look into), or a key alone, is held with its content
        unknown, so every clause looking into it applies.
        What a carrier's keyable attribute holds (a number, a bool, nothing) is
        no record, and reading the payloads again changes nothing."""
        payloads = HandoffRecord.payloads(
            {
                SceneRecords.SHADOWS.key: '{"planes": []}',
                SceneRecords.LIGHTMAPS.key: {"objects": []},
                "listed": "[1, 2]",
                "rows": [{"a": 1}],
                "broken": "{not json",
                "quoted": '"text"',
                "emissiveGroup_Neon": 0.5,
                "visible": True,
                "unset": None,
                "cleared": "",
                HandoffRecord.KEY: {},
            }
        )
        self.assertEqual(
            payloads,
            {
                SceneRecords.SHADOWS.key: {"planes": []},
                SceneRecords.LIGHTMAPS.key: {"objects": []},
                "listed": [1, 2],
                "rows": [{"a": 1}],
                "broken": Clause.UNKNOWN,
                "quoted": Clause.UNKNOWN,
            },
        )
        self.assertEqual(HandoffRecord.payloads(payloads), payloads, "read again")
        self.assertEqual(HandoffRecord.payloads(["a"]), {"a": Clause.UNKNOWN})
        self.assertEqual(HandoffRecord.carried({}), [])


class TestRendering(unittest.TestCase):
    def test_the_policy_is_values_alone(self):
        """A number, a vector or a name a runtime switches on -- never a
        sentence -- and a colour is linear RGB."""
        for section, fields in HandoffRecord.RENDERING_POLICY.items():
            for field, value in fields.items():
                with self.subTest(field=f"{section}.{field}"):
                    if isinstance(value, str):
                        self.assertFalse(any(c.isspace() for c in value), value)
                    else:
                        self.assertIsInstance(value, (int, float, list))
        self.assertEqual(
            HandoffRecord.RENDERING_POLICY["keyLight"]["color"], [1.0, 1.0, 1.0]
        )
        self.assertEqual(
            HandoffRecord.RENDERING_POLICY["keyLight"]["disabledWhen"], "lightmapped"
        )

    def test_an_exports_choice_lays_over_the_policy_and_only_onto_its_fields(self):
        before = copy.deepcopy(HandoffRecord.RENDERING_POLICY)
        merged = HandoffRecord.rendering_policy(
            {"lightmappedMaterials": {"envMapIntensity": 0.5}}
        )
        self.assertEqual(merged["lightmappedMaterials"]["envMapIntensity"], 0.5)
        self.assertEqual(merged["keyLight"], before["keyLight"])
        self.assertEqual(
            HandoffRecord.RENDERING_POLICY, before, "the constant is never edited"
        )
        self.assertIsNot(
            HandoffRecord.rendering_policy(), HandoffRecord.RENDERING_POLICY
        )
        for bad in (
            {"lightmapped": {"envMapIntensity": 1.0}},
            {"lightmappedMaterials": {"envMapIntensty": 1.0}},
        ):
            with self.subTest(override=bad), self.assertRaises(ValueError):
                HandoffRecord.rendering_policy(bad)

    def test_a_section_overriding_nothing_changes_nothing(self):
        """REGRESSION (2026-10-24): a section whose fields are ``None``
        (``{"keyLight": None}``) passed the unknown-field check, which reads
        ``fields or {}``, then raised TypeError at the merge. It overrides
        nothing, as an empty section does."""
        whole = HandoffRecord.rendering_policy()
        for empty in ({"keyLight": None}, {"keyLight": {}}):
            with self.subTest(override=empty):
                self.assertEqual(HandoffRecord.rendering_policy(empty), whole)

    def test_a_recipe_states_only_the_settings_its_file_uses(self):
        """A file without lightmaps has no lightmapped material to govern, no
        key light to switch off for one, no probe; one with lightmaps but no
        probe no probe to choose or level (``RENDERING_NEEDS``). ``held``
        ``None`` keeps every setting. Added: 2026-10-20"""
        lightmaps = SceneRecords.LIGHTMAPS.key
        whole = HandoffRecord.rendering_policy()
        self.assertEqual(HandoffRecord.rendering_policy(None, None), whole)
        bare = HandoffRecord.rendering_policy(None, {})
        self.assertNotIn("lightmappedMaterials", bare)
        self.assertNotIn("disabledWhen", bare["keyLight"])
        self.assertNotIn("source", bare["environment"])
        self.assertNotIn("probeIntensity", bare["environment"])
        lit = HandoffRecord.rendering_policy(None, {lightmaps: {"objects": []}})
        self.assertEqual(lit["lightmappedMaterials"], whole["lightmappedMaterials"])
        self.assertEqual(lit["keyLight"], whole["keyLight"])
        self.assertNotIn("source", lit["environment"])
        self.assertNotIn("probeIntensity", lit["environment"])
        probed = {lightmaps: {"probe": {"map": "room_Probe.exr"}}}
        self.assertEqual(HandoffRecord.rendering_policy(None, probed), whole)

    def test_the_settings_of_a_published_block_are_its_values(self):
        """A block published before the policy was values alone kept prose in
        its fields: its settings are read, its sentences never."""
        self.assertEqual(
            HandoffRecord.rendering_values(), HandoffRecord.RENDERING_POLICY
        )
        older = {
            "keyLight": {
                "intensity": 0.5,
                "disabled_when": "while any material is lightmapped",
                "note": "Scene-wide, so it cannot be spared the baked geometry.",
            },
            "environment": {
                "source": "the probe, else RoomEnvironment",
                "intensity": True,
            },
        }
        self.assertEqual(
            HandoffRecord.rendering_values(older), {"keyLight": {"intensity": 0.5}}
        )
        self.assertEqual(HandoffRecord.rendering_values({}), {})
        self.assertEqual(HandoffRecord.rendering_values("not a block"), {})

    def test_a_stated_value_counts_only_as_its_fields_kind(self):
        """A stated value is a setting of its field only in that field's kind:
        a number for a number, a numeric vector of the same length for a
        vector, an identifier for an identifier. Every FBX the released
        pipeline wrote states its key light as ``"#ffffff"`` -- the white the
        policy now states as linear ``[1.0, 1.0, 1.0]`` -- and a GLB built from
        one read the hex string as the export's choice and published it (found
        on an end-to-end rebuild of a production FBX). Added: 2026-10-20"""
        self.assertEqual(
            HandoffRecord.overrides({"keyLight": {"color": "#ffffff"}}), {}
        )
        self.assertEqual(
            HandoffRecord.rendering_values({"keyLight": {"position": [1, 2]}}), {}
        )
        self.assertEqual(
            HandoffRecord.rendering_values({"keyLight": {"intensity": "0.5"}}), {}
        )
        self.assertEqual(
            HandoffRecord.rendering_values({"renderer": {"toneMapping": 1}}), {}
        )
        released = {
            "environment": {
                "source": "the asset's reflection probe when it carries one",
                "intensity": 1.0,
            },
            "keyLight": {
                "type": "directional",
                "color": "#ffffff",
                "intensity": 0.9,
                "position": [3, 6, 4],
                "disabled_when": "any material in the asset is lightmapped",
            },
            "lightmappedMaterials": {
                "normalRelief": "scaled by the half-Lambert ratio of N and Ng",
            },
        }
        self.assertEqual(HandoffRecord.overrides(released), {}, "prose is no choice")
        self.assertEqual(
            HandoffRecord.overrides({"keyLight": {"intensity": 0.5}}),
            {"keyLight": {"intensity": 0.5}},
            "a real choice is kept",
        )
        published = HandoffRecord.block(
            "glb", [], rendering=HandoffRecord.overrides(released) or None
        )
        self.assertEqual(published["rendering"]["keyLight"]["color"], [1.0, 1.0, 1.0])

    def test_overrides_are_the_settings_that_differ_from_the_policy(self):
        """What an export chose, read back so a block built from another
        states the same lighting."""
        self.assertEqual(HandoffRecord.overrides(HandoffRecord.rendering_policy()), {})
        self.assertEqual(HandoffRecord.overrides(None), {})
        published = HandoffRecord.rendering_policy(
            {
                "lightmappedMaterials": {"envMapIntensity": 0.5},
                "keyLight": {"intensity": 0.4},
            }
        )
        self.assertEqual(
            HandoffRecord.overrides(published),
            {
                "lightmappedMaterials": {"envMapIntensity": 0.5},
                "keyLight": {"intensity": 0.4},
            },
        )


if __name__ == "__main__":
    unittest.main()
