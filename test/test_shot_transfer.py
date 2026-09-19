# !/usr/bin/python
# coding=utf-8
"""Tests for the shot transfer codec (``core_utils/engines/shots/shot_transfer.py``).

Pure: every scene contact is a callable the test hands in, standing in for the
two DCC adapters.  The Maya stand-in keys ledger claims by animCurve name and
spells long DAG paths to leaves; the Blender stand-in keys them by
``object|data_path|index``.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pythontk as ptk
from pythontk.core_utils.engines.shots.shot_ledger import ShotEditLedger
from pythontk.core_utils.engines.shots.shot_model import ShotBlock, ShotStore
from pythontk.core_utils.engines.shots.shot_transfer import ShotTransfer


# ---------------------------------------------------------------- stand-ins
_MAYA_CURVES = {
    "pCube1_translateX": ("|grp|pCube1", "translateX"),
    "pCube1_rotateY": ("|grp|pCube1", "rotateY"),
    "orphan_curve": None,  # nothing downstream: the claim cannot travel
    "pSphere1_translateZ": ("|pSphere1", "translateZ"),
}
_BLENDER_LABELS = {
    "translateX": ("location", 0),
    "translateZ": ("location", 2),
    "rotateY": ("rotation_euler", 1),
}


def _leaf(name):
    return str(name).split("|")[-1]


def _maya_curve_ref(curve):
    return _MAYA_CURVES.get(curve)


def _blender_curve_key(obj, label):
    path = _BLENDER_LABELS.get(label)
    return f"{obj}|{path[0]}|{path[1]}" if path else None


def _store_state():
    """A store with every kind of state the transfer must carry."""
    store = ShotStore()
    store.define_shot("Intro", 0.0, 48.0, objects=["|grp|pCube1", "|pSphere1"])
    store.define_shot("Walk", 60.0, 120.0, objects=["|grp|pCube1"])
    store.shots[0].description = "opening"
    store.shots[0].metadata = {
        "csv_objects": ["pCube1", {"name": "pSphere1", "role": "prop"}],
        "object_status": {"|grp|pCube1": "valid"},
    }
    store.shots[1].locked = True
    store.hidden_objects = {"|pSphere1", "|grp|pCube1"}
    store.pinned_objects = {"|missing_thing"}
    store.markers = [{"time": 12.0, "note": "beat", "color": "#ff0000"}]
    store.gap = 6.0
    store.detection_threshold = 3.0
    store.snap_whole_frames = True
    store.lock_gap(0, 1)  # ids are assigned from 0
    store.scene_fps = 24.0
    led = store.edit_ledger
    led.record_key("pCube1_translateX", 48.0, 0, "end")
    led.record_key("pCube1_translateX", 60.0, 1, "start")
    led.record_step("pCube1_rotateY", 48.0, "linear", "linear")
    led.record_key("orphan_curve", 48.0, 0, "end")
    led.record_key("pSphere1_translateZ", 48.0, 0, "end")
    return store.to_dict()


class TestEncode(unittest.TestCase):
    def test_public_name(self):
        self.assertIs(ptk.ShotTransfer, ShotTransfer)

    def test_nothing_to_say_is_none(self):
        self.assertIsNone(ShotTransfer.encode(ShotStore().to_dict()))
        self.assertIsNone(ShotTransfer.encode({}))

    def test_names_are_spelled_and_ledger_regrouped(self):
        section = ShotTransfer.encode(
            _store_state(), spell=_leaf, curve_ref=_maya_curve_ref
        )
        self.assertEqual(section["version"], ShotTransfer.VERSION)
        store = section["store"]
        self.assertEqual(store["shots"][0]["objects"], ["pCube1", "pSphere1"])
        self.assertEqual(store["shots"][0]["description"], "opening")
        self.assertTrue(store["shots"][1]["locked"])
        self.assertEqual(store["hidden_objects"], ["pCube1", "pSphere1"])
        self.assertEqual(store["pinned_objects"], ["missing_thing"])
        self.assertEqual(store["locked_gaps"], [[0, 1]])
        self.assertEqual(store["scene_fps"], 24.0)
        self.assertNotIn("edit_ledger", store)
        meta = store["shots"][0]["metadata"]
        self.assertEqual(meta["csv_objects"][0], "pCube1")
        self.assertEqual(meta["csv_objects"][1]["name"], "pSphere1")
        self.assertEqual(meta["object_status"], {"pCube1": "valid"})
        ledger = section["ledger"]
        self.assertEqual(
            ledger["keys"]["pCube1"]["translateX"],
            [[48.0, 0, "end"], [60.0, 1, "start"]],
        )
        self.assertEqual(
            ledger["steps"]["pCube1"]["rotateY"], [[48.0, "linear", "linear"]]
        )
        self.assertEqual(ledger["keys"]["pSphere1"]["translateZ"], [[48.0, 0, "end"]])
        # The orphan curve resolves to nothing, so its claim stays behind.
        for by_object in ledger.values():
            for by_label in by_object.values():
                self.assertNotIn("orphan_curve", by_label)

    def test_objects_scope_filters_members_and_claims_but_keeps_shots(self):
        section = ShotTransfer.encode(
            _store_state(),
            spell=_leaf,
            curve_ref=_maya_curve_ref,
            objects=["pCube1"],  # short form: scoping compares through spell
        )
        store = section["store"]
        self.assertEqual([s["name"] for s in store["shots"]], ["Intro", "Walk"])
        self.assertEqual(store["shots"][0]["objects"], ["pCube1"])
        self.assertEqual(store["hidden_objects"], ["pCube1"])
        self.assertEqual(store["pinned_objects"], [])
        self.assertNotIn("pSphere1", section["ledger"]["keys"])

    def test_no_curve_ref_means_no_ledger(self):
        section = ShotTransfer.encode(_store_state(), spell=_leaf)
        self.assertEqual(section["ledger"], {"steps": {}, "keys": {}})

    def test_encode_does_not_mutate_the_state(self):
        state = _store_state()
        before = ShotStore.from_dict(state).to_dict()
        ShotTransfer.encode(state, spell=_leaf, curve_ref=_maya_curve_ref, objects=[])
        self.assertEqual(state, before)


class TestDecode(unittest.TestCase):
    def setUp(self):
        self.section = ShotTransfer.encode(
            _store_state(), spell=_leaf, curve_ref=_maya_curve_ref
        )

    def test_round_trip_is_the_same_store(self):
        decoded = ShotTransfer.decode(
            self.section,
            resolve=lambda leaf: {"pCube1": "|grp|pCube1", "pSphere1": "|pSphere1"}.get(
                leaf
            ),
            curve_key=lambda node, label: next(
                (c for c, ref in _MAYA_CURVES.items() if ref == (node, label)), None
            ),
            scene_fps=24.0,
        )
        original = _store_state()
        store = ShotStore.from_dict(decoded)
        self.assertEqual(
            [(s.shot_id, s.name, s.start, s.end, s.objects) for s in store.shots],
            [
                (0, "Intro", 0.0, 48.0, ["|grp|pCube1", "|pSphere1"]),
                (1, "Walk", 60.0, 120.0, ["|grp|pCube1"]),
            ],
        )
        self.assertEqual(store.hidden_objects, {"|grp|pCube1", "|pSphere1"})
        # An unresolvable pinned name is kept as spelled: pinning tracks the missing.
        self.assertEqual(store.pinned_objects, {"missing_thing"})
        self.assertEqual(store.locked_gaps, {(0, 1)})
        self.assertEqual(store.gap, 6.0)
        self.assertEqual(store.detection_threshold, 3.0)
        self.assertEqual(store.markers, original["markers"])
        led = store.edit_ledger
        self.assertEqual(led.key_times("pCube1_translateX"), [48.0, 60.0])
        self.assertEqual(led.step_times("pCube1_rotateY"), [48.0])
        self.assertEqual(led.key_times("pSphere1_translateZ"), [48.0])
        self.assertEqual(led.key_times("orphan_curve"), [])

    def test_blender_side_keys_and_frame_offset(self):
        decoded = ShotTransfer.decode(
            self.section,
            resolve=lambda leaf: leaf if leaf in ("pCube1", "pSphere1") else None,
            curve_key=_blender_curve_key,
            scene_fps=24.0,
            frame_offset=1.0,
        )
        self.assertEqual(
            [(s["start"], s["end"]) for s in decoded["shots"]],
            [(1.0, 49.0), (61.0, 121.0)],
        )
        self.assertEqual(decoded["markers"][0]["time"], 13.0)
        self.assertEqual(
            decoded["edit_ledger"]["keys"]["pCube1|location|0"],
            [[49.0, 0, "end"], [61.0, 1, "start"]],
        )
        self.assertEqual(
            decoded["edit_ledger"]["steps"]["pCube1|rotation_euler|1"],
            [[49.0, "linear", "linear"]],
        )
        self.assertEqual(decoded["hidden_objects"], ["pCube1", "pSphere1"])

    def test_fps_rescale_matches_the_store_rule(self):
        decoded = ShotTransfer.decode(
            self.section,
            curve_key=_blender_curve_key,
            scene_fps=30.0,
        )
        self.assertEqual(
            [(s["start"], s["end"]) for s in decoded["shots"]],
            [(0.0, 60.0), (75.0, 150.0)],
        )
        self.assertEqual(decoded["gap"], 7.5)
        self.assertEqual(decoded["markers"][0]["time"], 15.0)
        self.assertEqual(decoded["scene_fps"], 30.0)
        self.assertEqual(
            [r[0] for r in decoded["edit_ledger"]["keys"]["pCube1|location|0"]],
            [60.0, 75.0],
        )
        # A store rebuilt from it does NOT rescale again on load.
        store = ShotStore.from_dict(decoded)
        self.assertEqual(store.scene_fps, 30.0)

    def test_sub_frame_bounds_survive_without_snap(self):
        section = ShotTransfer.encode(_store_state(), spell=_leaf)
        section["store"]["snap_whole_frames"] = False
        decoded = ShotTransfer.decode(section, scene_fps=30.0)
        self.assertEqual(decoded["shots"][1]["start"], 75.0)
        section["store"]["shots"][0]["end"] = 47.0
        decoded = ShotTransfer.decode(section, scene_fps=30.0)
        self.assertEqual(decoded["shots"][0]["end"], 58.75)

    def test_a_claim_without_its_key_is_dropped(self):
        decoded = ShotTransfer.decode(
            self.section,
            curve_key=_blender_curve_key,
            key_exists=lambda key, t: t != 60.0,  # the reducer removed frame 60
            scene_fps=24.0,
        )
        self.assertEqual(
            decoded["edit_ledger"]["keys"]["pCube1|location|0"], [[48.0, 0, "end"]]
        )

    def test_unresolvable_object_drops_its_claims_and_membership(self):
        decoded = ShotTransfer.decode(
            self.section,
            resolve=lambda leaf: leaf if leaf == "pCube1" else None,
            curve_key=_blender_curve_key,
        )
        self.assertEqual(decoded["shots"][0]["objects"], ["pCube1"])
        self.assertNotIn("pSphere1|location|2", decoded["edit_ledger"].get("keys", {}))

    def test_a_converted_root_swaps_its_up_axis_channels(self):
        # pSphere1 is a root the importer put through the Y-up / Z-up crossing:
        # its translateZ claim must look for the receiving Y channel.
        seen = []

        def curve_key(node, label):
            seen.append((node, label))
            return f"{node}|{label}"

        ShotTransfer.decode(
            self.section,
            curve_key=curve_key,
            converted=lambda node: node == "pSphere1",
        )
        self.assertIn(("pSphere1", "translateY"), seen)
        self.assertNotIn(("pSphere1", "translateZ"), seen)
        # A child keeps its parent-space channels.
        self.assertIn(("pCube1", "rotateY"), seen)
        self.assertEqual(ShotTransfer.swap_up_axis("scaleY"), "scaleZ")
        self.assertEqual(ShotTransfer.swap_up_axis("translateX"), "translateX")
        self.assertEqual(ShotTransfer.swap_up_axis("visibility"), "visibility")

    def test_no_curve_key_means_no_ledger(self):
        decoded = ShotTransfer.decode(self.section)
        self.assertEqual(decoded["edit_ledger"], {})
        self.assertFalse(ShotStore.from_dict(decoded).edit_ledger)

    def test_newer_schema_is_refused(self):
        with self.assertRaises(ValueError):
            ShotTransfer.decode({"version": ShotTransfer.VERSION + 1, "store": {}})

    def test_decoded_ledger_is_canonical(self):
        section = ShotTransfer.encode(
            _store_state(), spell=_leaf, curve_ref=_maya_curve_ref
        )
        # Records arriving out of order and as a bare time list still load.
        section["ledger"]["keys"]["pCube1"]["translateX"] = [[60.0, 1, "start"], 48.0]
        decoded = ShotTransfer.decode(section, curve_key=_blender_curve_key)
        led = ShotEditLedger.from_dict(decoded["edit_ledger"])
        self.assertEqual(led.key_times("pCube1|location|0"), [48.0, 60.0])


class TestChannelsAndAudio(unittest.TestCase):
    CHANNELS = {
        "|grp|pCube1": {
            "opacity": {
                "value": 1.0,
                "keys": [[24.0, 1.0, "linear"], [48.0, 0.0, "step"]],
            },
            "highlightColorR": {"value": 0.5, "keys": []},
        },
        "|pSphere1": {"opacity": {"value": 0.0, "keys": [[12.0, 0.0, "smooth"]]}},
    }
    AUDIO = [
        {
            "name": "footstep",
            "file": "C:/a.wav",
            "start": 24.0,
            "end": 60.0,
            "offset": 8.0,
        }
    ]

    def test_encode_spells_and_scopes_channels_and_carries_audio(self):
        section = ShotTransfer.encode(
            _store_state(),
            spell=_leaf,
            channels=self.CHANNELS,
            audio=self.AUDIO,
            objects=["pCube1"],
        )
        self.assertEqual(section["version"], 2)
        self.assertEqual(list(section["channels"]), ["pCube1"])
        self.assertEqual(
            section["channels"]["pCube1"]["opacity"]["keys"],
            [[24.0, 1.0, "linear"], [48.0, 0.0, "step"]],
        )
        self.assertEqual(section["audio"], self.AUDIO)

    def test_channels_or_audio_alone_make_a_section(self):
        self.assertIsNotNone(
            ShotTransfer.encode(ShotStore().to_dict(), channels=self.CHANNELS)
        )
        self.assertIsNotNone(ShotTransfer.encode({}, audio=self.AUDIO))
        self.assertNotIn("channels", ShotTransfer.encode(_store_state()))

    def test_decode_lands_channels_before_claims_and_retimes_everything(self):
        section = ShotTransfer.encode(
            _store_state(), spell=_leaf, channels=self.CHANNELS, audio=self.AUDIO
        )
        # A claim on the opacity channel the transfer itself is about to create.
        section["ledger"]["keys"]["pCube1"] = {"opacity": [[48.0, 0, "end"]]}
        landed = {}
        clips = []

        def write_channels(node, records):
            landed[node] = records

        ShotTransfer.decode(
            section,
            curve_key=lambda node, label: f"{node}|{label}",
            key_exists=lambda key, t: any(
                k[0] == t
                for k in landed.get(key.split("|")[0], {})
                .get(key.split("|")[1], {})
                .get("keys", [])
            ),
            write_channels=write_channels,
            write_audio=clips.extend,
            scene_fps=30.0,  # the section was authored at 24: x1.25
            frame_offset=1.0,
        )
        self.assertEqual(
            landed["pCube1"]["opacity"]["keys"],
            [[31.0, 1.0, "linear"], [61.0, 0.0, "step"]],
        )
        self.assertEqual(
            landed["pCube1"]["highlightColorR"], {"value": 0.5, "keys": []}
        )
        self.assertEqual(landed["pSphere1"]["opacity"]["keys"], [[16.0, 0.0, "smooth"]])
        self.assertEqual(
            clips,
            [
                {
                    "name": "footstep",
                    "file": "C:/a.wav",
                    "start": 31.0,
                    "end": 76.0,
                    "offset": 10.0,
                }
            ],
        )
        # The claim found its key only because the channel landed first.
        decoded = ShotTransfer.decode(
            section,
            curve_key=lambda node, label: f"{node}|{label}",
            key_exists=lambda key, t: t == 61.0,
            write_channels=lambda node, records: None,
            scene_fps=30.0,
            frame_offset=1.0,
        )
        self.assertEqual(
            decoded["edit_ledger"]["keys"]["pCube1|opacity"], [[61.0, 0, "end"]]
        )

    def test_a_v1_section_still_decodes(self):
        section = ShotTransfer.encode(_store_state(), spell=_leaf)
        section["version"] = 1
        called = []
        decoded = ShotTransfer.decode(
            section,
            write_channels=lambda *a: called.append(a),
            write_audio=called.append,
        )
        self.assertEqual([s["name"] for s in decoded["shots"]], ["Intro", "Walk"])
        self.assertEqual(called, [])

    def test_merge_keeps_the_scene_store_for_a_content_only_section(self):
        existing = _store_state()
        incoming = ShotTransfer.decode(
            ShotTransfer.encode(ShotStore().to_dict(), channels=self.CHANNELS)
        )
        self.assertEqual(ShotTransfer.merge(existing, incoming), existing)
        self.assertEqual(ShotTransfer.merge(None, incoming), incoming)


class TestMerge(unittest.TestCase):
    def test_a_scene_without_shots_adopts_the_incoming_store(self):
        incoming = _store_state()
        existing = ShotStore().to_dict()
        existing["gap"] = 99.0
        merged = ShotTransfer.merge(existing, incoming)
        self.assertEqual(merged, incoming)
        self.assertEqual(ShotTransfer.merge(None, incoming), incoming)

    def test_a_scene_with_shots_appends_under_fresh_ids(self):
        existing = ShotStore()
        existing.define_shot("Existing", 0.0, 30.0, objects=["a"])
        existing.shots[0].shot_id = 5
        existing.gap = 2.0
        existing.edit_ledger.record_key("a_tx", 30.0, 5, "end")
        incoming = _store_state()
        merged = ShotTransfer.merge(existing.to_dict(), incoming)
        store = ShotStore.from_dict(merged)
        self.assertEqual(
            [(s.shot_id, s.name) for s in store.sorted_shots()],
            [(5, "Existing"), (6, "Intro"), (7, "Walk")],
        )
        self.assertEqual(store.gap, 2.0)  # the scene's own settings stand
        self.assertIn((6, 7), store.locked_gaps)
        self.assertEqual(store.hidden_objects, {"|pSphere1", "|grp|pCube1"})
        self.assertEqual(store.markers, incoming["markers"])
        led = store.edit_ledger
        self.assertEqual(led.key_times("a_tx"), [30.0])
        # A claim's owner follows the shot it was remapped to.
        owners = {rec[0]: rec[1] for rec in led.to_dict()["keys"]["pCube1_translateX"]}
        self.assertEqual(owners, {48.0: 6, 60.0: 7})
        self.assertEqual(led.step_times("pCube1_rotateY"), [48.0])

    def test_an_incoming_name_the_scene_already_has_is_numbered(self):
        """Each store is unique on its own; merged, ``Intro`` and ``intro``
        would be one clip (Unity joins ignoring case) -- so the arrival is
        numbered, as the scene's own tools would have had it named."""
        existing = ShotStore()
        existing.define_shot("intro", 0.0, 30.0)
        merged = ShotTransfer.merge(existing.to_dict(), _store_state())
        store = ShotStore.from_dict(merged)
        names = [s.name for s in store.sorted_shots()]
        self.assertEqual(names, ["intro", "Intro_2", "Walk"])
        for shot in store.shots:
            self.assertIsNone(store.name_error(shot.name, shot.shot_id), shot.name)

    def test_an_incoming_legacy_name_is_merged_legal(self):
        """A store from before names were validated can bring ``"Shot 1"``;
        numbered beside the scene's ``Shot_1`` it must become a name the store
        accepts (``Shot_1_2``), never ``"Shot 1_2"`` -- and a blank one a
        name at all."""
        existing = ShotStore()
        existing.define_shot("Shot_1", 0.0, 30.0)
        incoming = ShotStore([ShotBlock(0, "Shot 1", 40, 50), ShotBlock(1, "", 60, 70)])
        merged = ShotTransfer.merge(existing.to_dict(), incoming.to_dict())
        store = ShotStore.from_dict(merged)
        self.assertEqual(
            [s.name for s in store.sorted_shots()], ["Shot_1", "Shot_1_2", "shot"]
        )
        for shot in store.shots:
            self.assertIsNone(store.name_error(shot.name, shot.shot_id), shot.name)

    def test_merge_is_idempotent_on_ledger_records(self):
        existing = _store_state()
        merged = ShotTransfer.merge(existing, _store_state())
        led = ShotEditLedger.from_dict(merged["edit_ledger"])
        # Same curve, same records: no duplicates, the remapped owners aside.
        self.assertEqual(len(led.to_dict()["steps"]["pCube1_rotateY"]), 1)

    def test_merge_does_not_mutate_inputs(self):
        existing = _store_state()
        incoming = _store_state()
        before = (
            ShotStore.from_dict(existing).to_dict(),
            ShotStore.from_dict(incoming).to_dict(),
        )
        ShotTransfer.merge(existing, incoming)
        self.assertEqual((existing, incoming), before)


if __name__ == "__main__":
    unittest.main()
