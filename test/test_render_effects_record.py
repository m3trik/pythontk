# !/usr/bin/python
# coding=utf-8
"""``RenderEffectsRecord`` (``file_utils/mesh_convert/render_effects_record.py``):
the ``render_effects`` record's payload, held to the producer both DCCs
publish through (``MeshConvert.build_visibility_tracks``) and to the transport
its channels ride (``SceneRecords.TRANSPORT``).
"""

import unittest

from pythontk import MeshConvert, RenderEffectsRecord, SceneRecords
from pythontk.file_utils.mesh_convert.glb.fades import CHANNELS
from pythontk.file_utils.mesh_convert.render_effects_record import (
    RenderEffectsMaterial,
    RenderEffectsTrack,
)


class TestRenderEffectsRecord(unittest.TestCase):
    def test_it_is_the_render_effects_records_shape(self):
        self.assertIs(SceneRecords.shape(SceneRecords.VISIBILITY), RenderEffectsRecord)

    def test_version_2_names_channels_and_holds_no_curve(self):
        """v2: a track names its channels and the highlight's colour stops; no
        ramp, frame or rate rides the record (the curves are the transport's,
        in the takes) -- v1's shape is gone from the declaration."""
        self.assertEqual(SceneRecords.VISIBILITY.version, 2)
        self.assertEqual(
            set(RenderEffectsTrack.json_schema()["properties"]),
            {"node", "channels", "highlight_color", "highlight_color_dim"},
        )
        self.assertEqual(
            set(RenderEffectsMaterial.json_schema()["properties"]),
            {"material", "channels", "emission_color", "emission_strength"},
        )
        self.assertEqual(
            set(RenderEffectsRecord.json_schema()["properties"]),
            {"version", "tracks", "materials"},
        )

    def test_channel_names_are_the_transports(self):
        """A track's channels are the transport's node channels, a material's
        its material channels, and every one the GLB writes as a pointer has a
        row in ``GlbFades.CHANNELS``; a highlight's colour stops are the
        track's stop keys."""
        transport = SceneRecords.TRANSPORT
        self.assertEqual(
            set(CHANNELS) | set(transport.stepped),
            set(transport.node_channels) | set(transport.material_channels),
        )
        stops = {key for row in CHANNELS.values() if row.color_stops for key in row.color_stops.keys}
        self.assertEqual(stops, {"highlight_color", "highlight_color_dim"})
        bad = {"version": 2, "tracks": [{"node": "A", "channels": ["opacity", "wobble"]}]}
        self.assertTrue(SceneRecords.problems(SceneRecords.VISIBILITY, bad))
        bad = {"version": 2, "tracks": [], "materials": [{"material": "m", "channels": ["opacity"]}]}
        self.assertTrue(SceneRecords.problems(SceneRecords.VISIBILITY, bad))

    def test_what_the_producer_builds_is_the_declared_shape(self):
        """``build_visibility_tracks`` writes the record a reader reads -- its
        tracks and material entries, and nothing for a scene keying none."""
        built = MeshConvert.build_visibility_tracks(
            [
                {"node": "GATE", "channels": ["visibility"]},
                {
                    "node": "SIGN",
                    "channels": ["opacity", "highlight"],
                    "highlight_color": [0.1, 0.4, 1.0],
                    "highlight_color_dim": [0.0, 0.0, 0.0],
                },
            ],
            materials=[{"material": "screen", "channels": ["emission"], "emission_strength": 4.0}],
        )
        self.assertEqual(built["version"], SceneRecords.VISIBILITY.version)
        self.assertEqual(SceneRecords.problems(SceneRecords.VISIBILITY, built), [])
        self.assertEqual(set(built), {"version", "tracks", "materials"})
        self.assertIsNone(MeshConvert.build_visibility_tracks([]))


if __name__ == "__main__":
    unittest.main()
