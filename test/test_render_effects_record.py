# !/usr/bin/python
# coding=utf-8
"""``RenderEffectsRecord`` (``file_utils/mesh_convert/render_effects_record.py``):
the ``render_effects`` record's payload, held to the producer both DCCs
publish through (``MeshConvert.build_visibility_tracks``) and to the channel
table the GLB build plays a track's effects through (``GlbFades.CHANNELS``).
"""

import unittest

from pythontk import MeshConvert, RenderEffectsRecord, SceneRecords
from pythontk.file_utils.mesh_convert.glb.fades import CHANNELS
from pythontk.file_utils.mesh_convert.render_effects_record import (
    RenderEffectsTrack,
)


class TestRenderEffectsRecord(unittest.TestCase):
    def test_it_is_the_render_effects_records_shape(self):
        self.assertIs(SceneRecords.shape(SceneRecords.VISIBILITY), RenderEffectsRecord)

    def test_a_track_carries_the_channel_tables_effects(self):
        """A track's effect keys are ``GlbFades.CHANNELS``' names and each
        channel's colour stops, beside its node and its stepped visibility:
        an effect the table gains is one the shape declares, and the shape
        declares no other."""
        effects = {"visibility"}
        for name, channel in CHANNELS.items():
            effects.add(name)
            effects.update(channel.color_stops.keys if channel.color_stops else ())
        declared = set(RenderEffectsTrack.json_schema()["properties"])
        self.assertEqual(declared, {"node", *effects})

    def test_what_the_producer_builds_is_the_declared_shape(self):
        """``build_visibility_tracks`` writes the record a reader reads: a
        track of every effect, and the clip spans alone for a scene keying
        none."""
        built = MeshConvert.build_visibility_tracks(
            [
                {"node": "GATE", "visibility": [[1, 0], [10, 1]]},
                {
                    "node": "SIGN",
                    "opacity": [[0, 1.0], [24, 0.0]],
                    "highlight": [[0, 0.0], [12, 1.0]],
                    "highlight_color": [0.1, 0.4, 1.0],
                    "highlight_color_dim": [0.0, 0.0, 0.0],
                },
            ],
            fps=30.0,
            clip_spans={"*": [1.0, 48.0], "A": [1.0, 24.0]},
        )
        spans = MeshConvert.build_visibility_tracks(
            [], fps=24.0, clip_spans={"*": [1.0, 40.0]}
        )
        for name, record in (("tracks", built), ("spans", spans)):
            with self.subTest(record=name):
                self.assertEqual(record["version"], SceneRecords.VISIBILITY.version)
                self.assertEqual(
                    SceneRecords.problems(SceneRecords.VISIBILITY, record), []
                )


if __name__ == "__main__":
    unittest.main()
