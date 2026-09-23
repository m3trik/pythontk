# !/usr/bin/python
# coding=utf-8
"""GlbPipeline -- the one FBX -> GLB build the exporters and the preview share."""

import os
import unittest
import unittest.mock
from pathlib import Path

from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert
from pythontk.file_utils.mesh_convert.glb_pipeline import GlbPipeline
from pythontk.file_utils.temp_artifacts import TempArtifacts


CONVERT = "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.fbx_to_glb"
OPTIMIZE = (
    "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.optimize_glb_textures"
)


class GlbPipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = TempArtifacts("test_glb_pipeline", policy="scoped")
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def _stub_fbx(self):
        path = self.temp.path(extension=".fbx")
        Path(path).write_bytes(b"not-a-binary-fbx")
        return path

    def _real_fbx(self):
        from test_fbx_media import build_fbx, png_bytes

        return build_fbx(
            self.temp.path(extension=".fbx"),
            {"wall_Base_color.png": png_bytes((512, 256))},
        )

    def _fake_convert(self, src, dst=None, **kwargs):
        self.calls.append(("convert", src, dst, kwargs))
        out = dst or os.path.splitext(src)[0] + ".glb"
        Path(out).write_bytes(b"glTF")
        report = kwargs.get("report")
        if report is not None:
            report["sidecar"] = {"emissive": "1 of 1"}
            report["lightmaps"] = {
                "expected": 1,
                "bound": 1,
                "unbound": [],
                "out_of_scope": 0,
            }
            if kwargs.get("data_export"):
                report["data_export"] = sorted(kwargs["data_export"])
        return out

    def _fake_optimize(self, glb, **kwargs):
        self.calls.append(("optimize", glb, kwargs))
        return {"images": 1, "bytes_before": 2e6, "bytes_after": 1e6}

    def _build(self, src, **kwargs):
        with (
            unittest.mock.patch(CONVERT, side_effect=self._fake_convert),
            unittest.mock.patch(OPTIMIZE, side_effect=self._fake_optimize),
        ):
            return GlbPipeline.build(src, **kwargs)

    def test_the_stages_run_in_order_with_the_dials_forwarded(self):
        """Convert (sidecar, lightmap dirs, no prompt) then optimize (the caller's
        params plus its fallback choice), and the report carries both outcomes."""
        src = self._stub_fbx()
        dst = self.temp.path(extension=".glb")
        envelope = {"sections": {"emissive": {}}}
        built = self._build(
            src,
            dst=dst,
            sidecar=envelope,
            lightmap_dirs=["D:/maps"],
            texture_params={
                "image_format": "KTX2",
                "max_size": 1024,
                "ktx2_fallback": False,
            },
        )
        self.assertEqual([c[0] for c in self.calls], ["convert", "optimize"])
        _, read, written, kwargs = self.calls[0]
        self.assertEqual((read, written), (src, dst))
        self.assertIs(kwargs["prompt"], False)
        self.assertIs(kwargs["overwrite"], True)
        self.assertEqual(kwargs["sidecar"], envelope)
        self.assertEqual(kwargs["lightmap_dirs"], ["D:/maps"])
        _, glb, params = self.calls[1]
        self.assertEqual(glb, dst)
        self.assertEqual(
            params, {"image_format": "KTX2", "max_size": 1024, "ktx2_fallback": False}
        )
        self.assertEqual(built["glb"], dst)
        self.assertEqual(built["src"], src)
        self.assertEqual(built["sidecar"], {"emissive": "1 of 1"})
        self.assertEqual(built["lightmaps"]["bound"], 1)
        self.assertEqual(built["textures"]["images"], 1)
        self.assertEqual(built["scratch"], [])

    def test_a_data_export_overlay_reaches_the_conversion(self):
        """The overlay is the conversion's to apply (its passes read the
        channels); the pipeline only forwards it, and ``None`` by default."""
        overlay = {"fbx_takes": None}
        built = self._build(self._stub_fbx(), data_export=overlay)
        self.assertIs(self.calls[0][3]["data_export"], overlay)
        # And the report says what the conversion replaced, so a caller can
        # tell an applied overlay from one that was quietly never honoured.
        self.assertEqual(built["data_export"], ["fbx_takes"])
        built = self._build(self._stub_fbx())
        self.assertIsNone(self.calls[-2][3]["data_export"])
        self.assertEqual(built["data_export"], [])

    def test_an_overlay_on_a_finished_glb_is_warned_not_swallowed(self):
        """A GLB source runs no pass, so an overlay cannot be honoured -- and a
        publish that silently dropped it would look like one that applied it."""
        src = self.temp.path(extension=".glb")
        Path(src).write_bytes(b"glTF")
        logger = unittest.mock.MagicMock()
        built = GlbPipeline.build(src, data_export={"fbx_takes": None}, logger=logger)
        logger.warning.assert_called_once()
        self.assertEqual(built["data_export"], [], "nothing landed, and it says so")
        logger.reset_mock()
        GlbPipeline.build(src, logger=logger)
        logger.warning.assert_not_called()

    def test_a_finished_glb_reports_under_the_same_keys_as_a_build(self):
        """One report shape whichever path ran. The ``.glb`` short-circuit
        returned before the key-reduction stage set ``"animation"``, so a
        caller reading the summary the docstring promised met a KeyError on a
        finished-GLB source alone.
        Added: 2026-09-15"""
        src = self.temp.path(extension=".glb")
        Path(src).write_bytes(b"glTF")
        finished = GlbPipeline.build(src)
        self.assertEqual(sorted(finished), sorted(self._build(self._stub_fbx())))
        self.assertIsNone(finished["animation"])

    def test_no_texture_params_means_the_shared_web_delivery_policy(self):
        self._build(self._stub_fbx())
        _, _glb, params = self.calls[-1]
        policy = MeshConvert.web_delivery_texture_params()
        self.assertEqual(params, policy, "the optimizer's own defaults apply")

    def test_a_key_tolerance_runs_the_reduction_between_convert_and_optimize(self):
        """The reduction is a stage of its own -- after the conversion (whose
        session already cut the clips and collapsed the constants) and before
        the texture pass -- and only when a tolerance is named.
        Added: 2026-09-13"""
        reduce_path = (
            "pythontk.file_utils.mesh_convert._mesh_convert."
            "MeshConvert.reduce_glb_animations"
        )
        summary = {"samplers": 3, "keys_before": 30, "keys_after": 9, "bytes": 84}

        def reduce_keys(glb, tolerance):
            self.calls.append(("reduce", glb, tolerance))
            return summary

        with unittest.mock.patch(reduce_path, side_effect=reduce_keys) as reduce:
            built = self._build(self._stub_fbx(), key_tolerance=1e-4)
        reduce.assert_called_once_with(built["glb"], 1e-4)
        # One record across the three stages, so the ORDER is what is asserted:
        # each mock having been called says nothing about when.
        self.assertEqual(
            [call[0] for call in self.calls], ["convert", "reduce", "optimize"]
        )
        self.assertEqual(built["animation"], summary)
        with unittest.mock.patch(reduce_path) as reduce:
            self.assertIsNone(self._build(self._stub_fbx())["animation"])
        reduce.assert_not_called()

    def test_a_conversion_failure_raises_and_runs_no_texture_pass(self):
        with (
            unittest.mock.patch(CONVERT, side_effect=RuntimeError("FBX2glTF exit 1")),
            unittest.mock.patch(OPTIMIZE, side_effect=self._fake_optimize),
        ):
            with self.assertRaises(RuntimeError):
                GlbPipeline.build(self._stub_fbx())
        self.assertEqual(self.calls, [])

    def test_a_texture_pass_failure_raises_rather_than_shipping_raw(self):
        """The old exporter chain failed the deliverable and the old preview
        chain swallowed it and published 280 MB; one chain, one answer."""
        with (
            unittest.mock.patch(CONVERT, side_effect=self._fake_convert),
            unittest.mock.patch(OPTIMIZE, side_effect=RuntimeError("encode failed")),
        ):
            with self.assertRaises(RuntimeError):
                GlbPipeline.build(self._stub_fbx())

    def test_a_progress_hook_hears_every_stage(self):
        heard = []
        self._build(self._stub_fbx(), progress=heard.append)
        self.assertEqual(len(heard), 2, heard)
        self.assertIn("converting", heard[0])
        self.assertIn("texture pass", heard[1])

    # -- the downsize stage -----------------------------------------------

    def test_an_fbx_over_the_ceiling_is_read_through_a_downsized_scratch_copy(self):
        """The converter reads the 2K copy; the caller's allocator names it,
        owns it (listed in ``scratch``) and is told the original is superseded."""
        src = self._real_fbx()
        minted, released = [], []

        def allocate(extension):
            path = self.temp.path(extension=extension)
            minted.append(path)
            return path

        built = self._build(
            src,
            texture_params={"image_format": "WEBP", "max_size": 128},
            scratch_path=allocate,
            release_source=released.append,
        )
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia

        self.assertEqual(minted, [built["src"]])
        self.assertNotEqual(built["src"], src)
        self.assertEqual(built["scratch"], minted)
        self.assertEqual(released, [src])
        self.assertEqual(self.calls[0][1], built["src"], "the converter read the copy")
        self.assertEqual(FbxMedia.embedded(built["src"])[0]["size"], (128, 64))
        self.assertEqual(built["downsized"]["resized"], 1)
        self.assertTrue(os.path.exists(src), "releasing is the caller's decision")

    def test_without_an_allocator_the_scratch_copy_is_swept_when_the_build_returns(
        self,
    ):
        src = self._real_fbx()
        built = self._build(
            src, texture_params={"image_format": "WEBP", "max_size": 128}
        )
        self.assertNotEqual(built["src"], src)
        self.assertFalse(os.path.exists(built["src"]), "own scratch must not leak")
        self.assertTrue(os.path.exists(built["glb"]))

    def test_nothing_over_the_ceiling_means_the_original_is_read(self):
        src = self._real_fbx()
        built = self._build(
            src, texture_params={"image_format": "WEBP", "max_size": 4096}
        )
        self.assertEqual(built["src"], src)
        self.assertEqual(built["scratch"], [])
        self.assertEqual(built["downsized"]["resized"], 0)

    def test_downsize_off_or_a_non_fbx_source_skips_the_stage(self):
        src = self._real_fbx()
        built = self._build(
            src,
            texture_params={"image_format": "WEBP", "max_size": 128},
            downsize=False,
        )
        self.assertEqual(built["src"], src)
        self.assertIsNone(built["downsized"])
        stub = self._stub_fbx()
        built = self._build(
            stub, texture_params={"image_format": "WEBP", "max_size": 128}
        )
        self.assertEqual(built["src"], stub)
        self.assertIsNone(built["downsized"])

    def test_a_downsize_failure_is_a_warning_and_the_original_is_read(self):
        """A speed win the texture ceiling re-applies anyway must never cost the build."""
        import logging

        src = self._real_fbx()
        log = logging.getLogger("test_glb_pipeline")
        with (
            unittest.mock.patch(CONVERT, side_effect=self._fake_convert),
            unittest.mock.patch(OPTIMIZE, side_effect=self._fake_optimize),
            unittest.mock.patch(
                "pythontk.file_utils.mesh_convert.fbx_media.FbxMedia.downsize",
                side_effect=OSError("disk full"),
            ),
            self.assertLogs(log, level="WARNING") as caught,
        ):
            built = GlbPipeline.build(
                src,
                texture_params={"image_format": "WEBP", "max_size": 128},
                logger=log,
            )
        self.assertEqual(built["src"], src)
        self.assertTrue(
            any("downsize skipped" in m for m in caught.output), caught.output
        )

    # -- the split-take stage ---------------------------------------------

    def _animated_fbx(
        self, declared, takes=("Take 001", "Shot_A", "Shot_B"), media=None
    ):
        from test_fbx_media import build_animated_fbx

        return build_animated_fbx(
            self.temp.path(extension=".fbx"),
            list(takes),
            declared=declared,
            media=media,
        )

    def _convert_reading_takes(self, src, dst=None, **kwargs):
        """The fake converter, also recording which takes it was handed."""
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

        self.read_takes = FbxFile.load(src).take_names()
        return self._fake_convert(src, dst, **kwargs)

    def _build_reading_takes(self, src, **kwargs):
        with (
            unittest.mock.patch(CONVERT, side_effect=self._convert_reading_takes),
            unittest.mock.patch(OPTIMIZE, side_effect=self._fake_optimize),
        ):
            return GlbPipeline.build(src, **kwargs)

    def test_declared_shot_takes_never_reach_the_converter(self):
        """The converter reads a copy holding only the whole-timeline stack;
        the caller's allocator owns it and is told the original is superseded."""
        src = self._animated_fbx(declared=["Shot_A", "Shot_B"])
        minted, released = [], []

        def allocate(extension):
            path = self.temp.path(extension=extension)
            minted.append(path)
            return path

        built = self._build_reading_takes(
            src, scratch_path=allocate, release_source=released.append
        )
        self.assertEqual(self.read_takes, ["Take 001"])
        self.assertEqual(built["takes"]["takes"], ["Shot_A", "Shot_B"])
        # The downsize stage also asks the allocator for a path; nothing in a
        # media-less file qualifies, so only the strip's copy is kept.
        self.assertEqual(built["scratch"], [built["src"]])
        self.assertEqual(minted[0], built["src"])
        self.assertEqual(released, [src])
        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

        self.assertEqual(
            FbxFile.load(src).take_names(), ["Take 001", "Shot_A", "Shot_B"]
        )

    def test_undeclared_takes_or_no_stack_left_to_cut_from_reads_the_original(self):
        for declared in (None, [], ["Take 001", "Shot_A", "Shot_B"]):
            with self.subTest(declared=declared):
                src = self._animated_fbx(declared=declared)
                built = self._build_reading_takes(src)
                self.assertEqual(built["src"], src)
                self.assertIsNone(built["takes"])
                self.assertEqual(self.read_takes, ["Take 001", "Shot_A", "Shot_B"])

    def test_the_downsize_reads_the_stripped_copy_and_the_intermediate_is_released(
        self,
    ):
        """Two payload stages chain: the downsize reads the strip's output, and
        with no caller allocator the intermediate goes as soon as it is superseded."""
        from test_fbx_media import png_bytes

        src = self._animated_fbx(
            declared=["Shot_A", "Shot_B"], media={"wall.png": png_bytes((512, 256))}
        )
        seen = []

        def spy(path):
            seen.append(path)

        with unittest.mock.patch.object(
            TempArtifacts, "release", autospec=True
        ) as release:
            release.side_effect = lambda store, path: spy(path) or True
            built = self._build_reading_takes(
                src, texture_params={"image_format": "WEBP", "max_size": 128}
            )
        self.assertEqual(self.read_takes, ["Take 001"])
        self.assertEqual(built["downsized"]["resized"], 1)
        self.assertEqual(len(built["scratch"]), 2)
        self.assertEqual(
            seen, [built["scratch"][0]], "only OUR intermediate is released"
        )
        self.assertEqual(built["src"], built["scratch"][1])

    def test_the_glb_lands_beside_the_callers_fbx_whatever_the_converter_read(self):
        """No *dst* means beside *src* -- the caller's FBX -- even when a payload
        stage swapped a scratch copy in as the converter's input. Regression: a
        production export's GLB was written into the temp dir beside the
        stripped copy and never reached the export folder.
        Added: 2026-09-12
        """
        cases = {
            "split takes": (self._animated_fbx(declared=["Shot_A", "Shot_B"]), None),
            "downsize": (
                self._real_fbx(),
                {"image_format": "WEBP", "max_size": 128},
            ),
        }
        for stage, (src, params) in cases.items():
            with self.subTest(stage=stage):
                beside = os.path.splitext(src)[0] + ".glb"
                self.addCleanup(lambda p=beside: os.path.exists(p) and os.remove(p))
                built = self._build_reading_takes(src, texture_params=params)
                self.assertNotEqual(built["src"], src, "fixture: the stage must run")
                self.assertEqual(built["glb"], beside)
                self.assertEqual(self.calls[-1][1], beside, "the texture pass too")
                self.assertTrue(os.path.exists(beside))

    # -- the sidecar envelope --------------------------------------------

    def test_envelope_wraps_the_hosts_read_once_for_every_producer(self):
        envelope = GlbPipeline.envelope(
            lambda: {"emissive": {"m": {"color": [1, 0, 0]}}},
            source={"application": "test", "version": "0"},
            asset="scene.fbx",
        )
        self.assertEqual(envelope["version"], MeshConvert.SIDECAR_VERSION)
        self.assertEqual(envelope["asset"], "scene.fbx")
        self.assertIn("emissive", envelope["sections"])
        self.assertEqual(
            envelope["handoff"]["rendering"],
            MeshConvert.RENDERING_POLICY,
            "no choices made: the recipe as declared",
        )

    def test_the_envelope_publishes_the_exports_lighting_choices(self):
        """Both producers hand their rows' choices in (``ExportRun.rendering``);
        the envelope publishes them as ``handoff.rendering``, where the viewer
        and any other reader find them. Added: 2026-09-21"""
        rendering = {"lightmappedMaterials": {"envMapIntensity": 0.5}}
        envelope = GlbPipeline.envelope(
            lambda: {}, source={"application": "test"}, rendering=rendering
        )
        self.assertEqual(
            envelope["handoff"]["rendering"], MeshConvert.rendering_policy(rendering)
        )

    def test_a_failing_read_degrades_to_an_empty_envelope_not_a_failed_build(self):
        """Requested-but-empty stays distinguishable from switched off: the
        producer still attaches an envelope, and the panel says 'nothing to
        carry' rather than 'off' -- a bare GLB still beats no GLB."""
        import logging

        log = logging.getLogger("test_glb_pipeline")

        def _boom():
            raise RuntimeError("scene read failed")

        with self.assertLogs(log, level="WARNING") as caught:
            envelope = GlbPipeline.envelope(
                _boom, source={"application": "test", "version": "0"}, logger=log
            )
        self.assertEqual(envelope["sections"], {})
        self.assertTrue(any("sidecar skipped" in m for m in caught.output))


class TestDeclaredTakes(unittest.TestCase):
    """Which takes an FBX declares: the shot record's clips (each carries its
    range since 0.11.0) first, the legacy ``fbx_takes`` channel for a file
    written before."""

    class _Fbx:
        def __init__(self, props):
            self.props = props

        def user_properties(self, name):
            return self.props.get(name, [])

    def test_the_shot_record_is_read_first(self):
        fbx = self._Fbx(
            {
                "shot_metadata": [
                    b'{"version": 1, "shots": [{"clip": "A", "start": 1, "end": 9}]}'
                ],
                "fbx_takes": [b'[{"name": "legacy"}]'],
            }
        )
        self.assertEqual(GlbPipeline._declared_takes(fbx), {"A"})

    def test_every_carrier_is_read(self):
        """An imported reference brings its own carrier and clips."""
        fbx = self._Fbx(
            {
                "shot_metadata": [
                    b'{"shots": [{"clip": "A", "start": 1, "end": 9}]}',
                    b'{"shots": [{"clip": "B", "start": 20, "end": 30}]}',
                ]
            }
        )
        self.assertEqual(GlbPipeline._declared_takes(fbx), {"A", "B"})

    def test_clips_without_ranges_are_an_older_file(self):
        fbx = self._Fbx(
            {
                "shot_metadata": [b'{"shots": [{"clip": "A"}]}'],
                "fbx_takes": [b'[{"name": "legacy"}]'],
            }
        )
        self.assertEqual(GlbPipeline._declared_takes(fbx), {"legacy"})

    def test_an_older_file_falls_back_to_fbx_takes(self):
        fbx = self._Fbx({"fbx_takes": [b'[{"name": "legacy"}]', b""]})
        self.assertEqual(GlbPipeline._declared_takes(fbx), {"legacy"})
        self.assertEqual(GlbPipeline._declared_takes(self._Fbx({})), set())


if __name__ == "__main__":
    unittest.main()
