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

    def test_no_texture_params_means_the_shared_web_delivery_policy(self):
        self._build(self._stub_fbx())
        _, _glb, params = self.calls[-1]
        policy = MeshConvert.web_delivery_texture_params()
        self.assertEqual(params, policy, "the optimizer's own defaults apply")

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


if __name__ == "__main__":
    unittest.main()
