#!/usr/bin/python
# coding=utf-8
"""Unit tests for pythontk Ktx2Encoder (toktx CLI wrapper).

Argument assembly, discovery, and error shaping are covered WITHOUT the
binary (the argv is the contract, so it is asserted directly); one
integration test runs a real encode and is skipped when toktx is absent.

Run with:
    python -m pytest test_ktx2_encoder.py -v
    python test_ktx2_encoder.py
"""

import contextlib
import os
import shutil
import struct
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from pythontk import Ktx2Encoder

from conftest import BaseTestCase


class _TempDirTestCase(BaseTestCase):
    """Per-test scratch dir under ``test/temp_tests/`` (repo convention)."""

    def setUp(self):
        super().setUp()
        self.out_dir = os.path.join(
            os.path.dirname(__file__), "temp_tests", f"ktx2_{self._testMethodName}"
        )
        os.makedirs(self.out_dir, exist_ok=True)
        self.addCleanup(shutil.rmtree, self.out_dir, ignore_errors=True)


class Ktx2ManagedInstallIntegrityTest(BaseTestCase):
    """The managed install must never run unverified bytes.

    Windows is typed ``nsis`` -- the one AppInstaller type that RUNS the download
    rather than unpacking it, and re-runs it through ShellExecuteExW('runas') on
    WinError 740. Consent is obtained for KTX-Software; without a pinned digest,
    what executes as Administrator was never confirmed to BE KTX-Software.
    """

    def test_every_platform_entry_has_a_pinned_digest(self):
        from pythontk.img_utils.ktx2_encoder import (
            KTX_SOFTWARE_PLATFORMS,
            KTX_SOFTWARE_SHA256,
        )

        self.assertEqual(
            set(KTX_SOFTWARE_PLATFORMS),
            set(KTX_SOFTWARE_SHA256),
            "a platform can be installed with nothing to verify it against",
        )
        for plat, digest in KTX_SOFTWARE_SHA256.items():
            with self.subTest(platform=plat):
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_resolve_toktx_passes_the_digests_to_the_installer(self):
        """A digest table nothing hands to ensure() verifies nothing."""
        from pythontk.core_utils import app_installer as ai
        from pythontk.img_utils import ktx2_encoder as mod

        seen = {}

        def fake_ensure(name, **kw):
            seen.update(kw)
            return "/tools/toktx"

        with (
            # drive discovery to empty so the managed install is reached
            mock.patch.object(mod.shutil, "which", return_value=None),
            mock.patch.object(mod.Ktx2Encoder, "_WINDOWS_INSTALL_PATHS", ()),
            mock.patch.object(
                ai.AppInstaller, "get_path", staticmethod(lambda *a, **k: None)
            ),
            mock.patch.object(ai.AppInstaller, "ensure", staticmethod(fake_ensure)),
        ):
            mod.Ktx2Encoder.resolve_toktx(auto_install=True, prompt=False)

        self.assertEqual(seen.get("sha256"), mod.KTX_SOFTWARE_SHA256)


class Ktx2EncoderArgsTest(BaseTestCase):
    """The argv contract — flag surface per codec/colorspace/quality."""

    def _encoder(self, **kw):
        # Explicit binary path bypasses discovery, so these run everywhere.
        return Ktx2Encoder(toktx="toktx-test-bin", **kw)

    def test_uastc_defaults(self):
        args = self._encoder().args_for("in.png", "out.ktx2")
        self.assertEqual(args[0], "toktx-test-bin")
        self.assertIn("--t2", args)
        self.assertEqual(args[args.index("--encode") + 1], "uastc")
        self.assertIn("--genmipmap", args)
        self.assertEqual(args[args.index("--assign_oetf") + 1], "srgb")
        self.assertEqual(args[args.index("--uastc_quality") + 1], "2")
        self.assertEqual(args[args.index("--zcmp") + 1], "18")
        self.assertNotIn("--qlevel", args)
        # File order is toktx's contract: output first, then input.
        self.assertEqual(args[-2:], ["out.ktx2", "in.png"])

    def test_etc1s_defaults(self):
        args = self._encoder().args_for("in.png", "out.ktx2", codec="ETC1S")
        self.assertEqual(args[args.index("--encode") + 1], "etc1s")
        self.assertEqual(args[args.index("--qlevel") + 1], "128")
        self.assertEqual(args[args.index("--clevel") + 1], "2")
        # BasisLZ is ETC1S's own supercompression; zstd is the UASTC pairing.
        self.assertNotIn("--zcmp", args)
        self.assertNotIn("--uastc_quality", args)

    def test_linear_label(self):
        args = self._encoder().args_for("in.png", "out.ktx2", srgb=False)
        self.assertEqual(args[args.index("--assign_oetf") + 1], "linear")

    def test_no_mipmaps(self):
        args = self._encoder().args_for("in.png", "out.ktx2", mipmaps=False)
        self.assertNotIn("--genmipmap", args)

    def test_quality_maps_to_etc1s_qlevel(self):
        enc = self._encoder()
        for quality, expected in ((1, 3), (50, 128), (100, 255)):
            args = enc.args_for("i.png", "o.ktx2", codec="ETC1S", quality=quality)
            self.assertEqual(args[args.index("--qlevel") + 1], str(expected))

    def test_quality_ignored_for_uastc(self):
        args = self._encoder().args_for("i.png", "o.ktx2", codec="UASTC", quality=40)
        self.assertNotIn("--qlevel", args)

    def test_unknown_codec_raises(self):
        with self.assertRaises(ValueError):
            self._encoder().args_for("i.png", "o.ktx2", codec="DXT5")

    def test_extra_args_ride_before_files(self):
        args = self._encoder(extra_args=("--uastc_rdo_l", "0.5")).args_for(
            "in.png", "out.ktx2"
        )
        self.assertLess(args.index("--uastc_rdo_l"), args.index("out.ktx2"))


class Ktx2EncoderRunTest(_TempDirTestCase):
    """encode() behavior with the subprocess mocked out."""

    def _run_capture(self, returncode=0, stderr=""):
        result = mock.Mock(returncode=returncode, stderr=stderr, stdout="")
        return mock.patch(
            "pythontk.img_utils.ktx2_encoder.subprocess.run", return_value=result
        )

    def test_encode_invokes_toktx(self):
        enc = Ktx2Encoder(toktx="toktx-test-bin")
        out = os.path.join(self.out_dir, "map.ktx2")
        with self._run_capture() as run:
            self.assertEqual(enc.encode("src.png", out, codec="ETC1S"), out)
        argv = run.call_args[0][0]
        self.assertEqual(argv[0], "toktx-test-bin")
        self.assertEqual(argv[-2:], [out, "src.png"])

    def test_encode_failure_carries_stderr_tail(self):
        enc = Ktx2Encoder(toktx="toktx-test-bin")
        with self._run_capture(returncode=1, stderr="boom\nlast line"):
            with self.assertRaises(RuntimeError) as ctx:
                enc.encode("src.png", os.path.join(self.out_dir, "x.ktx2"))
        self.assertIn("last line", str(ctx.exception))

    def test_hung_toktx_times_out_instead_of_blocking_forever(self):
        """A hung `toktx` must not block a DCC worker thread forever — the
        subprocess is given a timeout and a TimeoutExpired is re-raised as
        the same fix-shaped RuntimeError style `_run` already uses for a
        non-zero exit."""
        import subprocess

        enc = Ktx2Encoder(toktx="toktx-test-bin", timeout=5)
        with mock.patch(
            "pythontk.img_utils.ktx2_encoder.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="toktx-test-bin", timeout=5),
        ) as run:
            with self.assertRaises(RuntimeError) as ctx:
                enc.encode("src.png", os.path.join(self.out_dir, "x.ktx2"))
        self.assertIn("timed out", str(ctx.exception).lower())
        self.assertEqual(run.call_args.kwargs.get("timeout"), 5)

    def test_default_timeout_is_passed_to_subprocess_run(self):
        enc = Ktx2Encoder(toktx="toktx-test-bin")
        with self._run_capture() as run:
            enc.encode("src.png", os.path.join(self.out_dir, "map.ktx2"))
        self.assertEqual(run.call_args.kwargs.get("timeout"), 300)

    def test_staged_modes_match_the_fallback_table(self):
        """Every ktx2 row of ``_CONTAINER_MODE_FALLBACKS``, measured where it
        actually applies — the scratch PNG handed to toktx. (The Pillow rows
        of that table are measured in test_img.py; these are the encoder's.)
        """
        from pythontk import ImgUtils

        enc = Ktx2Encoder(toktx="toktx-test-bin")
        for mode, expected in ImgUtils._CONTAINER_MODE_FALLBACKS["ktx2"].items():
            staged_modes = []

            def fake_run(argv, **kw):
                source = argv[-1]
                self.assertTrue(os.path.isfile(source), "staged PNG must exist at call")
                staged_modes.append(Image.open(source).mode)
                return mock.Mock(returncode=0, stderr="", stdout="")

            with mock.patch(
                "pythontk.img_utils.ktx2_encoder.subprocess.run",
                side_effect=fake_run,
            ):
                enc.encode(
                    Image.new(mode, (8, 8)),
                    os.path.join(self.out_dir, f"m_{mode.replace(';', '')}.ktx2"),
                )
            self.assertEqual(staged_modes, [expected], f"{mode} -> ktx2")

    def test_late_provisioned_pillow_still_stages_a_pil_image(self):
        """A stale ``PIL_AVAILABLE`` must not send a live image down the
        path branch.

        In Blender, ``import pythontk`` runs before ``ensure_image_deps()``
        can provision Pillow, so this module caches ``Image = None``.
        blendertk's ``_rebind_pil_globals`` repairs it afterwards -- but its
        own docstring scopes that to "a name the module itself set to
        ``None``", so the ``PIL_AVAILABLE`` bool stays False forever. This
        pins the post-repair state: bool stale, ``Image`` live.
        """
        import pythontk.img_utils.ktx2_encoder as mod

        enc = Ktx2Encoder(toktx="toktx-test-bin")
        seen = []

        def fake_run(argv, **kw):
            source = argv[-1]
            seen.append((source, os.path.isfile(source)))
            return mock.Mock(returncode=0, stderr="", stdout="")

        with mock.patch.object(mod, "PIL_AVAILABLE", False), mock.patch(
            "pythontk.img_utils.ktx2_encoder.subprocess.run", side_effect=fake_run
        ):
            enc.encode(
                Image.new("RGB", (8, 8)),
                os.path.join(self.out_dir, "late_provisioned.ktx2"),
            )

        self.assertEqual(len(seen), 1)
        source, existed = seen[0]
        self.assertTrue(
            existed and source.lower().endswith(".png"),
            f"toktx was handed {source!r} where a staged PNG path belongs",
        )

    def test_sixteen_bit_source_is_rescaled_not_clipped(self):
        """A 16-bit source must be RANGE-REDUCED to 8-bit, not clipped.

        Pillow implements ``I;16`` -> ``L`` as a clip at 255, so the plain
        ``convert`` this used to do turned a smooth 0..65535 ramp into two
        values, 99.6% of them pure white — a destroyed map handed to toktx
        under a log line that called it a precision reduction.
        """
        enc = Ktx2Encoder(toktx="toktx-test-bin")
        ramp = Image.new("I;16", (256, 1))
        pixels = ramp.load()
        for x in range(256):
            pixels[x, 0] = x * 257  # full-scale 0..65535 ramp
        staged = os.path.join(self.out_dir, "ramp.png")
        enc._stage_image(ramp, staged)

        with Image.open(staged) as out:
            self.assertEqual(out.mode, "L")
            values = list(out.get_flattened_data())
        self.assertEqual(len(set(values)), 256, "the ramp was clipped, not rescaled")
        self.assertEqual((values[0], values[-1]), (0, 255))

    def test_thirty_two_bit_int_source_is_reduced_too(self):
        """Mode "I" is the same trap one width up, and it is absent from the
        ktx2 fallback table — so the mode-only lookup left it untouched and
        staged a 16-bit PNG for an encoder that is 8-bit LDR."""
        enc = Ktx2Encoder(toktx="toktx-test-bin")
        ramp = Image.new("I", (256, 1))
        pixels = ramp.load()
        for x in range(256):
            pixels[x, 0] = x * 257
        staged = os.path.join(self.out_dir, "ramp_i32.png")
        enc._stage_image(ramp, staged)

        with Image.open(staged) as out:
            self.assertEqual(out.mode, "L")
            self.assertEqual(len(set(out.get_flattened_data())), 256)

    def test_big_endian_sixteen_bit_source_is_reduced_not_refused(self):
        """``I;16B`` is what Pillow hands back for a big-endian 16-bit TIFF,
        and ``Image.point`` does not support byte-order-qualified modes -- so
        gating the rescale on ``startswith("I;")`` and calling ``point``
        turned a working (if clipping) convert into a hard ValueError."""
        enc = Ktx2Encoder(toktx="toktx-test-bin")
        ramp = np.array([x * 257 for x in range(256)], dtype=np.uint16).reshape(1, 256)
        source = Image.frombytes("I;16B", (256, 1), ramp.astype(">u2").tobytes())
        staged = os.path.join(self.out_dir, "ramp_be.png")
        enc._stage_image(source, staged)

        with Image.open(staged) as out:
            self.assertEqual(out.mode, "L")
            # Rescaled, not clipped: a full-scale ramp keeps all 256 steps.
            self.assertEqual(len(set(out.get_flattened_data())), 256)

    def test_palettised_source_keeps_its_transparency(self):
        """``effective_mode`` is mode-only, so it maps every "P" to "RGB" and
        destroys tRNS alpha. Both Basis codecs carry alpha, so nothing forces
        that drop — ``ImgUtils.depalettize_image`` already owns the rule."""
        from pythontk import ImgUtils

        enc = Ktx2Encoder(toktx="toktx-test-bin")
        palettised = Image.new("P", (8, 8))
        palettised.putpalette([v for i in range(256) for v in (i, i, i)])
        palettised.info["transparency"] = 0

        staged = os.path.join(self.out_dir, "pal.png")
        enc._stage_image(palettised, staged)
        with Image.open(staged) as out:
            self.assertEqual(
                out.mode,
                ImgUtils.depalettize_image(palettised).mode,
                "staging must reuse depalettize_image's tRNS rule",
            )
            self.assertEqual(out.mode, "RGBA")

    def test_float_source_is_refused_by_name(self):
        """A float (EXR/HDR) source must be named, not clipped or crashed.

        ``F`` is deliberately absent from the ktx2 fallback table. An integer
        source has a known full-scale range, so staging RESCALES it into 0-255
        with the image intact (pinned above); ``F`` carries no such range, and
        PIL's ``F`` -> ``L`` clips to 0-255, which flattens the 0..1 data an
        EXR carries to black/white. Before the guard this reached
        ``Image.save`` and surfaced as ``OSError: cannot write mode F as PNG``.
        """
        from pythontk import ImgUtils

        self.assertNotIn(
            "F",
            ImgUtils._CONTAINER_MODE_FALLBACKS["ktx2"],
            "a float row here would mean silent clipping",
        )
        enc = Ktx2Encoder(toktx="toktx-test-bin")
        with self.assertRaises(ValueError) as ctx:
            enc._stage_image(
                Image.new("F", (8, 8)), os.path.join(self.out_dir, "f.png")
            )
        message = str(ctx.exception)
        self.assertIn("8-bit LDR", message)
        self.assertIn("Tonemap", message)

    def test_missing_binary_error_names_the_fix(self):
        with (
            mock.patch("shutil.which", return_value=None),
            mock.patch("os.path.isfile", return_value=False),
            mock.patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=None,
            ),
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                Ktx2Encoder.resolve_toktx(required=True)
            self.assertFalse(Ktx2Encoder.available())
        message = str(ctx.exception)
        self.assertIn("KTX-Software", message)
        self.assertIn("register_ktx2_encoder", message)

    # -- auto_install: the managed download behind a consent seam ---------

    @contextlib.contextmanager
    def _no_toktx(self):
        """Discovery finds nothing: PATH, conventional paths, catalog."""
        with (
            mock.patch("shutil.which", return_value=None),
            mock.patch("os.path.isfile", return_value=False),
            mock.patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=None,
            ),
        ):
            yield

    def test_auto_install_downloads_through_app_installer(self):
        """``auto_install=True, prompt=False`` hands the managed install to
        AppInstaller under the tool name the catalog probe reads back, with
        the Windows release typed as the NSIS installer it is."""
        from pythontk.core_utils.app_installer import AppInstaller
        from pythontk.img_utils.ktx2_encoder import (
            KTX_SOFTWARE_PLATFORMS,
            KTX_SOFTWARE_VERSION,
        )

        installed = os.path.join(self.out_dir, "bin", "toktx.exe")
        with (
            self._no_toktx(),
            mock.patch.object(AppInstaller, "ensure", return_value=installed) as ensure,
        ):
            path = Ktx2Encoder.resolve_toktx(
                required=True, auto_install=True, prompt=False
            )
        self.assertEqual(path, installed)
        ensure.assert_called_once()
        self.assertEqual(ensure.call_args.args[0], Ktx2Encoder.TOOL_NAME)
        kwargs = ensure.call_args.kwargs
        self.assertIs(kwargs["platforms"], KTX_SOFTWARE_PLATFORMS)
        self.assertEqual(kwargs["executable"], "toktx")
        self.assertEqual(kwargs["version"], KTX_SOFTWARE_VERSION)
        self.assertEqual(KTX_SOFTWARE_PLATFORMS["windows"]["type"], "nsis")
        self.assertIn(KTX_SOFTWARE_VERSION, KTX_SOFTWARE_PLATFORMS["windows"]["url"])

    def test_auto_install_asks_the_prompt_and_honours_a_decline(self):
        """A callable *prompt* is the consent seam a GUI plugs its dialog
        into: it is asked a question naming the tool, and a "no" raises the
        fix-shaped error WITHOUT touching the network."""
        from pythontk.core_utils.app_installer import AppInstaller

        asked = []

        def decline(question):
            asked.append(question)
            return False

        with self._no_toktx(), mock.patch.object(AppInstaller, "ensure") as ensure:
            with self.assertRaises(FileNotFoundError) as ctx:
                Ktx2Encoder.resolve_toktx(
                    required=True, auto_install=True, prompt=decline
                )
            self.assertIsNone(
                Ktx2Encoder.resolve_toktx(
                    required=False, auto_install=True, prompt=decline
                )
            )
        ensure.assert_not_called()
        self.assertEqual(len(asked), 2)
        self.assertIn("KTX-Software", asked[0])
        self.assertIn("KTX-Software", str(ctx.exception))

    def test_auto_install_failure_is_the_fix_shaped_error(self):
        """A failed download/install surfaces as FileNotFoundError naming the
        manual install source, so a caller's one except clause covers both
        "not found" and "could not be fetched"."""
        from pythontk.core_utils.app_installer import AppInstaller

        with (
            self._no_toktx(),
            mock.patch.object(
                AppInstaller,
                "ensure",
                side_effect=RuntimeError("Download failed for x"),
            ),
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                Ktx2Encoder.resolve_toktx(
                    required=True, auto_install=True, prompt=False
                )
        self.assertIn("Download failed", str(ctx.exception))
        self.assertIn("KTX-Software", str(ctx.exception))


class Ktx2HeaderTest(_TempDirTestCase):
    """The fixed-layout KTX 2.0 header — readable without a transcoder.

    ``MapOptimizer.assess`` gets pointed at the staged output an encode just
    wrote, and PIL cannot open a Basis payload at all; the geometry the dry run
    needs sits at a fixed offset in every valid file.
    """

    @staticmethod
    def _header(width, height, vk_format=0, type_size=1, depth=0):
        return Ktx2Encoder.KTX2_MAGIC + struct.pack(
            "<5I", vk_format, type_size, width, height, depth
        )

    def test_read_header_reports_the_encoded_geometry(self):
        path = os.path.join(self.out_dir, "h.ktx2")
        with open(path, "wb") as fh:
            fh.write(self._header(512, 256) + bytes(64))
        header = Ktx2Encoder.read_header(path)
        self.assertEqual((header["width"], header["height"]), (512, 256))
        # Basis payloads are supercompressed: vkFormat is UNDEFINED and the
        # real format is the transcode target chosen at load time.
        self.assertEqual(header["vk_format"], 0)
        self.assertEqual(header["depth"], 0)

    def test_read_header_rejects_a_non_ktx2_file(self):
        path = os.path.join(self.out_dir, "not.ktx2")
        with open(path, "wb") as fh:
            fh.write(bytes.fromhex("89504e470d0a1a0a") + bytes(64))
        with self.assertRaises(ValueError) as ctx:
            Ktx2Encoder.read_header(path)
        self.assertIn("KTX2", str(ctx.exception))

    def test_read_header_rejects_a_truncated_header(self):
        path = os.path.join(self.out_dir, "short.ktx2")
        with open(path, "wb") as fh:
            fh.write(Ktx2Encoder.KTX2_MAGIC + bytes(4))
        with self.assertRaises(ValueError) as ctx:
            Ktx2Encoder.read_header(path)
        self.assertIn("runcated", str(ctx.exception))


@unittest.skipUnless(Ktx2Encoder.available(), "toktx not installed")
class Ktx2EncoderIntegrationTest(_TempDirTestCase):
    """Real toktx round-trip — runs only where KTX-Software is installed."""

    def test_real_encode_writes_ktx2_magic(self):
        out = os.path.join(self.out_dir, "real.ktx2")
        Ktx2Encoder().encode(
            Image.new("RGB", (16, 16), (128, 64, 32)), out, codec="ETC1S"
        )
        with open(out, "rb") as fh:
            self.assertEqual(fh.read(12), Ktx2Encoder.KTX2_MAGIC)

    def test_real_uastc_linear(self):
        out = os.path.join(self.out_dir, "n.ktx2")
        Ktx2Encoder().encode(
            Image.new("RGB", (16, 16), (127, 127, 255)),
            out,
            codec="UASTC",
            srgb=False,
        )
        with open(out, "rb") as fh:
            self.assertEqual(fh.read(12), Ktx2Encoder.KTX2_MAGIC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
