# !/usr/bin/python
# coding=utf-8
"""Unit tests for MeshConvert.

Network-free — FBX2glTF resolution and subprocess invocation are mocked.
An opt-in integration test triggers a real install when
``PYTHONTK_INTEGRATION_TESTS=1``; ``TestGrayscaleMapsThroughTheConverter``
runs an FBX2glTF that is already installed (it downloads nothing) and skips
otherwise.

Run with:
    python -m pytest test_mesh_convert.py -v
    python test_mesh_convert.py
"""

import copy
import errno
import hashlib
import io
import json
import logging
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
import zlib
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk import (
    FileUtils,
    GlbFades,
    GlbReader,
    HandoffRecord,
    ImgUtils,
    Ktx2Encoder,
    LightmapRecord,
    MeshConvert,
    SceneRecords,
)
from pythontk.core_utils.engines.scene_export.lightmap_record import LightmapEntry
from pythontk.file_utils.mesh_convert.fbx_file import FbxFile
from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia
from test_fbx_media import build_stingray_fbx, encoded
from test_glb_tangents import read_tangents, tangent_quads_glb
from pythontk.core_utils.engines.scene_export.material_record import MaterialFamilies


def _require_cv2():
    """``cv2``, or skip the test: an optional dependency (CI installs none),
    which only writing or reading a bake's maps needs."""
    try:
        import cv2
    except ImportError:
        raise unittest.SkipTest("cv2 unavailable; a bake's maps cannot be written or read")
    return cv2



def flat_material(record, material):
    """A record's (or a GLB's) glTF *material* flat: its factors, its maps by
    slot and file name, its emission whole, ``unlit`` and the strength --
    what a test reads of a material."""
    pbr = material.get("pbrMetallicRoughness") or {}
    extensions = material.get("extensions") or {}
    found = {
        "name": material.get("name"),
        "baseColorFactor": list(pbr.get("baseColorFactor") or [1, 1, 1, 1]),
        "metallicFactor": pbr.get("metallicFactor", 1.0),
        "roughnessFactor": pbr.get("roughnessFactor", 1.0),
        "emissiveFactor": MaterialFamilies.emission(material),
        "alphaMode": material.get("alphaMode", "OPAQUE"),
    }
    found.update(MaterialFamilies.material_files(record, material))
    if "alphaCutoff" in material:
        found["alphaCutoff"] = material["alphaCutoff"]
    if MaterialFamilies.UNLIT_EXTENSION in extensions:
        found["unlit"] = True
    strength = extensions.get(MaterialFamilies.STRENGTH_EXTENSION)
    if strength:
        found["emissiveStrength"] = strength["emissiveStrength"]
    return found


def stated_record(maps, *statements):
    """``(record, maps)``: *statements* (flat, maps by file name) built into
    the record, each name in *maps* read from its path there and a metal or
    roughness pair packed beside it (``MaterialFamilies.pack_metallic_roughness``);
    *maps* returned with the packed images added."""
    maps = dict(maps)
    resolved = []
    for statement in statements:
        statement = dict(statement)
        for field in MaterialFamilies.TEXTURE_FIELDS:
            if statement.get(field) in maps:
                statement[field] = maps[statement[field]]
        resolved.append(statement)

    def pack(*sources, name=None):
        try:
            packed = MaterialFamilies.pack_metallic_roughness(*sources, name=name)
        except FileNotFoundError:
            packed = None
        if packed:
            maps[os.path.basename(packed)] = packed
        return packed if packed else name or MaterialFamilies.packed_name(*sources)

    record = MaterialFamilies.payload(resolved, pack=pack)
    return record, maps


def assert_materials_equal(case, edit, record):
    """Every material *record* names is, in the open GLB *edit*, the record's:
    the same JSON -- factors, alpha, sidedness, extensions, each map's
    transform -- with each map the GLB image of the file the record names (by
    name; the build may re-encode the bytes)."""

    def resolved(material, image_of):
        found = copy.deepcopy(material)
        found.pop("extras", None)
        for container, slot in MaterialFamilies.TEXTURE_SLOTS:
            holder = found.get(container) if container else found
            if isinstance(holder, dict) and isinstance(holder.get(slot), dict):
                holder[slot]["index"] = image_of(holder[slot]["index"])
        return found

    def glb_image(index):
        label = edit.image_label(edit.image_for_texture(index))
        return os.path.splitext(os.path.basename(label))[0].casefold()

    def record_image(index):
        return os.path.splitext(MaterialFamilies.texture_file(record, index))[0].casefold()

    by_name = {m.get("name"): m for m in edit.materials}
    for material in record["materials"]:
        with case.subTest(material=material["name"]):
            case.assertIn(material["name"], by_name)
            case.assertEqual(
                resolved(by_name[material["name"]], glb_image),
                resolved(material, record_image),
            )
    used = set(edit.gltf.get("extensionsUsed") or ())
    case.assertTrue(set(record.get("extensionsUsed") or ()) <= used, used)

class TestResolveBinary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_returns_path_when_on_system_path(self):
        with patch("shutil.which", return_value="/usr/bin/FBX2glTF"):
            self.assertEqual(MeshConvert.resolve_binary(), "/usr/bin/FBX2glTF")

    def test_returns_managed_path_when_in_catalog(self):
        managed = os.path.join(self.tmp, "FBX2glTF.exe")
        with (
            patch("shutil.which", return_value=None),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=managed,
            ),
        ):
            self.assertEqual(MeshConvert.resolve_binary(), managed)

    def test_raises_when_missing_and_required(self):
        with (
            patch("shutil.which", return_value=None),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=None,
            ),
        ):
            with self.assertRaises(FileNotFoundError):
                MeshConvert.resolve_binary(required=True, auto_install=False)

    def test_returns_none_when_missing_and_not_required(self):
        with (
            patch("shutil.which", return_value=None),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=None,
            ),
        ):
            self.assertIsNone(
                MeshConvert.resolve_binary(required=False, auto_install=False)
            )

    def test_no_tty_with_prompt_refuses_install(self):
        """prompt=True without a TTY should NOT silently install."""
        with (
            patch("shutil.which", return_value=None),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=None,
            ),
            patch("pythontk.core_utils.app_installer.AppInstaller.ensure") as ensure,
            patch("sys.stdin") as stdin,
            # consent() writes the question to stdout; without this the suite
            # prints a live-looking "Download ... [y/N]" prompt mid-run, which
            # reads as the tests asking for input when they are not.
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            stdin.isatty.return_value = False
            with self.assertRaises(FileNotFoundError) as cm:
                MeshConvert.resolve_binary(
                    auto_install=True, prompt=True, required=True
                )
            self.assertIn("interactive", str(cm.exception).lower())
            ensure.assert_not_called()

    def test_no_tty_without_prompt_installs_silently(self):
        """prompt=False allows non-interactive install (CI/automation)."""
        installed = os.path.join(self.tmp, "FBX2glTF.exe")
        with (
            patch("shutil.which", return_value=None),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=None,
            ),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.ensure",
                return_value=installed,
            ) as ensure,
            patch("sys.stdin") as stdin,
            # consent() writes the question to stdout; without this the suite
            # prints a live-looking "Download ... [y/N]" prompt mid-run, which
            # reads as the tests asking for input when they are not.
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            stdin.isatty.return_value = False
            result = MeshConvert.resolve_binary(auto_install=True, prompt=False)
            self.assertEqual(result, installed)
            ensure.assert_called_once()

    def test_prompt_decline_raises_when_required(self):
        with (
            patch("shutil.which", return_value=None),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=None,
            ),
            patch("sys.stdin") as stdin,
            # consent() writes the question to stdout; without this the suite
            # prints a live-looking "Download ... [y/N]" prompt mid-run, which
            # reads as the tests asking for input when they are not.
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            stdin.isatty.return_value = True
            stdin.readline.return_value = "n\n"
            with self.assertRaises(FileNotFoundError):
                MeshConvert.resolve_binary(
                    auto_install=True, prompt=True, required=True
                )

    def test_prompt_accept_triggers_install(self):
        installed = os.path.join(self.tmp, "FBX2glTF.exe")
        with (
            patch("shutil.which", return_value=None),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.get_path",
                return_value=None,
            ),
            patch(
                "pythontk.core_utils.app_installer.AppInstaller.ensure",
                return_value=installed,
            ) as ensure,
            patch("sys.stdin") as stdin,
            # consent() writes the question to stdout; without this the suite
            # prints a live-looking "Download ... [y/N]" prompt mid-run, which
            # reads as the tests asking for input when they are not.
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            stdin.isatty.return_value = True
            stdin.readline.return_value = "y\n"
            result = MeshConvert.resolve_binary(auto_install=True, prompt=True)
            self.assertEqual(result, installed)
            ensure.assert_called_once()

    def test_platform_exe_name_known(self):
        name = MeshConvert._platform_exe_name()
        self.assertTrue(name.startswith("FBX2glTF"))


class TestFbxToGlb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = os.path.join(self.tmp, "model.fbx")
        with open(self.src, "wb") as fh:
            fh.write(b"fake-fbx")
        self.fake_bin = os.path.join(self.tmp, "FBX2glTF.exe")

    def test_missing_src_raises(self):
        with self.assertRaises(FileNotFoundError):
            MeshConvert.fbx_to_glb(os.path.join(self.tmp, "nope.fbx"))

    def test_wrong_extension_raises(self):
        bad = os.path.join(self.tmp, "model.obj")
        with open(bad, "wb") as fh:
            fh.write(b"")
        with self.assertRaises(ValueError):
            MeshConvert.fbx_to_glb(bad)

    def test_existing_dst_without_overwrite_raises(self):
        dst = os.path.join(self.tmp, "model.glb")
        with open(dst, "wb") as fh:
            fh.write(b"existing")
        with self.assertRaises(FileExistsError):
            MeshConvert.fbx_to_glb(self.src, dst, overwrite=False)

    def _run_simulator(self, captured):
        """Return a subprocess.run replacement that records the cmd and
        creates the expected .glb at <output_base>.glb."""

        def _run(cmd, **kw):
            captured["cmd"] = cmd
            captured["kwargs"] = kw
            # FBX2glTF writes to <output_base>.glb
            output_base_idx = cmd.index("-o") + 1
            with open(cmd[output_base_idx] + ".glb", "wb") as fh:
                fh.write(b"glb-bytes")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        return _run

    def test_a_failed_conversion_keeps_the_glb_it_would_replace(self):
        """The previous GLB survives a conversion that never finished.

        ``overwrite`` deleted it BEFORE FBX2glTF ran -- 13 minutes on a
        production assembly -- so a conversion that failed or timed out also
        cost the file it was to replace, and in a synced export folder the
        deletion reached every copy. Seen 2026-09-14: a re-export's GLB left
        the folder the moment its conversion began.
        """
        dst = os.path.join(self.tmp, "model.glb")

        def failed(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

        def timed_out(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 5)

        for outcome in (failed, timed_out):
            with self.subTest(outcome=outcome.__name__):
                with open(dst, "wb") as fh:
                    fh.write(b"previous")
                with (
                    patch.object(
                        MeshConvert, "resolve_binary", return_value=self.fake_bin
                    ),
                    patch("subprocess.run", side_effect=outcome),
                    self.assertRaises(RuntimeError),
                ):
                    MeshConvert.fbx_to_glb(
                        self.src, dst, overwrite=True, auto_install=False
                    )
                with open(dst, "rb") as fh:
                    self.assertEqual(fh.read(), b"previous")
        captured = {}
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=self._run_simulator(captured)),
        ):
            MeshConvert.fbx_to_glb(self.src, dst, overwrite=True, auto_install=False)
        with open(dst, "rb") as fh:
            self.assertEqual(fh.read(), b"glb-bytes", "a finished one replaces it")
        store = os.path.dirname(captured["cmd"][captured["cmd"].index("-o") + 1])
        self.assertFalse(os.path.exists(store), "the conversion's store outlived it")

    def test_a_locked_destination_fails_before_the_conversion(self):
        """A GLB another process holds open cannot be replaced (Windows refuses
        while a viewer or a sync client has it), so that is found out BEFORE
        the conversion -- 13 minutes on a production assembly -- rather than
        at the move that ends it and would throw its output away."""
        dst = os.path.join(self.tmp, "model.glb")
        with open(dst, "wb") as fh:
            fh.write(b"previous")
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch.object(FileUtils, "is_locked", return_value=True),
            patch("subprocess.run") as run,
            self.assertRaises(PermissionError),
        ):
            MeshConvert.fbx_to_glb(self.src, dst, overwrite=True, auto_install=False)
        run.assert_not_called()
        with open(dst, "rb") as fh:
            self.assertEqual(fh.read(), b"previous")

    def test_a_failed_copy_keeps_the_glb_it_would_replace(self):
        """A copy that fails part way leaves the previous GLB whole.

        The conversion's store sits in the system temp dir -- usually another
        volume, where a move is a copy -- and on Windows a rename refuses an
        existing destination on any volume, so the move that ends a conversion
        copied straight over *dst*: a disk-full or I/O error part way through a
        multi-hundred-MB GLB left a truncated deliverable where the previous one
        had been. Pinned to the cross-volume branch (the rename refused with
        EXDEV), so the test means the same on every machine.
        Added: 2026-09-15
        """
        dst = os.path.join(self.tmp, "model.glb")
        with open(dst, "wb") as fh:
            fh.write(b"previous")

        def disk_full(src, target, *args, **kwargs):
            # ENOSPC as the writer meets it: part of the file lands, then the
            # write fails.
            with open(src, "rb") as reader, open(target, "wb") as writer:
                writer.write(reader.read(3))
            raise OSError(errno.ENOSPC, "No space left on device")

        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=self._run_simulator({})),
            patch("os.rename", side_effect=OSError(errno.EXDEV, "Cross-device link")),
            patch("shutil.copyfile", side_effect=disk_full),
            self.assertRaises(OSError),
        ):
            MeshConvert.fbx_to_glb(self.src, dst, overwrite=True, auto_install=False)
        with open(dst, "rb") as fh:
            self.assertEqual(fh.read(), b"previous")
        self.assertEqual(
            sorted(os.listdir(self.tmp)),
            ["model.fbx", "model.glb"],
            "the failed copy left a partial file beside the GLB",
        )

    def test_the_sdk_extracts_embedded_media_into_a_store_of_its_own(self):
        """FBX2glTF's SDK import extracts every embedded texture into
        ``<stem>.fbm`` BESIDE the FBX it reads, and nothing removes it.

        Measured on a production assembly: 311 MB in the synced export folder,
        and 311 MB per export in the temp dir once the GLB pipeline converted a
        scratch copy. ``--fbx-temp-dir`` sends the extraction elsewhere (probed
        on v0.13.1 with a Maya-written textured FBX: nothing beside the input,
        the image still embedded), and the conversion owns that directory.
        Added: 2026-09-12
        """
        captured = {}
        simulate = self._run_simulator(captured)

        def _run(cmd, **kw):
            if "--fbx-temp-dir" in cmd:  # the SDK's extraction, as probed
                media = os.path.join(cmd[cmd.index("--fbx-temp-dir") + 1], "2DC020BA")
                os.makedirs(os.path.join(media, "model.fbm"))
                with open(os.path.join(media, "model.fbm", "wall.png"), "wb") as fh:
                    fh.write(b"png")
            return simulate(cmd, **kw)

        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=_run),
        ):
            MeshConvert.fbx_to_glb(self.src, auto_install=False)
        cmd = captured["cmd"]
        self.assertIn("--fbx-temp-dir", cmd)
        media = cmd[cmd.index("--fbx-temp-dir") + 1]
        self.assertFalse(os.path.exists(media), "the extraction outlived the call")
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "model.fbm")))

    def test_the_converter_reads_grayscale_embeds_expanded(self):
        """FBX2glTF reads a copy whose grayscale embeds were expanded
        (``FbxMedia.expand_grayscale``), minted for the call and gone after it;
        an expansion that fails costs a warning, never the conversion. What the
        expansion buys is measured in ``TestGrayscaleMapsThroughTheConverter``.
        """
        copies = []

        def expand(src, dst, **_kwargs):
            copies.append(dst)
            with open(dst, "wb") as fh:
                fh.write(b"expanded")
            return {"images": 2, "expanded": 1, "before": 8, "after": 8}

        captured = {}
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch.object(FbxFile, "is_fbx", return_value=True),
            patch.object(FbxMedia, "expand_grayscale", side_effect=expand),
            patch("subprocess.run", side_effect=self._run_simulator(captured)),
        ):
            MeshConvert.fbx_to_glb(self.src, overwrite=True, auto_install=False)
        read = captured["cmd"][captured["cmd"].index("-i") + 1]
        self.assertEqual(read, copies[-1])
        self.assertFalse(os.path.exists(read), "the expanded copy outlived the call")

        captured = {}
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch.object(FbxFile, "is_fbx", return_value=True),
            patch.object(
                FbxMedia,
                "expand_grayscale",
                side_effect=ValueError("unrecognised FBX footer"),
            ),
            patch("subprocess.run", side_effect=self._run_simulator(captured)),
            self.assertLogs(
                "pythontk.file_utils.mesh_convert", level="WARNING"
            ) as logs,
        ):
            MeshConvert.fbx_to_glb(self.src, overwrite=True, auto_install=False)
        read = captured["cmd"][captured["cmd"].index("-i") + 1]
        self.assertEqual(read, os.path.abspath(self.src))
        self.assertTrue(any("unrecognised FBX footer" in line for line in logs.output))

    def test_a_callers_own_fbx_temp_dir_is_used_and_left_alone(self):
        own = os.path.join(self.tmp, "sdk")
        os.makedirs(own)
        # Both spellings the CLI accepts: a second option beside the caller's
        # `--fbx-temp-dir=DIR` is a duplicate the parser can refuse.
        for extra in (["--fbx-temp-dir", own], [f"--fbx-temp-dir={own}"]):
            with self.subTest(extra=extra):
                captured = {}
                with (
                    patch.object(
                        MeshConvert, "resolve_binary", return_value=self.fake_bin
                    ),
                    patch("subprocess.run", side_effect=self._run_simulator(captured)),
                ):
                    MeshConvert.fbx_to_glb(
                        self.src, overwrite=True, auto_install=False, extra_args=extra
                    )
                named = [a for a in captured["cmd"] if a.startswith("--fbx-temp-dir")]
                self.assertEqual(named, extra[:1], "one temp dir, the caller's")
                self.assertTrue(os.path.isdir(own), "a caller's directory is theirs")

    def test_lightmap_dirs_reach_the_applier(self):
        """The conversion must be able to say where the maps live NOW.

        The applier has always taken *search_dirs*; ``fbx_to_glb`` had no way
        to offer them, so every exporter and preview that converts through it
        was stuck with whatever authoring directory the bake recorded. A
        reorganised project then bound nothing, and the deliverable shipped
        unlit -- measured on a delivered room.
        """
        seen = {}

        def _spy(edit, search_dirs=(), hdr_codec=None):
            seen["search_dirs"] = list(search_dirs)
            seen["hdr_codec"] = hdr_codec
            return []

        def _run(cmd, **kw):
            # A REAL (if empty) GLB: the post-process opens the converted file,
            # and the plain-bytes simulator the other tests use never reaches
            # the lightmap pass at all.
            body = json.dumps({"asset": {"version": "2.0"}}).encode()
            body += b" " * ((4 - len(body) % 4) % 4)
            with open(cmd[cmd.index("-o") + 1] + ".glb", "wb") as fh:
                fh.write(
                    struct.pack("<4sII", b"glTF", 2, 20 + len(body))
                    + struct.pack("<I4s", len(body), b"JSON")
                    + body
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=_run),
            patch.object(MeshConvert, "apply_glb_lightmaps", side_effect=_spy),
        ):
            codec = object()
            MeshConvert.fbx_to_glb(
                self.src, lightmap_dirs=["D:/maps", "D:/more"], hdr_codec=codec
            )
        self.assertEqual(seen.get("search_dirs"), ["D:/maps", "D:/more"])
        # ...and so must the host's own float-image codec (Added: 2026-10-17):
        # a GLB built in Blender, which has no cv2, shipped unlit without it.
        self.assertIs(seen.get("hdr_codec"), codec)

    def test_default_dst_derived_from_src(self):
        expected_dst = os.path.join(self.tmp, "model.glb")
        captured = {}
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=self._run_simulator(captured)),
        ):
            result = MeshConvert.fbx_to_glb(self.src, auto_install=False)
            self.assertEqual(result, expected_dst)
            self.assertTrue(os.path.isfile(expected_dst))

    def test_dst_glb_extension_appended_if_missing(self):
        captured = {}
        dst_no_ext = os.path.join(self.tmp, "out")
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=self._run_simulator(captured)),
        ):
            result = MeshConvert.fbx_to_glb(self.src, dst_no_ext, auto_install=False)
            self.assertEqual(result, dst_no_ext + ".glb")

    def test_command_uses_input_output_binary_flags(self):
        dst = os.path.join(self.tmp, "out.glb")
        captured = {}
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=self._run_simulator(captured)),
        ):
            MeshConvert.fbx_to_glb(self.src, dst, auto_install=False)

        cmd = captured["cmd"]
        self.assertEqual(cmd[0], self.fake_bin)
        self.assertIn("-i", cmd)
        self.assertIn("-o", cmd)
        self.assertIn("--binary", cmd)
        # Default-on (measured v0.13.1 + Maya 2025): without it the DataNodes
        # channels deliberately embedded in the FBX are silently dropped.
        self.assertIn("--user-properties", cmd)
        # -o argument must be the output base WITHOUT .glb suffix
        output_base = cmd[cmd.index("-o") + 1]
        self.assertFalse(output_base.lower().endswith(".glb"))

    def test_extra_args_forwarded(self):
        dst = os.path.join(self.tmp, "out.glb")
        captured = {}
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=self._run_simulator(captured)),
        ):
            MeshConvert.fbx_to_glb(
                self.src, dst, auto_install=False, extra_args=["--draco"]
            )
        self.assertIn("--draco", captured["cmd"])

    def test_subprocess_failure_raises(self):
        dst = os.path.join(self.tmp, "out.glb")
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["x"], 1, stdout="", stderr="boom"
                ),
            ),
        ):
            with self.assertRaises(RuntimeError) as cm:
                MeshConvert.fbx_to_glb(self.src, dst, auto_install=False)
            self.assertIn("boom", str(cm.exception))

    def test_subprocess_zero_exit_but_no_output_raises(self):
        dst = os.path.join(self.tmp, "out.glb")
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    ["x"], 0, stdout="", stderr=""
                ),
            ),
        ):
            with self.assertRaises(RuntimeError) as cm:
                MeshConvert.fbx_to_glb(self.src, dst, auto_install=False)
            self.assertIn("not created", str(cm.exception))

    def test_timeout_raises_runtime_error(self):
        dst = os.path.join(self.tmp, "out.glb")
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["x"], timeout=1),
            ),
        ):
            with self.assertRaises(RuntimeError) as cm:
                MeshConvert.fbx_to_glb(self.src, dst, auto_install=False, timeout=1)
            self.assertIn("timed out", str(cm.exception))

    def test_timeout_kwarg_forwarded_to_subprocess(self):
        dst = os.path.join(self.tmp, "out.glb")
        captured = {}
        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=self._run_simulator(captured)),
        ):
            MeshConvert.fbx_to_glb(self.src, dst, auto_install=False, timeout=42)
        self.assertEqual(captured["kwargs"].get("timeout"), 42)

    def test_the_input_is_parsed_once_for_the_budget_and_the_effect_curves(self):
        """The budget's census and the render effects' curves each parsed the
        input for the same curves -- 8 s apiece on a 133 MB Maya assembly
        (2026-10-04 review). One parse serves both: the effect reader is
        handed it (a file with no curve proxy parses no further)."""
        from test_export_verify import build_take_fbx

        src = build_take_fbx(
            os.path.join(self.tmp, "takes.fbx"),
            {"Take 001": {("lift", "d|Y"): (1, 100)}},
        )
        budget = MeshConvert.conversion_timeout(src)
        captured, loads, stamped = {}, [], []

        def _run(cmd, **kw):
            captured.update(kw)
            _write_glb_file(
                cmd[cmd.index("-o") + 1] + ".glb", {"asset": {"version": "2.0"}}
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        real_load = FbxFile.load

        def counted(*args, **kwargs):
            loads.append(args[0])
            return real_load(*args, **kwargs)

        def read(fbx, bake_fps=None):
            stamped.append((type(fbx), bake_fps))
            return {}

        with (
            patch.object(MeshConvert, "resolve_binary", return_value=self.fake_bin),
            patch("subprocess.run", side_effect=_run),
            patch.object(FbxFile, "load", side_effect=counted),
            patch.object(MeshConvert, "render_effect_curves", side_effect=read),
        ):
            MeshConvert.fbx_to_glb(src, auto_install=False)

        self.assertEqual(len(loads), 1, loads)
        self.assertEqual(stamped, [(FbxFile, float(MeshConvert.CONVERSION_BAKE_FPS))])
        self.assertEqual(captured["timeout"], budget)


class TestCheckGlbMaterials(unittest.TestCase):
    """Verify the post-conversion material sanity check."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_check_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    @staticmethod
    def _png_bytes(mode, size=(4, 4), alpha=None):
        """Build a small PNG and return its bytes."""
        from io import BytesIO
        from PIL import Image

        if mode == "RGBA":
            im = Image.new(
                "RGBA", size, (200, 100, 50, alpha if alpha is not None else 255)
            )
        elif mode == "RGB":
            im = Image.new("RGB", size, (200, 100, 50))
        else:
            raise ValueError(mode)
        buf = BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _build_glb(materials, images, textures, image_blobs):
        """Pack a synthetic GLB. image_blobs is a list of bytes objects."""
        import struct

        buffer_views = []
        bin_chunks = []
        offset = 0
        for blob in image_blobs:
            buffer_views.append(
                {"buffer": 0, "byteOffset": offset, "byteLength": len(blob)}
            )
            bin_chunks.append(blob)
            # 4-byte align
            pad = (4 - (len(blob) % 4)) % 4
            if pad:
                bin_chunks.append(b"\x00" * pad)
                offset += pad
            offset += len(blob)

        bin_data = b"".join(bin_chunks)
        gltf = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": len(bin_data)}],
            "bufferViews": buffer_views,
            "images": images,
            "textures": textures,
            "materials": materials,
        }
        json_bytes = json.dumps(gltf).encode("utf-8")
        # Pad JSON chunk to 4-byte boundary with spaces
        pad_json = (4 - (len(json_bytes) % 4)) % 4
        json_bytes += b" " * pad_json

        header = struct.pack(
            "<4sII", b"glTF", 2, 12 + 8 + len(json_bytes) + 8 + len(bin_data)
        )
        json_chunk = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
        bin_chunk = struct.pack("<I4s", len(bin_data), b"BIN\x00") + bin_data
        return header + json_chunk + bin_chunk

    def _write_glb(self, name, **kw):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(self._build_glb(**kw))
        return path

    def test_flags_blend_with_opaque_alpha(self):
        """Texture is RGBA but alpha=255 everywhere → must flag."""
        blob = self._png_bytes("RGBA", alpha=255)
        path = self._write_glb(
            "opaque_blend.glb",
            materials=[
                {
                    "name": "Body_base",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png", "name": "color"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        findings = MeshConvert.check_glb_materials(path)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["material"], "Body_base")
        self.assertEqual(findings[0]["alpha_mode"], "BLEND")

    def test_does_not_flag_genuine_transparency(self):
        """Texture is RGBA with varying alpha → genuine transparency, no flag."""
        from io import BytesIO
        from PIL import Image

        im = Image.new("RGBA", (4, 4))
        for y in range(4):
            for x in range(4):
                im.putpixel((x, y), (200, 100, 50, 30 + 50 * x))
        buf = BytesIO()
        im.save(buf, format="PNG")
        path = self._write_glb(
            "real_blend.glb",
            materials=[
                {
                    "name": "Glass",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[buf.getvalue()],
        )
        self.assertEqual(MeshConvert.check_glb_materials(path), [])

    def test_does_not_flag_opaque_material(self):
        """alphaMode=OPAQUE is never flagged, even if texture is RGBA."""
        blob = self._png_bytes("RGBA", alpha=255)
        path = self._write_glb(
            "opaque.glb",
            materials=[
                {
                    "name": "Plain",
                    "alphaMode": "OPAQUE",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        self.assertEqual(MeshConvert.check_glb_materials(path), [])

    def test_does_not_flag_rgb_texture(self):
        """No alpha channel → can't have leaked transparency, no flag."""
        blob = self._png_bytes("RGB")
        path = self._write_glb(
            "no_alpha.glb",
            materials=[
                {
                    "name": "RGBish",
                    "alphaMode": "BLEND",  # weird but possible
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        self.assertEqual(MeshConvert.check_glb_materials(path), [])

    def test_mask_mode_is_also_checked(self):
        """alphaMode=MASK with uniformly-255 alpha is still wrong."""
        blob = self._png_bytes("RGBA", alpha=255)
        path = self._write_glb(
            "mask.glb",
            materials=[
                {
                    "name": "Leaf",
                    "alphaMode": "MASK",
                    "alphaCutoff": 0.5,
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        findings = MeshConvert.check_glb_materials(path)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["alpha_mode"], "MASK")

    def test_does_not_flag_transparency_from_basecolorfactor(self):
        """Material is legitimately transparent via baseColorFactor[3] < 1.0 —
        even with an opaque texture this is real transparency, not a leak."""
        blob = self._png_bytes("RGBA", alpha=255)
        path = self._write_glb(
            "factor_alpha.glb",
            materials=[
                {
                    "name": "TintedGlass",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 0.4],
                        "baseColorTexture": {"index": 0},
                    },
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        self.assertEqual(MeshConvert.check_glb_materials(path), [])

    def test_basecolorfactor_alpha_1_still_flags(self):
        """Factor alpha = 1.0 is fully opaque so it must NOT exempt the
        material — the texture-alpha leak should still be caught."""
        blob = self._png_bytes("RGBA", alpha=255)
        path = self._write_glb(
            "factor_one.glb",
            materials=[
                {
                    "name": "Body",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                        "baseColorTexture": {"index": 0},
                    },
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        self.assertEqual(len(MeshConvert.check_glb_materials(path)), 1)

    def test_reason_differs_per_alpha_mode(self):
        """BLEND reason mentions depth-write; MASK reason mentions no-op
        alpha-test. They must not be the same boilerplate."""
        blob = self._png_bytes("RGBA", alpha=255)

        blend_path = self._write_glb(
            "blend_reason.glb",
            materials=[
                {
                    "name": "B",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        mask_path = self._write_glb(
            "mask_reason.glb",
            materials=[
                {
                    "name": "M",
                    "alphaMode": "MASK",
                    "alphaCutoff": 0.5,
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )
        blend_reason = MeshConvert.check_glb_materials(blend_path)[0]["reason"]
        mask_reason = MeshConvert.check_glb_materials(mask_path)[0]["reason"]
        self.assertIn("depth-write", blend_reason)
        self.assertIn("no-op", mask_reason)
        self.assertNotEqual(blend_reason, mask_reason)

    def test_shared_image_decoded_once(self):
        """Two materials referencing the same image should both flag, but
        the underlying image must only be decoded a single time (cache)."""
        from unittest.mock import patch
        from PIL import Image as PILImage

        blob = self._png_bytes("RGBA", alpha=255)
        path = self._write_glb(
            "shared_image.glb",
            materials=[
                {
                    "name": "A",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                },
                {
                    "name": "B",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                },
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[blob],
        )

        real_open = PILImage.open
        calls = {"n": 0}

        def counting_open(*a, **kw):
            calls["n"] += 1
            return real_open(*a, **kw)

        with patch("PIL.Image.open", side_effect=counting_open):
            findings = MeshConvert.check_glb_materials(path)

        self.assertEqual(len(findings), 2, "both materials must be flagged")
        self.assertEqual(calls["n"], 1, "image should be decoded only once")

    def test_raises_on_non_glb(self):
        path = os.path.join(self.tmp, "not_a_glb.bin")
        with open(path, "wb") as f:
            f.write(b"not glTF")
        with self.assertRaises(ValueError):
            MeshConvert.check_glb_materials(path)

    def test_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            MeshConvert.check_glb_materials(os.path.join(self.tmp, "nope.glb"))


class TestFixGlbPhantomOpaqueAlpha(unittest.TestCase):
    """Verify the post-conversion fix for the Maya phong → FBX → glTF
    transparency translation bug, where baseColorFactor[3]=0 cancels the
    per-pixel alpha of a real cutout mask."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_fix_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    @staticmethod
    def _varying_alpha_png():
        from io import BytesIO
        from PIL import Image

        im = Image.new("RGBA", (4, 4))
        for y in range(4):
            for x in range(4):
                im.putpixel((x, y), (200, 100, 50, 30 + 50 * x))
        buf = BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _opaque_alpha_png():
        from io import BytesIO
        from PIL import Image

        im = Image.new("RGBA", (4, 4), (200, 100, 50, 255))
        buf = BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def _write_glb(self, name, materials, images, textures, image_blobs):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(
                TestCheckGlbMaterials._build_glb(
                    materials=materials,
                    images=images,
                    textures=textures,
                    image_blobs=image_blobs,
                )
            )
        return path

    @staticmethod
    def _read_alpha_factor(path):
        with open(path, "rb") as f:
            f.read(12)
            chunk0_len = struct.unpack("<I", f.read(4))[0]
            f.read(4)  # JSON
            gltf = json.loads(f.read(chunk0_len).decode("utf-8"))
        return gltf["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"][3]

    def test_fixes_blend_with_zero_alpha_and_varying_texture(self):
        """The bug pattern: BLEND + baseColorFactor[3]=0 + varying-alpha texture."""
        path = self._write_glb(
            "buggy.glb",
            materials=[
                {
                    "name": "TREELINE_D",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                        "baseColorTexture": {"index": 0},
                    },
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[self._varying_alpha_png()],
        )
        fixes = MeshConvert.fix_glb_phantom_opaque_alpha(path)
        self.assertEqual(len(fixes), 1)
        self.assertEqual(fixes[0]["material"], "TREELINE_D")
        self.assertEqual(fixes[0]["new_alpha"], 1.0)
        self.assertEqual(self._read_alpha_factor(path), 1.0)

    def test_fixes_mask_with_zero_alpha(self):
        """Same bug under alphaMode=MASK."""
        path = self._write_glb(
            "buggy_mask.glb",
            materials=[
                {
                    "name": "FoliageMask",
                    "alphaMode": "MASK",
                    "alphaCutoff": 0.5,
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                        "baseColorTexture": {"index": 0},
                    },
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[self._varying_alpha_png()],
        )
        fixes = MeshConvert.fix_glb_phantom_opaque_alpha(path)
        self.assertEqual(len(fixes), 1)
        self.assertEqual(self._read_alpha_factor(path), 1.0)

    def test_skips_opaque_material(self):
        """alphaMode=OPAQUE never gets touched."""
        path = self._write_glb(
            "opaque.glb",
            materials=[
                {
                    "name": "Plain",
                    "alphaMode": "OPAQUE",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                        "baseColorTexture": {"index": 0},
                    },
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[self._varying_alpha_png()],
        )
        self.assertEqual(MeshConvert.fix_glb_phantom_opaque_alpha(path), [])
        self.assertEqual(self._read_alpha_factor(path), 0.0)

    def test_skips_when_factor_already_nonzero(self):
        """Genuine partial-transparency factor must not be promoted."""
        path = self._write_glb(
            "partial.glb",
            materials=[
                {
                    "name": "Glass",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 0.5],
                        "baseColorTexture": {"index": 0},
                    },
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[self._varying_alpha_png()],
        )
        self.assertEqual(MeshConvert.fix_glb_phantom_opaque_alpha(path), [])
        self.assertEqual(self._read_alpha_factor(path), 0.5)

    def test_skips_when_texture_alpha_uniform(self):
        """Uniformly-opaque alpha (the 'check' bug) is NOT fixed by this pass."""
        path = self._write_glb(
            "uniform.glb",
            materials=[
                {
                    "name": "UniformAlpha",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                        "baseColorTexture": {"index": 0},
                    },
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[self._opaque_alpha_png()],
        )
        self.assertEqual(MeshConvert.fix_glb_phantom_opaque_alpha(path), [])
        self.assertEqual(self._read_alpha_factor(path), 0.0)

    def test_skips_when_no_base_color_texture(self):
        """No texture means no per-pixel alpha to recover; leave alone."""
        path = self._write_glb(
            "no_tex.glb",
            materials=[
                {
                    "name": "NoTex",
                    "alphaMode": "BLEND",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 0.0],
                    },
                }
            ],
            images=[],
            textures=[],
            image_blobs=[],
        )
        self.assertEqual(MeshConvert.fix_glb_phantom_opaque_alpha(path), [])

    def test_returns_empty_when_nothing_to_fix(self):
        """No changes → no rewrite, empty list returned, file untouched."""
        path = self._write_glb(
            "clean.glb",
            materials=[
                {
                    "name": "Plain",
                    "alphaMode": "OPAQUE",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
            images=[{"bufferView": 0, "mimeType": "image/png"}],
            textures=[{"source": 0}],
            image_blobs=[self._opaque_alpha_png()],
        )
        before = open(path, "rb").read()
        self.assertEqual(MeshConvert.fix_glb_phantom_opaque_alpha(path), [])
        self.assertEqual(open(path, "rb").read(), before)


class TestGlbReadGuards(unittest.TestCase):
    """A truncated GLB must fail as a ValueError, not a struct.error.

    Batch callers catch ``(RuntimeError, ValueError, OSError)`` per file.
    ``struct.error`` derives straight from ``Exception``, so a GLB truncated
    mid-write (converter killed, disk full) escaped the per-file handler and
    aborted the whole batch.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_glb_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _truncated(self, payload):
        path = os.path.join(self.tmp, "truncated.glb")
        with open(path, "wb") as f:
            f.write(payload)
        return path

    def test_header_shorter_than_12_bytes_raises_value_error(self):
        path = self._truncated(b"glTF\x02\x00\x00\x00")  # valid magic, no length
        with self.assertRaises(ValueError):
            MeshConvert.check_glb_materials(path)

    def test_missing_chunk_header_raises_value_error(self):
        path = self._truncated(b"glTF" + struct.pack("<II", 2, 12))  # header only
        with self.assertRaises(ValueError):
            MeshConvert.check_glb_materials(path)

    def test_a_file_that_is_not_a_glb_raises_value_error(self):
        path = self._truncated(b"not a glb at all")  # no glTF magic
        with self.assertRaises(ValueError):
            MeshConvert.check_glb_materials(path)


def _write_glb_file(path, gltf, bin_chunk=b""):
    """Pack *gltf* (+ optional BIN payload) into a GLB at *path*."""
    payload = json.dumps(gltf).encode("utf-8")
    payload += b" " * ((4 - (len(payload) % 4)) % 4)
    rest = b""
    if bin_chunk:
        bin_chunk += b"\x00" * ((4 - (len(bin_chunk) % 4)) % 4)
        rest = struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(payload) + len(rest)))
        f.write(struct.pack("<I4s", len(payload), b"JSON") + payload)
        f.write(rest)
    return path


def _shot_record(takes, fps=30.0):
    """The ``animation`` record (version 3) declaring one shot per entry of
    *takes* -- a clip name, or a ``{"name", ...}`` take, whose range is the
    file's own, never the record's -- at the
    scene's rate *fps*, as the producer states it."""
    return {
        "version": SceneRecords.SHOTS.version,
        "fps": fps,
        "shots": [
            {"clip": t["name"] if isinstance(t, dict) else str(t)} for t in takes
        ],
    }


def _carried_lightmaps(glb):
    """The ``lightmaps`` record *glb*'s ``data_export`` carrier holds, or
    ``None``: what a build reads before the records leave the file."""
    with MeshConvert.open_glb(glb) as edit:
        return MeshConvert.data_export_channel(edit.gltf, SceneRecords.LIGHTMAPS.key)


def _baked(entry):
    """*entry* -- a fixture's baked object, naming the fields its test is
    about -- as every producer writes one (``LightmapEntry``): each field the
    record declares, at its declared default where the test names none. One
    that is no object, or names no object or map, is left as written: the
    malformed entry a test is about."""
    if not isinstance(entry, dict) or "name" not in entry or "map" not in entry:
        return entry
    return {**json.loads(json.dumps(LightmapEntry.skeleton())), **entry}


def _lightmaps_record(*names, version=None):
    """A ``lightmaps`` record (version 3 unless *version* says otherwise)
    baking each of *names* into a map of its own (``<name>_LM.exr``)."""
    return {
        "version": SceneRecords.LIGHTMAPS.version if version is None else version,
        "objects": [_baked({"name": n, "map": f"{n}_LM.exr"}) for n in names],
    }


def _ride_media(path, names, mime_type="image/png", scale=1.0):
    """Give the GLB at *path* a bufferView per file of *names*, as the media
    transport carries a record's files (``SceneRecords.GLB_MEDIA``)."""
    media = SceneRecords.GLB_MEDIA
    with MeshConvert.open_glb(path) as edit:
        for name in names:
            (index,) = MeshConvert._append_bin_views(edit, [b"\x89PNG"])
            edit.gltf["bufferViews"][index].update(
                {media.name: name, "extras": media.extras(mime_type, scale)}
            )
        edit.dirty = True
    return path


def _convert_prebuilt(tmp, converted, src=None, **kwargs):
    """`fbx_to_glb` with FBX2glTF stubbed by *converted*, a pre-built GLB.

    Everything after the converter is real: the one edit session the preview
    and the Scene Exporters both run. *src* is the FBX it converts (a real
    one, for a pass that reads it); a stand-in by default.
    """

    def _run(cmd, **kw):
        shutil.copyfile(converted, cmd[cmd.index("-o") + 1] + ".glb")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    if src is None:
        src = os.path.join(tmp, "in.fbx")
        with open(src, "wb") as fh:
            fh.write(b"fbx")
    with (
        unittest.mock.patch.object(
            MeshConvert, "resolve_binary", return_value="FBX2glTF"
        ),
        unittest.mock.patch("subprocess.run", side_effect=_run),
    ):
        return MeshConvert.fbx_to_glb(src, overwrite=True, prompt=False, **kwargs)


def _glb_pointer_channels(path):
    """``{clip: [(pointer, times, values)]}`` for every animation in *path*.

    Every clip is listed, those without a pointer channel as ``[]``, so a
    lookup of a clip the file lacks raises instead of reading as empty. Values
    are sized by the output accessor's type (VEC3 emissive, VEC4 base colour).
    """
    with MeshConvert.open_glb(path) as edit:
        gltf, blob = edit.gltf, edit.bin_data

    def read(index):
        acc = gltf["accessors"][index]
        per = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[acc["type"]]
        view = gltf["bufferViews"][acc["bufferView"]]
        off = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        flat = struct.unpack_from(f"<{acc['count'] * per}f", blob, off)
        return [flat[i : i + per] for i in range(0, len(flat), per)]

    out = {}
    for animation in gltf.get("animations") or []:
        channels = out.setdefault(animation.get("name"), [])
        for channel in animation.get("channels") or []:
            target = channel.get("target") or {}
            if target.get("path") != "pointer":
                continue
            pointer = target["extensions"]["KHR_animation_pointer"]["pointer"]
            sampler = animation["samplers"][channel["sampler"]]
            times = [t[0] for t in read(sampler["input"])]
            channels.append((pointer, times, read(sampler["output"])))
    return out


def _converted_orm_glb(tmp, name="converted.glb"):
    """A GLB the way FBX2glTF hands one over: the material samples its
    converted ORM (image 0, bufferView 0 in the BIN) and the geometry
    (accessor -> bufferView 1) sits BEHIND it in the same BIN, so dropping
    the image must move the geometry's view without moving its bytes.

    Returns ``(path, geometry_bytes)``.
    """
    import io as iolib

    from PIL import Image

    buf = iolib.BytesIO()
    Image.new("RGB", (4, 4), (255, 255, 255)).save(buf, format="PNG")
    orm_bytes = buf.getvalue()
    orm_padded = orm_bytes + b"\x00" * ((4 - (len(orm_bytes) % 4)) % 4)
    geometry = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(orm_padded) + len(geometry)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(orm_bytes)},
            {
                "buffer": 0,
                "byteOffset": len(orm_padded),
                "byteLength": len(geometry),
                "target": 34962,
            },
        ],
        "accessors": [
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [1, 1, 0],
            }
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "materials": [
            {
                "name": "Room",
                "pbrMetallicRoughness": {
                    "metallicRoughnessTexture": {"index": 0},
                    "metallicFactor": 1.0,
                },
                "occlusionTexture": {"index": 0},
            }
        ],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{}],
        "images": [
            {"name": "ao_met_rough_Room", "bufferView": 0, "mimeType": "image/png"}
        ],
    }
    path = _write_glb_file(os.path.join(tmp, name), gltf, orm_padded + geometry)
    return path, geometry


def _assert_no_dead_payload(tc, path, geometry):
    """*path* ships no unreferenced image, the converted ORM is gone, and the
    geometry survived the BIN rewrite (view moved, bytes did not)."""
    edit = MeshConvert._read_glb(path)
    gltf = edit.gltf
    referenced = {
        gltf["textures"][t["index"]]["source"]
        for m in gltf["materials"]
        for t in (
            m["pbrMetallicRoughness"]["metallicRoughnessTexture"],
            m["occlusionTexture"],
        )
    }
    tc.assertEqual(
        referenced, set(range(len(gltf["images"]))), "unreferenced image shipped"
    )
    tc.assertNotIn("ao_met_rough_Room", [i.get("name") for i in gltf["images"]])
    view = gltf["bufferViews"][gltf["accessors"][0]["bufferView"]]
    blob = edit.bin_data
    tc.assertEqual(
        bytes(blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]),
        geometry,
    )
    tc.assertEqual(view.get("target"), 34962, "view attributes must survive")
    # The displaced ORM's view must be gone, stated as the invariant that
    # outlives the layout: nothing ships a view no accessor and no image reads.
    # (A count was the proxy for this while embeds lived in the JSON chunk as
    # base64; they are relocated into the BIN on close now, so the packed ORM
    # legitimately owns a view of its own.)
    live_views = {gltf["accessors"][0]["bufferView"]} | {
        i["bufferView"] for i in gltf["images"] if "bufferView" in i
    }
    tc.assertEqual(
        live_views,
        set(range(len(gltf["bufferViews"]))),
        "a bufferView shipped that nothing references",
    )
    tc.assertEqual(gltf["buffers"][0]["byteLength"], len(blob))
    return gltf


class TestGlbEditSession(unittest.TestCase):
    """One open GLB shared by several repairs: read once, write once.

    Every repair edits the JSON chunk and nothing else, but each used to open,
    parse and rewrite the whole file for itself — so a preview push, which runs
    three of them back to back, read a file the size of its geometry three
    times to change a handful of fields. These pin the four properties that
    made collapsing it safe.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_session_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write_glb(self, gltf, bin_chunk=b"", pretty=False, name="s.glb", tail=b""):
        """Pack a GLB. *pretty* pads the JSON chunk the way a producer that
        indents its output does — which is what leaves room for the in-place
        rewrite, since this module always re-serializes compactly. *tail* is
        appended raw, for the extension chunks a repack must carry over."""
        payload = json.dumps(gltf, indent=4 if pretty else None).encode("utf-8")
        payload += b" " * ((4 - (len(payload) % 4)) % 4)
        rest = b""
        if bin_chunk:
            bin_chunk += b"\x00" * ((4 - (len(bin_chunk) % 4)) % 4)
            rest = struct.pack("<I4s", len(bin_chunk), b"BIN\x00") + bin_chunk
        rest += tail
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(payload) + len(rest)))
            f.write(struct.pack("<I4s", len(payload), b"JSON") + payload)
            f.write(rest)
        return path

    def _png(self, name="map.png"):
        from PIL import Image

        path = os.path.join(self.tmp, name)
        Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(path)
        return path

    @staticmethod
    def _record(**materials):
        """A ``materials`` record stating each ``name={fields}`` (a flat
        statement, its maps by file name) -- what
        :meth:`MeshConvert.apply_glb_materials` reads; an unstated field takes
        its default."""
        return MaterialFamilies.payload(
            [{"name": name, **fields} for name, fields in materials.items()]
        )

    @staticmethod
    def _maps(*paths):
        """``{file name: path}`` for each map on disk, as the record names them."""
        return {os.path.basename(path): path for path in paths}

    @staticmethod
    def _counting_write():
        """``(patcher, writes)`` — wraps ``GlbEdit.write`` and records each call."""
        real = MeshConvert.GlbEdit.write
        writes = []

        def counting(edit):
            writes.append(edit.path)
            return real(edit)

        return patch.object(MeshConvert.GlbEdit, "write", counting), writes

    @staticmethod
    def _xtra_chunk(payload=b"\x01\x02\x03\x04" * 4):
        """A chunk of a type this module does not know, to be carried over."""
        return struct.pack("<I4s", len(payload), b"XTRA") + payload

    def _assert_container_intact(self, path):
        """The header's total length must still describe the file on disk.

        Nothing re-reads this field -- ``_read_glb`` takes the JSON length and
        treats the remainder as the tail -- so a tail this module grew or shrank
        without recomputing it round-trips happily here and is rejected by a
        strict loader instead.
        """
        with open(path, "rb") as f:
            declared = struct.unpack("<I", f.read(12)[8:12])[0]
        self.assertEqual(declared, os.path.getsize(path), "GLB total length is stale")

    def test_one_session_writes_once_for_every_applier(self):
        """Two appliers sharing a session must produce one write."""
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [
                    {"name": "Body"},
                    {"name": "Lens", "normalTexture": {"index": 0}},
                ],
                "textures": [{"source": 0}],
                "images": [{"uri": "lens_Normal.png"}],
            },
            bin_chunk=b"GEOMETRY",
        )
        patcher, writes = self._counting_write()
        with patcher:
            with MeshConvert.open_glb(path) as session:
                MeshConvert.apply_glb_materials(
                    session,
                    self._record(
                        Body={
                            "baseColorFactor": [0.2, 0.4, 0.8, 1.0],
                            "emissiveFactor": [1.0, 0.0, 0.0],
                        }
                    ),
                )
                MeshConvert.set_glb_normal_scale(session, 1.5, lightmapped_only=False)

        self.assertEqual(len(writes), 1, "the session must write once, not per applier")
        body, lens = MeshConvert._read_glb(path).gltf["materials"]
        self.assertEqual(
            body["pbrMetallicRoughness"]["baseColorFactor"], [0.2, 0.4, 0.8, 1.0]
        )
        self.assertEqual(body["emissiveFactor"], [1.0, 0.0, 0.0])
        self.assertEqual(lens["normalTexture"]["scale"], 1.5)

    def test_a_path_target_still_writes_for_itself(self):
        """The path form stays self-contained — sharing is opt-in, not required."""
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )
        patcher, writes = self._counting_write()
        with patcher:
            MeshConvert.apply_glb_materials(
                path, self._record(Body={"baseColorFactor": [0.2, 0.4, 0.8, 1.0]})
            )
            MeshConvert.apply_glb_materials(
                path, self._record(Body={"emissiveFactor": [1.0, 0.0, 0.0]})
            )
        self.assertEqual(len(writes), 2)

    def test_a_black_emissive_switches_the_converters_glow_off(self):
        """Off means off: FBX2glTF glows a mapped emissive at 1.0 whatever the DCC
        said, so a material authored dark (Maya's ``emissive_intensity`` 0) must
        leave with factor 0 and no map -- skipped, a production room's ceiling
        panels lit WebXR at full (2026-10-17). The map the record names is not
        embedded for it."""
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [
                    {
                        "name": "Panel",
                        "emissiveFactor": [1.0, 1.0, 1.0],
                        "emissiveTexture": {"index": 0},
                        "extensions": {
                            "KHR_materials_emissive_strength": {"emissiveStrength": 4.0}
                        },
                    }
                ],
                "textures": [{"source": 0}],
                "images": [{"uri": "panel_Emissive.png"}],
            }
        )
        png = self._png()
        records = MeshConvert.apply_glb_materials(
            path,
            self._record(
                Panel={"emissiveFactor": [0.0, 0.0, 0.0], "emissiveTexture": "map.png"}
            ),
            self._maps(png),
        )
        gltf = MeshConvert._read_glb(path).gltf
        mat = gltf["materials"][0]
        self.assertEqual(mat["emissiveFactor"], [0.0, 0.0, 0.0])
        self.assertNotIn("emissiveTexture", mat)
        self.assertNotIn("KHR_materials_emissive_strength", mat.get("extensions") or {})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["maps"], [])  # nothing embedded for an unlit map
        self.assertEqual(gltf["images"], [{"uri": "panel_Emissive.png"}])

    def test_an_applier_that_matches_nothing_writes_nothing(self):
        """No match, no edit, no I/O — the usual case for the alpha repair. A
        record stating no material is the same no-op."""
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )
        before = open(path, "rb").read()
        patcher, writes = self._counting_write()
        with patcher:
            for record in (self._record(Absent={}), {"materials": []}, None):
                with self.subTest(record=record):
                    self.assertEqual(MeshConvert.apply_glb_materials(path, record), [])
        self.assertEqual(writes, [])
        self.assertEqual(open(path, "rb").read(), before)

    def test_a_json_only_edit_leaves_the_file_size_and_bin_chunk_alone(self):
        """A shrinking edit is written in place, not by copying the geometry.

        This module re-serializes compactly while most producers do not, so the
        new JSON usually fits the chunk it came from; it is then padded back to
        exactly that length, leaving the total size and every byte after the
        JSON chunk untouched.
        """
        geometry = b"GEOMETRY" * 8
        # The converter's material as the record restates it, one field apart:
        # the compact rewrite of the same fields fits the indented chunk.
        body = {
            "name": "Body",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 1.0,
                "roughnessFactor": 1.0,
            },
            "emissiveFactor": [0.0, 0.0, 0.0],
            "alphaMode": "OPAQUE",
            "doubleSided": False,
        }
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [body]},
            bin_chunk=geometry,
            pretty=True,
        )
        size_before = os.path.getsize(path)

        MeshConvert.apply_glb_materials(
            path, self._record(Body={"emissiveFactor": [1.0, 0.5, 0.0]})
        )

        self.assertEqual(os.path.getsize(path), size_before)
        edit = MeshConvert._read_glb(path)
        self.assertEqual(edit.gltf["materials"][0]["emissiveFactor"], [1.0, 0.5, 0.0])
        self.assertEqual(bytes(edit.bin_data), geometry)

    def test_a_growing_json_only_edit_keeps_the_bin_byte_identical(self):
        """JSON that outgrows its chunk with nothing embedded: the file is
        rewritten, and the BIN chunk comes back as it was, byte for byte."""
        geometry = b"GEOMETRYDATA1234"
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]},
            bin_chunk=geometry,
        )
        size_before = os.path.getsize(path)

        MeshConvert.apply_glb_materials(
            path, self._record(Body={"emissiveFactor": [1.0, 1.0, 1.0]})
        )

        self.assertGreater(os.path.getsize(path), size_before)
        self.assertEqual(bytes(MeshConvert._read_glb(path).bin_data), geometry)

    def test_a_growing_edit_falls_back_to_a_full_rewrite(self):
        """When the JSON no longer fits, the geometry must still come back whole."""
        geometry = b"GEOMETRYDATA1234"
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]},
            bin_chunk=geometry,
        )
        size_before = os.path.getsize(path)

        MeshConvert.apply_glb_materials(
            path,
            self._record(
                Body={"emissiveFactor": [1.0, 1.0, 1.0], "emissiveTexture": "map.png"}
            ),
            self._maps(self._png()),
        )

        # Growth is what distinguishes this from the in-place path, which holds
        # the size fixed — asserting only that the file parses would pass on
        # either, and the fallback is the branch that has to move the geometry.
        self.assertGreater(os.path.getsize(path), size_before)
        edit = MeshConvert._read_glb(path)
        # The geometry keeps its bytes AND its place at the front of the BIN --
        # the embed is appended behind it, never interleaved.
        self.assertEqual(bytes(edit.bin_data)[: len(geometry)], geometry)
        image = edit.gltf["images"][0]
        self.assertNotIn("uri", image, "the embed stayed base64 in the JSON chunk")
        self.assertIn("bufferView", image)

    def test_an_unreadable_texture_is_skipped_not_raised(self):
        """A texture that exists but cannot be read must not abort the writer.

        The re-encode branch already warned and skipped; the direct PNG/JPEG
        read did not, so the same failure aborted the channel writer depending
        only on the file's extension — and it was the one file operation left
        inside an applier, able to take a whole materials record with it.
        """
        png = self._png()
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )

        real_open = open

        def refusing_open(file, *args, **kwargs):
            if os.path.abspath(str(file)) == os.path.abspath(png):
                raise PermissionError(13, "Permission denied", str(file))
            return real_open(file, *args, **kwargs)

        with patch("builtins.open", refusing_open):
            records = MeshConvert.apply_glb_materials(
                path,
                self._record(
                    Body={
                        "baseColorFactor": [1.0, 0.0, 0.0, 1.0],
                        "baseColorTexture": "map.png",
                    }
                ),
                self._maps(png),
            )

        self.assertEqual(len(records), 1, "the colour must still be written")
        self.assertEqual(
            (records[0]["maps"], records[0]["missing"]),
            ([], ["map.png"]),
            "no texture was embedded — don't claim one",
        )
        gltf = MeshConvert._read_glb(path).gltf
        self.assertNotIn("images", gltf)
        self.assertEqual(
            gltf["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"],
            [1.0, 0.0, 0.0, 1.0],
        )

    def test_an_image_no_decoder_reads_is_skipped_not_embedded(self):
        """A file no decoder can read must be reported, not written as a data URI.

        Junk bytes under a TIFF extension: the Pillow fallback tries and fails
        to decode it, so it takes the same rejected path as a format with no
        decoder at all (EXR, or no Pillow installed).
        """
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )
        tif = os.path.join(self.tmp, "e.tif")
        with open(tif, "wb") as f:
            f.write(b"II*\x00")
        (written,) = MeshConvert.apply_glb_materials(
            path,
            self._record(
                Body={"emissiveFactor": [1.0, 1.0, 1.0], "emissiveTexture": "e.tif"}
            ),
            self._maps(tif),
        )
        self.assertEqual(written["missing"], ["e.tif"])
        gltf = MeshConvert._read_glb(path).gltf
        self.assertNotIn("emissiveTexture", gltf["materials"][0])
        self.assertNotIn("images", gltf)

    def test_a_tga_map_is_reencoded_to_png(self):
        """TGA is everywhere in game art, and glTF cannot hold it natively.

        Rejecting it by extension silently downgraded the material to
        colour-only; with Pillow present it must arrive as an embedded PNG.
        """
        from PIL import Image

        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )
        tga = os.path.join(self.tmp, "emis.tga")
        Image.new("RGB", (2, 2), (255, 0, 0)).save(tga)
        (written,) = MeshConvert.apply_glb_materials(
            path,
            self._record(
                Body={"emissiveFactor": [1.0, 1.0, 1.0], "emissiveTexture": "emis.tga"}
            ),
            self._maps(tga),
        )
        self.assertEqual(written["maps"], ["emis.tga"])
        edit = MeshConvert._read_glb(path)
        image = edit.gltf["images"][0]
        self.assertEqual(image["mimeType"], "image/png")
        # The payload really is a PNG, not TGA bytes under a PNG label.
        view = edit.gltf["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        self.assertTrue(
            bytes(edit.bin_data[start : start + view["byteLength"]]).startswith(
                b"\x89PNG"
            )
        )

    def test_a_clean_glb_is_never_read_past_its_json_chunk(self):
        """The repair that runs after *every* conversion must stay cheap.

        Nothing in the alpha check needs the BIN chunk until a material already
        looks wrong, so a GLB with nothing to fix should touch only the first
        few kilobytes of what is routinely a hundreds-of-megabytes file.
        """
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "Plain", "alphaMode": "OPAQUE"}],
            },
            bin_chunk=b"GEOMETRY" * 16,
        )
        with MeshConvert.open_glb(path) as session:
            self.assertEqual(MeshConvert.fix_glb_phantom_opaque_alpha(session), [])
            self.assertIsNone(
                session._rest,
                "a repair with nothing to fix must not pull the geometry in",
            )

    def test_bin_data_is_a_view_rather_than_a_copy(self):
        """Slicing the BIN chunk out put peak memory at twice the file size."""
        path = self._write_glb({"asset": {"version": "2.0"}}, bin_chunk=b"GEOMETRY")
        edit = MeshConvert._read_glb(path)
        self.assertIsInstance(edit.bin_data, memoryview)
        self.assertEqual(bytes(edit.bin_data), b"GEOMETRY")

    def test_a_shared_texture_is_embedded_once_across_channels(self):
        """One file on disk, one embedded copy — the embed cache spans the
        session: two appliers on it, the map in two channels."""
        png = self._png()
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "Body"}, {"name": "Lens"}],
            }
        )
        maps = self._maps(png)
        with MeshConvert.open_glb(path) as session:
            MeshConvert.apply_glb_materials(
                session, self._record(Body={"baseColorTexture": "map.png"}), maps
            )
            MeshConvert.apply_glb_materials(
                session,
                self._record(
                    Lens={"emissiveFactor": [1.0, 1.0, 1.0], "emissiveTexture": "map.png"}
                ),
                maps,
            )

        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 1)
        body, lens = gltf["materials"]
        self.assertEqual(
            body["pbrMetallicRoughness"]["baseColorTexture"]["index"],
            lens["emissiveTexture"]["index"],
        )

    def test_an_embed_reuses_bytes_the_glb_already_carries(self):
        """The converter's own embedded media must not be base64'd in a second time.

        The embed cache is keyed by source PATH, so it can only dedupe embeds
        made this session — it cannot see that the FBX->GLB conversion already
        wrote the very same texture into the BIN chunk. Since the materials
        record restates exactly the maps FBX translation drops, the normal case
        was two copies of the same bytes, the second inflated ~33% by base64.
        Measured on a production room: 23 MB duplicated, 31 MB on disk, a
        quarter of the deliverable.
        """
        png = self._png()
        raw = open(png, "rb").read()
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "Body"}],
                "images": [{"bufferView": 0, "mimeType": "image/png", "name": "c.png"}],
                "textures": [{"source": 0}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(raw)}],
                "buffers": [{"byteLength": len(raw)}],
            },
            bin_chunk=raw,
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.apply_glb_materials(
                session,
                self._record(Body={"baseColorTexture": "map.png"}),
                self._maps(png),
            )

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        self.assertEqual(len(gltf["images"]), 1, "the payload was embedded twice")
        # A "no data: URI" check no longer bites here — the relocation pass
        # would turn a duplicate embed into a second bufferView, not a second
        # base64 blob. The BIN's size is what still catches it.
        self.assertEqual(
            bytes(edit.bin_data).count(raw),
            1,
            "a second copy of the payload reached the BIN",
        )
        index = gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"][
            "index"
        ]
        self.assertEqual(gltf["textures"][index]["source"], 0)

    def test_two_source_files_with_identical_bytes_embed_once(self):
        """Path-keyed dedupe misses this; content-keyed catches it.

        Duplicated textures under different names are routine in a DCC source
        tree, and neither copy need be in the GLB already — so the reuse index
        has to include what this session appends, not only what it opened.
        """
        import shutil

        first = self._png("a.png")
        second = os.path.join(self.tmp, "b.png")
        shutil.copyfile(first, second)
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "One"}, {"name": "Two"}],
            }
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.apply_glb_materials(
                session,
                self._record(
                    One={"baseColorTexture": "a.png"}, Two={"baseColorTexture": "b.png"}
                ),
                self._maps(first, second),
            )

        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 1, "identical bytes embedded twice")
        indices = {
            m["pbrMetallicRoughness"]["baseColorTexture"]["index"]
            for m in gltf["materials"]
        }
        self.assertTrue(
            all(gltf["textures"][i]["source"] == 0 for i in indices),
            "both materials must resolve to the single embedded image",
        )

    def test_an_embedded_image_lands_in_the_bin_not_the_json(self):
        """An embedded map ships as a bufferView, not base64 in the JSON chunk.

        Measured on a delivered asset (TURRETS_WIRES.glb): the sidecar's packed
        ORM rode in as a ``data:`` URI while the converter's own maps sat in the
        BIN, putting 4.0 MB of base64 in the JSON chunk -- 45% of an 8.9 MB
        deliverable, 1.0 MB of it pure base64 overhead, and every byte of it
        parsed before a loader can draw anything. The 33% premium was priced for
        a local preview; ``create_glb`` ships the same path as a deliverable.
        """
        png = self._png()
        raw = open(png, "rb").read()
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.apply_glb_materials(
                session,
                self._record(Body={"baseColorTexture": "map.png"}),
                self._maps(png),
            )

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        image = gltf["images"][0]
        self.assertNotIn("uri", image, "the payload is still base64 in the JSON")
        view = gltf["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        self.assertEqual(
            bytes(edit.bin_data[start : start + view["byteLength"]]),
            raw,
            "the relocated bytes must be the source PNG, verbatim",
        )
        self.assertEqual(gltf["buffers"][0]["byteLength"], len(bytes(edit.bin_data)))

    def test_relocating_an_embed_preserves_existing_bin_payloads(self):
        """Appending must not disturb a byte or an offset already in the BIN.

        Recomputing existing offsets is the part of GLB surgery that silently
        corrupts a file -- which is why the embed took the JSON in the first
        place. Appending past the end is what makes the BIN safe to touch:
        every prior view keeps its index, offset and bytes.
        """
        geometry = b"GEOMETRY" * 64
        png = self._png()
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "Body"}],
                "bufferViews": [
                    {"buffer": 0, "byteOffset": 0, "byteLength": len(geometry)}
                ],
                "buffers": [{"byteLength": len(geometry)}],
                "accessors": [
                    {
                        "bufferView": 0,
                        "componentType": 5126,
                        "count": 1,
                        "type": "SCALAR",
                    }
                ],
            },
            bin_chunk=geometry,
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.apply_glb_materials(
                session,
                self._record(Body={"baseColorTexture": "map.png"}),
                self._maps(png),
            )

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        self.assertEqual(gltf["bufferViews"][0]["byteOffset"], 0)
        self.assertEqual(gltf["bufferViews"][0]["byteLength"], len(geometry))
        self.assertEqual(gltf["accessors"][0]["bufferView"], 0)
        self.assertEqual(bytes(edit.bin_data)[: len(geometry)], geometry)
        self.assertGreater(len(gltf["bufferViews"]), 1, "no view was appended")

    def test_an_external_buffer_is_left_alone(self):
        """Buffer 0 is the BIN only when it declares no ``uri``.

        Appending to a GLB whose first buffer is EXTERNAL would strand the new
        views on bytes the file does not carry and overwrite that buffer's
        byteLength. Base64 is a size cost; a corrupt buffer table is not.
        """
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "Body"}],
                "buffers": [{"uri": "geometry.bin", "byteLength": 64}],
            }
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.apply_glb_materials(
                session,
                self._record(Body={"baseColorTexture": "map.png"}),
                self._maps(self._png()),
            )

        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(gltf["buffers"][0]["uri"], "geometry.bin")
        self.assertEqual(gltf["buffers"][0]["byteLength"], 64)
        self.assertTrue(gltf["images"][0]["uri"].startswith("data:"))
        self.assertNotIn("bufferView", gltf["images"][0])

    def test_an_unknown_trailing_chunk_survives_a_repack_without_a_bin(self):
        """Bytes after the JSON but no BIN: the relocation must not eat them.

        A repack rebuilds the whole tail from one payload (``replace_rest``),
        and the spec tells a client that meets a chunk type it does not know to
        IGNORE it, not to discard it -- nothing here decodes such a chunk, so
        nothing here could put one back. The new BIN goes FIRST, which is where
        the spec wants it, and the stranger rides along behind.
        """
        extra = self._xtra_chunk()
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]},
            tail=extra,
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.apply_glb_materials(
                session,
                self._record(Body={"baseColorTexture": "map.png"}),
                self._maps(self._png()),
            )

        self._assert_container_intact(path)
        edit = MeshConvert._read_glb(path)
        self.assertIn("bufferView", edit.gltf["images"][0], "nothing was relocated")
        self.assertEqual(edit.rest[4:8], b"BIN\x00", "BIN must lead the tail")
        self.assertTrue(edit.rest.endswith(extra), "the XTRA chunk was dropped")

    def test_an_unknown_trailing_chunk_survives_a_repack_behind_a_bin(self):
        """The same file WITH a BIN -- the case a `bin_data is None` guard misses.

        ``replace_rest`` rewrites every byte after the JSON, so a chunk sitting
        past the BIN is exactly as easy to lose as one standing in for it, and
        far likelier to exist. The relocation appends, so the original geometry
        must come back byte-for-byte too.
        """
        extra = self._xtra_chunk(b"\xaa\xbb" * 8)
        geometry = b"\x07" * 32
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "Body"}],
                "buffers": [{"byteLength": len(geometry)}],
            },
            bin_chunk=geometry,
            tail=extra,
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.apply_glb_materials(
                session,
                self._record(Body={"baseColorTexture": "map.png"}),
                self._maps(self._png()),
            )

        self._assert_container_intact(path)
        edit = MeshConvert._read_glb(path)
        self.assertIn("bufferView", edit.gltf["images"][0], "nothing was relocated")
        self.assertTrue(edit.rest.endswith(extra), "the XTRA chunk was dropped")
        self.assertEqual(
            bytes(edit.bin_data[: len(geometry)]),
            geometry,
            "the original geometry moved -- the append is not append-only",
        )

    def test_an_embedded_texture_carries_its_name(self):
        """The embed names its image; the texture sampling it went out unnamed.

        FBX2glTF names the textures it writes, so an unnamed one in the middle
        of the list is a tell that a later pass added it -- and it costs the
        only human-readable handle on which slot the map serves.
        """
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.apply_glb_materials(
                session,
                self._record(Body={"baseColorTexture": "map.png"}),
                self._maps(self._png()),
            )

        gltf = MeshConvert._read_glb(path).gltf
        self.assertTrue(
            gltf["textures"][0].get("name"), "the embedded texture has no name"
        )

    def test_an_embed_of_new_bytes_still_lands(self):
        """The reuse path must not swallow a texture the file does NOT already have."""
        existing = self._png("existing.png")
        fresh = self._png("fresh.png")
        from PIL import Image

        Image.new("RGBA", (2, 2), (0, 255, 0, 255)).save(fresh)
        raw = open(existing, "rb").read()
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "Body"}],
                "images": [{"bufferView": 0, "mimeType": "image/png", "name": "e.png"}],
                "textures": [{"source": 0}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(raw)}],
                "buffers": [{"byteLength": len(raw)}],
            },
            bin_chunk=raw,
        )
        with MeshConvert.open_glb(path) as session:
            MeshConvert.apply_glb_materials(
                session,
                self._record(Body={"baseColorTexture": "fresh.png"}),
                self._maps(fresh),
            )

        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 2)

    def test_optimize_repacks_images_and_preserves_geometry(self):
        """WebP re-encode + BIN repack: image shrinks, geometry bytes survive.

        The geometry view sits AFTER the image view on purpose — its offset
        must be recomputed when the image payload shrinks, and a stale offset
        reads garbage that no schema validator would catch.
        """
        from PIL import Image

        big = Image.new("RGB", (256, 256))
        for y in range(256):
            for x in range(0, 256, 16):
                big.putpixel((x, y), (x, y, 128))
        buffer = io.BytesIO()
        big.save(buffer, format="PNG")
        png = buffer.getvalue()
        geometry = bytes(range(256)) * 4  # recognizable, order-sensitive

        pad = (4 - len(png) % 4) % 4
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"bufferView": 0, "mimeType": "image/png", "name": "c.png"}],
                "textures": [{"source": 0}],
                "bufferViews": [
                    {"buffer": 0, "byteOffset": 0, "byteLength": len(png)},
                    {
                        "buffer": 0,
                        "byteOffset": len(png) + pad,
                        "byteLength": len(geometry),
                    },
                ],
                "buffers": [{"byteLength": len(png) + pad + len(geometry)}],
            },
            bin_chunk=png + b"\x00" * pad + geometry,
        )
        summary = MeshConvert.optimize_glb_textures(path, max_size=64)
        self.assertEqual(summary["images"], 1)
        self.assertLess(summary["bytes_after"], summary["bytes_before"])

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        image = gltf["images"][0]
        self.assertEqual(image["mimeType"], "image/webp")
        blob = edit.bin_data
        img_view = gltf["bufferViews"][image["bufferView"]]
        webp = bytes(
            blob[
                img_view["byteOffset"] : img_view["byteOffset"] + img_view["byteLength"]
            ]
        )
        self.assertEqual(webp[:4], b"RIFF", "payload must be a real WebP container")
        resized = Image.open(io.BytesIO(webp))
        self.assertEqual(max(resized.size), 64)

        geo_view = gltf["bufferViews"][1]
        survived = bytes(
            blob[
                geo_view["byteOffset"] : geo_view["byteOffset"] + geo_view["byteLength"]
            ]
        )
        self.assertEqual(survived, geometry, "geometry bytes corrupted by the repack")
        self.assertIn("EXT_texture_webp", gltf.get("extensionsUsed", []))
        self.assertEqual(
            gltf["textures"][0]["extensions"]["EXT_texture_webp"]["source"], 0
        )
        # Nothing core-readable survives a WebP pass -- the image IS the webp --
        # so the extension is REQUIRED, not merely used. Left un-escalated, the
        # file claimed any reader could open it while handing a core one webp
        # bytes through `source`, which glTF 2.0 does not permit.
        self.assertIn("EXT_texture_webp", gltf.get("extensionsRequired", []))
        self.assertNotIn(
            "source",
            gltf["textures"][0],
            "no PNG/JPEG twin exists here, so no fallback may be claimed",
        )

    def test_describe_texture_pass_reports_the_outcome_not_the_mode(self):
        """The sentence both DCC exporters log, pinned in the one place it lives.

        A ceiling is not a target: an asset authored at it resamples nothing,
        and the wording must not imply otherwise -- a "(resized)" label over an
        unchanged 2048 set reads as "the exporter upscaled my maps to 2K".
        """
        full = {"images": 4, "bytes_before": 13.1e6, "bytes_after": 15.7e6}

        untouched = MeshConvert.describe_texture_pass(
            {**full, "resized": 0}, "KTX2", 2048
        )
        self.assertIn("none resampled", untouched)
        self.assertIn("2048px", untouched)
        self.assertNotIn("resized", untouched)

        shrunk = MeshConvert.describe_texture_pass({**full, "resized": 3}, "KTX2", 2048)
        self.assertIn("3 resampled down to fit 2048px", shrunk)

        # max_size 0 is "never resample", so no resize claim may appear at all.
        container = MeshConvert.describe_texture_pass({**full, "resized": 0}, "WEBP", 0)
        self.assertIn("pixels untouched", container)
        self.assertNotIn("resampled", container)

        # An empty summary is a DIFFERENT outcome from never running.
        nothing = MeshConvert.describe_texture_pass({}, "KTX2", 2048)
        self.assertIn("changed nothing", nothing)

        # With no ceiling, a resample is credited to what actually caused it:
        # the data-map cap (a row of its own), not a power-of-two snap a WEBP
        # never takes -- and the cap is named once.
        capped = MeshConvert.describe_texture_pass(
            {**full, "resized": 2}, "WEBP", 0, secondary_max_size=512
        )
        self.assertIn("2 resampled down (data maps to 512px)", capped)
        self.assertNotIn("power-of-two", capped)
        self.assertEqual(capped.count("512px"), 1, capped)
        snapped = MeshConvert.describe_texture_pass({**full, "resized": 1}, "KTX2", 0)
        self.assertIn("1 resampled down (power-of-two edges for KTX2)", snapped)

    def test_the_conversion_timeout_scales_with_the_file_it_converts(self):
        """A flat 300s budget cannot fit both a prop and a production assembly.

        Measured: a 250 MB scene FBX (757 meshes, 98k triangles, 30 textures)
        converts in 180-230s on an idle machine and BLEW the 300s budget under
        ordinary CPU contention -- the Scene Exporter aborted with "produced no
        file", losing the whole deliverable. The failure is timing-dependent,
        which is the worst kind: green on a quiet machine, red mid-workday.

        So the default is derived from the input size, with the old constant
        kept as a FLOOR so nothing small converts on a shorter leash than
        before. An explicit timeout still wins outright -- a caller that says
        60 means 60 -- and ``None`` still means no limit.
        """
        MB = 1024 * 1024
        # Patched rather than written: the method reads a SIZE, so a real
        # 250 MB file would test the filesystem and churn a quarter gigabyte
        # per run to measure arithmetic.
        with unittest.mock.patch("os.path.getsize", return_value=1 * MB):
            self.assertEqual(
                MeshConvert.conversion_timeout("small.fbx"),
                MeshConvert.DEFAULT_TIMEOUT,
                "floor unchanged, so nothing small converts on a shorter leash",
            )
        with unittest.mock.patch("os.path.getsize", return_value=250 * MB):
            large = MeshConvert.conversion_timeout("assembly.fbx")
        self.assertGreater(
            large,
            300,
            "a 250 MB assembly measured 180-230s idle; 300s leaves no headroom",
        )
        # Enough headroom that ordinary contention does not decide the outcome.
        self.assertGreaterEqual(large, 600)

        # Second incident (2026-08-31): a 173 MB assembly at 3 s/MB got a
        # 495s budget and timed out while a test suite shared the machine --
        # the same "green quiet, red busy" failure this derivation exists
        # to prevent, one size class down.
        with unittest.mock.patch("os.path.getsize", return_value=173 * MB):
            busy = MeshConvert.conversion_timeout("assembly.fbx")
        self.assertGreaterEqual(
            busy,
            1500,
            "a 173 MB assembly timed out at 495s under ordinary contention",
        )

        # An unreadable size must fall back to the floor: a budget is never a
        # reason not to ATTEMPT a conversion.
        self.assertEqual(
            MeshConvert.conversion_timeout(os.path.join(self.tmp, "gone.fbx")),
            MeshConvert.DEFAULT_TIMEOUT,
        )

    def test_the_conversion_timeout_also_scales_with_the_bake(self):
        """Size alone is a WRONG proxy for FBX2glTF's cost, measured.

        The converter bakes every node at every frame of every take, so a
        12 MB textureless export of an animated assembly took longer than the
        300 s its size earned it -- and lost the push. The budget therefore
        also reads the file's census: nodes x baked frames.
        """
        from test_fbx_media import build_fbx

        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

        ticks = FbxFile.TICKS_PER_SECOND
        # 2000 nodes over a 100 s take at the 30 fps bake = 6M node-frames.
        animated = build_fbx(
            os.path.join(self.tmp, "animated.fbx"),
            {},
            models=2000,
            takes={"Take 001": (0, 100 * ticks)},
        )
        self.assertEqual(MeshConvert.CONVERSION_BAKE_FPS, 30.0)
        self.assertEqual(MeshConvert.bake_node_frames(animated), 2000 * 100 * 30)
        expected = 2000 * 100 * 30 * MeshConvert.TIMEOUT_SECONDS_PER_NODE_FRAME
        self.assertGreater(
            expected, MeshConvert.DEFAULT_TIMEOUT, "fixture too small to test"
        )
        self.assertAlmostEqual(MeshConvert.conversion_timeout(animated), expected)

        # No takes: nothing to bake, the size rule stands alone.
        static = build_fbx(os.path.join(self.tmp, "static.fbx"), {}, models=200)
        self.assertEqual(MeshConvert.bake_node_frames(static), 0)
        self.assertEqual(
            MeshConvert.conversion_timeout(static), MeshConvert.DEFAULT_TIMEOUT
        )

        # Not a binary FBX: the census degrades to zero rather than raising.
        junk = os.path.join(self.tmp, "junk.fbx")
        with open(junk, "wb") as fh:
            fh.write(b"not an fbx" * 10)
        self.assertEqual(MeshConvert.bake_node_frames(junk), 0)

    def test_the_bake_is_each_take_s_key_extent_not_its_declared_span(self):
        """FBX2glTF bakes a take over the key extent of every curve in it and
        ignores the span the take declares (measured 2026-10-04 on 0.13.1: a
        take declaring 1-120 baked 3-200). A budget read from the declared
        span undercounts every take whose keys outrun it -- the case that
        discards a finished conversion."""
        from test_export_verify import TICK, build_take_fbx

        from pythontk.file_utils.mesh_convert.fbx_file import FbxFile

        path = build_take_fbx(
            os.path.join(self.tmp, "outrun.fbx"),
            {"Take 001": {("lift", "d|Y"): (1, 3000), ("door", "d|Y"): (1, 10)}},
        )
        seconds = 2999 * TICK / FbxFile.TICKS_PER_SECOND
        self.assertEqual(
            MeshConvert.bake_node_frames(path),
            int(2 * seconds * MeshConvert.CONVERSION_BAKE_FPS),
        )

    def test_a_truncated_fbx_degrades_the_budget_rather_than_raising(self):
        """Cut inside a zlib-encoded key array, the budget's census raised
        ``zlib.error`` out of ``fbx_to_glb`` -- past its documented
        exceptions and every task manager's catch -- before the conversion
        was attempted (2026-10-04 review). A budget is never the reason a
        conversion is not attempted."""
        from test_export_verify import build_truncated_fbx

        cut = build_truncated_fbx(os.path.join(self.tmp, "cut.fbx"))
        self.assertEqual(MeshConvert.bake_node_frames(cut), 0)
        self.assertEqual(
            MeshConvert.conversion_timeout(cut), MeshConvert.DEFAULT_TIMEOUT
        )

    def test_describe_texture_pass_owns_up_to_the_power_of_two_snap(self):
        """``max_size=0`` means "never CLAMP", not "never resample".

        KTX2 snaps every non-exempt image down to a power of two regardless of
        the ceiling, because KHR_texture_basisu needs multiple-of-4 edges and a
        full mip pyramid. Reporting "pixels untouched" over images this pass
        just downsampled is the exact false claim the wording exists to remove
        -- and the misreport lands on the KTX2 delivery mode, where a silently
        halved 3000px map is precisely what someone would go looking for.
        """
        full = {"images": 4, "bytes_before": 13.1e6, "bytes_after": 9.2e6}

        snapped = MeshConvert.describe_texture_pass({**full, "resized": 2}, "KTX2", 0)
        self.assertNotIn("untouched", snapped)
        self.assertIn("2", snapped)
        self.assertIn("power-of-two", snapped)

    def test_max_size_is_a_ceiling_and_the_summary_says_what_it_resampled(self):
        """A ceiling never upscales, and ``resized`` counts what actually moved.

        Pins the pair of claims a delivery log makes on this pass. The mode
        label alone ("resized") reads, to whoever set the ceiling, as "your
        textures were rescaled to 2K" -- so an asset authored AT the ceiling
        has to come back ``resized: 0`` with its pixels untouched, and only a
        genuinely oversized source may be counted.
        """
        import base64

        from PIL import Image

        def png_bytes(size):
            buffer = io.BytesIO()
            Image.new("RGB", (size, size), (200, 180, 40)).save(buffer, format="PNG")
            return buffer.getvalue()

        sizes = {"under.png": 32, "at.png": 64, "over.png": 128}
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "images": [
                    {
                        "name": name,
                        "uri": "data:image/png;base64,"
                        + base64.b64encode(png_bytes(size)).decode("ascii"),
                        "mimeType": "image/png",
                    }
                    for name, size in sizes.items()
                ],
                "textures": [{"source": i} for i in range(len(sizes))],
            }
        )
        summary = MeshConvert.optimize_glb_textures(path, max_size=64)
        self.assertEqual(
            summary["resized"], 1, "only the oversized source may be counted"
        )

        edit = MeshConvert._read_glb(path)
        blob = edit.bin_data
        for image in edit.gltf["images"]:
            view = edit.gltf["bufferViews"][image["bufferView"]]
            raw = bytes(
                blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
            )
            got = Image.open(io.BytesIO(raw)).size
            expected = min(sizes[image["name"]], 64)
            self.assertEqual(
                got,
                (expected, expected),
                f"{image['name']}: a ceiling clamps, never upscales",
            )

    def test_optimize_relocates_data_uris_and_exempts_lightmaps(self):
        """A data-URI image lands in the BIN; a lightmap resists the resize."""
        import base64 as b64

        from PIL import Image

        def png_bytes(size):
            im = Image.new("RGB", (size, size), (200, 180, 40))
            buffer = io.BytesIO()
            im.save(buffer, format="PNG")
            return buffer.getvalue()

        source = png_bytes(128)
        lightmap = png_bytes(128)
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                # A lightmap is found by its binding -- a texCoord-1 occlusion
                # map -- never by its name.
                "materials": [
                    {"name": "M", "occlusionTexture": {"index": 1, "texCoord": 1}}
                ],
                "images": [
                    {
                        "name": "source.png",
                        "uri": "data:image/png;base64,"
                        + b64.b64encode(source).decode("ascii"),
                        "mimeType": "image/png",
                    },
                    {
                        "name": "room_Lightmap.png",
                        "uri": "data:image/png;base64,"
                        + b64.b64encode(lightmap).decode("ascii"),
                        "mimeType": "image/png",
                    },
                ],
                "textures": [{"source": 0}, {"source": 1}],
            }
        )
        summary = MeshConvert.optimize_glb_textures(path, max_size=64)
        self.assertEqual(summary["images"], 2)

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        for image in gltf["images"]:
            self.assertNotIn("uri", image, "data URI should have moved into the BIN")
            self.assertIn("bufferView", image)
        blob = edit.bin_data

        def decode(image):
            view = gltf["bufferViews"][image["bufferView"]]
            raw = bytes(
                blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
            )
            return Image.open(io.BytesIO(raw))

        self.assertEqual(max(decode(gltf["images"][0]).size), 64, "source must resize")
        self.assertEqual(
            max(decode(gltf["images"][1]).size), 128, "lightmap must keep its size"
        )

    def test_optimize_splits_a_shared_bufferview_when_owners_diverge(self):
        """Two images can share ONE bufferView, and then need different bytes.

        FBX2glTF really emits this (measured: 4 such pairs on a production room
        GLB). Co-owners differ in the one thing that decides their encoding --
        a lightmap is exempt from the resize and encodes lossless, its co-owner
        is not -- so recording a single owner per view handed the loser the
        winner's bytes. With the lightmap winning, its co-owner silently kept
        full resolution; with the order reversed, the LIGHTMAP got resized and
        lossy-encoded, which is exactly the corruption the structural exemption
        exists to prevent. Asserted in BOTH orders, because the old bug was
        invisible in one of them.
        """
        import random

        from PIL import Image

        rng = random.Random(11)
        source = Image.new("RGB", (128, 128))
        source.putdata([(rng.randrange(256),) * 3 for _ in range(128 * 128)])
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
        png = buffer.getvalue()

        for names in (
            ["source.png", "room_Lightmap.png"],
            ["room_Lightmap.png", "source.png"],
        ):
            lit = names.index("room_Lightmap.png")
            path = self._write_glb(
                {
                    "asset": {"version": "2.0"},
                    # The lightmap is the image its texCoord-1 occlusion binds.
                    "materials": [
                        {
                            "name": "M",
                            "occlusionTexture": {"index": lit, "texCoord": 1},
                        }
                    ],
                    # Both images point at bufferView 0.
                    "images": [
                        {"bufferView": 0, "mimeType": "image/png", "name": name}
                        for name in names
                    ],
                    "textures": [{"source": i} for i in range(len(names))],
                    "bufferViews": [
                        {"buffer": 0, "byteOffset": 0, "byteLength": len(png)}
                    ],
                    "buffers": [{"byteLength": len(png)}],
                },
                bin_chunk=png + b"\x00" * ((4 - len(png) % 4) % 4),
            )
            MeshConvert.optimize_glb_textures(path, max_size=64)

            edit = MeshConvert._read_glb(path)
            gltf, blob = edit.gltf, edit.bin_data
            by_name = {}
            for image in gltf["images"]:
                view = gltf["bufferViews"][image["bufferView"]]
                by_name[image["name"]] = bytes(
                    blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
                )
            self.assertNotEqual(
                by_name["source.png"],
                by_name["room_Lightmap.png"],
                f"co-owners must not share one encoding (order {names})",
            )
            self.assertEqual(
                max(Image.open(io.BytesIO(by_name["source.png"])).size),
                64,
                f"source must resize (order {names})",
            )
            self.assertEqual(
                max(Image.open(io.BytesIO(by_name["room_Lightmap.png"])).size),
                128,
                f"lightmap must keep its size (order {names})",
            )
            # VP8L is the lossless WebP chunk; VP8 (no L) is the lossy one.
            self.assertEqual(
                by_name["room_Lightmap.png"][12:16],
                b"VP8L",
                f"lightmap must encode lossless (order {names})",
            )

    def test_optimize_is_byte_identical_across_worker_counts(self):
        """The concurrent encode pass must be a pure speedup, not a new variable.

        The encode is ~60% of the pass and Pillow releases the GIL through it,
        so it runs on a thread pool (measured on a production room GLB: 31.8s ->
        7.1s). Threads make the ordering of the per-image work nondeterministic,
        and the pass also *dedupes* identical payloads across images -- so if a
        result were ever attributed to the wrong image index, the artifact would
        differ run to run with nothing failing. Pin the whole file: serial and
        concurrent must produce the same bytes.

        Distinct payloads AND a duplicated one, so both the encode path and the
        dedupe fan-out are covered.
        """
        import random

        from PIL import Image

        def noisy_png(seed, size=96):
            rng = random.Random(seed)
            im = Image.new("RGB", (size, size))
            im.putdata(
                [
                    (rng.randrange(256), rng.randrange(256), rng.randrange(256))
                    for _ in range(size * size)
                ]
            )
            buffer = io.BytesIO()
            im.save(buffer, format="PNG")
            return buffer.getvalue()

        payloads = [noisy_png(seed) for seed in range(5)]
        payloads.append(payloads[0])  # a duplicate: two images, one encode

        views, chunks, offset = [], [], 0
        for png in payloads:
            views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(png)})
            padded = png + b"\x00" * ((4 - len(png) % 4) % 4)
            chunks.append(padded)
            offset += len(padded)
        gltf = {
            "asset": {"version": "2.0"},
            "images": [
                {"bufferView": i, "mimeType": "image/png", "name": f"t{i}.png"}
                for i in range(len(payloads))
            ],
            "textures": [{"source": i} for i in range(len(payloads))],
            "bufferViews": views,
            "buffers": [{"byteLength": offset}],
        }
        bin_chunk = b"".join(chunks)

        digests, summaries = [], []
        for workers in (1, 4):
            path = self._write_glb(json.loads(json.dumps(gltf)), bin_chunk=bin_chunk)
            summaries.append(
                MeshConvert.optimize_glb_textures(path, max_size=48, workers=workers)
            )
            digests.append(hashlib.sha256(open(path, "rb").read()).hexdigest())

        self.assertEqual(
            digests[0],
            digests[1],
            "concurrent encode produced a different GLB than the serial path",
        )
        self.assertEqual(summaries[0], summaries[1])
        self.assertEqual(summaries[0]["images"], len(payloads))

    def test_optimize_encodes_lightmaps_lossless(self):
        """Lightmaps re-encode LOSSLESS (VP8L), colour sources stay lossy (VP8).

        Lossy WebP is YUV 4:2:0 -- on a lightmap's near-black texels the
        chroma quantization reads as magenta/green blotching, and the 2x2
        chroma blocks smear color across atlas rect borders. The exemption
        must therefore cover the ENCODE, not just the resize.

        The source is bound as BASE COLOUR, because the lossy encoder is
        reserved for colour slots (`LOSSY_SAFE_SEMANTICS`): an image no slot
        samples has no semantic and is kept lossless on purpose, the way
        KTX2's ``None`` row takes UASTC.
        """
        import base64 as b64
        import random

        from PIL import Image

        def noisy_png(seed):
            rng = random.Random(seed)
            im = Image.new("RGB", (64, 64))
            im.putdata(
                [
                    (rng.randrange(256), rng.randrange(256), rng.randrange(256))
                    for _ in range(64 * 64)
                ]
            )
            buffer = io.BytesIO()
            im.save(buffer, format="PNG")
            return im, buffer.getvalue()

        lm_img, lm_png = noisy_png(1)
        _src_img, src_png = noisy_png(2)
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "extras": {
                    SceneRecords.LIGHTMAPS.key: {
                        "version": 1,
                        "materials": {"M": {"map": "room_Lightmap.png"}},
                    }
                },
                "images": [
                    {
                        "name": "room_Lightmap.png",
                        "uri": "data:image/png;base64,"
                        + b64.b64encode(lm_png).decode("ascii"),
                        "mimeType": "image/png",
                    },
                    {
                        "name": "source.png",
                        "uri": "data:image/png;base64,"
                        + b64.b64encode(src_png).decode("ascii"),
                        "mimeType": "image/png",
                    },
                ],
                "textures": [{"source": 0}, {"source": 1}],
                "materials": [
                    {
                        "name": "M",
                        "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}},
                    }
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path)

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        blob = edit.bin_data

        def payload(image):
            view = gltf["bufferViews"][image["bufferView"]]
            return bytes(
                blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
            )

        lm_bytes = payload(gltf["images"][0])
        self.assertEqual(lm_bytes[12:16], b"VP8L", "lightmap must be lossless WebP")
        self.assertEqual(
            Image.open(io.BytesIO(lm_bytes)).convert("RGB").tobytes(),
            lm_img.tobytes(),
            "lossless lightmap must round-trip pixel-exact",
        )
        self.assertEqual(
            payload(gltf["images"][1])[12:16], b"VP8 ", "source stays lossy"
        )

    def test_optimize_exempts_by_binding_when_the_name_lies(self):
        """Structural exemption: an image bound as a texCoord-1 occlusion map
        is a lightmap whatever its name says (digest dedupe can hand a
        lightmap payload another image's name), so it must keep its size."""
        import base64 as b64

        from PIL import Image

        im = Image.new("RGB", (128, 128), (10, 12, 14))
        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "images": [
                    {
                        "name": "definitely_not_a_lightmap.png",
                        "uri": "data:image/png;base64,"
                        + b64.b64encode(buffer.getvalue()).decode("ascii"),
                        "mimeType": "image/png",
                    }
                ],
                "textures": [{"source": 0}],
                "materials": [
                    {
                        "name": "M",
                        "occlusionTexture": {"index": 0, "texCoord": 1},
                    }
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path, max_size=64)
        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
        raw = bytes(
            edit.bin_data[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
        )
        self.assertEqual(
            max(Image.open(io.BytesIO(raw)).size),
            128,
            "texCoord-1 occlusion image must not resize",
        )

    def test_prune_unreferenced_textures_drops_the_displaced_converted_orm(self):
        """The image the ORM repack displaces must not ship as dead payload.

        Measured on a production delivery (HOOKS_PINS.glb): after the ORM
        repack rebound both slots to its packed image, FBX2glTF's
        ``ao_met_rough_<mat>`` stayed in ``images``/``textures`` and its
        full-size PNG in the BIN -- 2 MB the reviewer flagged as an orphaned
        texture, per material, per export.
        """
        from PIL import Image

        rough = os.path.join(self.tmp, "rough.png")
        ao = os.path.join(self.tmp, "ao.png")
        Image.new("L", (4, 4), 128).save(rough)
        Image.new("L", (4, 4), 200).save(ao)
        path, geometry = _converted_orm_glb(self.tmp)
        with MeshConvert.open_glb(path) as session:
            MeshConvert.apply_glb_materials(
                session,
                *stated_record(
                    self._maps(rough, ao),
                    {
                        "name": "Room",
                        "metallicFactor": 0.0,
                        "roughnessTexture": "rough.png",
                        "occlusionTexture": "ao.png",
                    },
                ),
            )
            dropped = MeshConvert.prune_glb_unreferenced_textures(session)
        self.assertEqual(dropped["images"], 1)
        gltf = _assert_no_dead_payload(self, path, geometry)
        # And the repacked ORM is still what both slots sample.
        mat = gltf["materials"][0]
        self.assertEqual(
            mat["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"],
            mat["occlusionTexture"]["index"],
        )

    def test_prune_unreferenced_textures_is_a_no_op_on_a_clean_file(self):
        path, geometry = _converted_orm_glb(self.tmp)
        with open(path, "rb") as f:
            before = f.read()
        with MeshConvert.open_glb(path) as session:
            dropped = MeshConvert.prune_glb_unreferenced_textures(session)
        self.assertEqual(dropped, {"textures": 0, "images": 0, "bytes": 0})
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before, "clean file must not be rewritten")

    def test_prune_unreferenced_textures_keeps_the_files_a_record_names(self):
        """A record's files ride as the bufferViews named by them
        (``SceneRecords.GLB_MEDIA``), which no texture index reaches: the
        prune drops an orphan image and renumbers the survivors, and the
        ``shadows`` record's horizon map rides on byte for byte, its record
        untouched. (The v2 block bound its maps by INDEX, which the prune had
        to rewrite -- and once deleted the horizon map and renumbered the
        survivors out from under it.) Added: 2026-11-07"""
        import base64
        import io as iolib

        from PIL import Image

        def _data_uri(color):
            buf = iolib.BytesIO()
            Image.new("RGB", (4, 4), color).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return {"uri": f"data:image/png;base64,{b64}", "mimeType": "image/png"}

        key = SceneRecords.SHADOWS.key
        record = {"version": SceneRecords.SHADOWS.version, "unit_scale": 1.0, "planes": []}
        horizon = b"a horizon map: data, never decoded" * 3
        gltf = {
            "asset": {"version": "2.0"},
            # 0 = orphan (forces the renumber), 1 = sampled.
            "images": [_data_uri(c) for c in ((255, 0, 0), (0, 255, 0))],
            "textures": [{"source": 0}, {"source": 1}],
            "materials": [
                {
                    "name": "Plane",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}},
                }
            ],
            "extras": {key: record},
        }
        path = _write_glb_file(os.path.join(self.tmp, "shadow_prune.glb"), gltf)
        with MeshConvert.open_glb(path) as session:
            MeshConvert._media.put(
                session, "Box_horizon.png", horizon, "image/png"
            )
        with MeshConvert.open_glb(path) as session:
            dropped = MeshConvert.prune_glb_unreferenced_textures(session)
        self.assertEqual((dropped["textures"], dropped["images"]), (1, 1))

        with MeshConvert.open_glb(path) as session:
            out = session.gltf
            view = SceneRecords.GLB_MEDIA.view(out, "Box_horizon.png")
            self.assertIsNotNone(view, "the record's file was collected")
            entry = out["bufferViews"][view]
            start = int(entry.get("byteOffset", 0))
            payload = bytes(session.bin_data[start : start + entry["byteLength"]])
        self.assertEqual(payload, horizon)
        self.assertEqual(out["extras"][key], record)
        self.assertEqual(len(out["textures"]), 1)
        self.assertEqual(
            out["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"], 0
        )

    def test_extras_of_reads_any_owners_extras_as_an_object(self):
        """glTF allows any JSON value under ``extras``, so a reader assuming
        an object raised on a valid file: ``GlbEdit.extras_of`` reads a glTF
        object's -- or a block of ours, which keeps glTF's convention -- in
        place, and ``{}`` for none or for a value that is no object; a write
        to that ``{}`` goes nowhere. Added: 2026-10-20"""
        extras_of = MeshConvert.GlbEdit.extras_of
        held = {"handoff": {"version": 3}}
        self.assertIs(extras_of({"name": "n", "extras": held}), held)
        for owner in (
            {"name": "n"},
            {"extras": None},
            {"extras": ["source"]},
            {"extras": "text"},
            {"extras": 3},
            None,
            ["not", "an", "object"],
        ):
            with self.subTest(owner=owner):
                self.assertEqual(extras_of(owner), {})
        bare = {"name": "n"}
        extras_of(bare)["written"] = True
        self.assertEqual(bare, {"name": "n"}, "a write to the stand-in goes nowhere")

    def test_prune_unreferenced_textures_keeps_a_view_another_owner_reads(self):
        """A bufferView shared with a live image (FBX2glTF does emit two images
        on one view) or with an accessor stays; only its exclusive owner goes."""
        path, geometry = _converted_orm_glb(self.tmp)
        with MeshConvert.open_glb(path) as session:
            gltf = session.gltf
            # A second, LIVE image on the same view, sampled as base colour.
            gltf["images"].append(
                {"name": "twin", "bufferView": 0, "mimeType": "image/png"}
            )
            gltf["textures"].append({"source": 1, "sampler": 0})
            mat = gltf["materials"][0]
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 1}
            # Unbind the converted ORM from both slots.
            del mat["pbrMetallicRoughness"]["metallicRoughnessTexture"]
            del mat["occlusionTexture"]
            session.dirty = True
            dropped = MeshConvert.prune_glb_unreferenced_textures(session)
        self.assertEqual((dropped["textures"], dropped["images"]), (1, 1))
        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        self.assertEqual([i["name"] for i in gltf["images"]], ["twin"])
        self.assertEqual(len(gltf["bufferViews"]), 2, "the shared view must stay")
        self.assertEqual(
            gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"],
            0,
        )
        view = gltf["bufferViews"][gltf["accessors"][0]["bufferView"]]
        self.assertEqual(
            bytes(
                edit.bin_data[
                    view["byteOffset"] : view["byteOffset"] + view["byteLength"]
                ]
            ),
            geometry,
        )

    def test_prune_unreferenced_textures_bails_under_an_image_referring_extension(self):
        """EXT_lights_image_based names images from the root; pruning under it
        would renumber indices the walk cannot see. Left alone, loudly."""
        path, geometry = _converted_orm_glb(self.tmp)
        with MeshConvert.open_glb(path) as session:
            gltf = session.gltf
            gltf["extensionsUsed"] = ["EXT_lights_image_based"]
            gltf["extensions"] = {
                "EXT_lights_image_based": {"lights": [{"specularImages": [[0]]}]}
            }
            del gltf["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"]
            del gltf["materials"][0]["occlusionTexture"]
            session.dirty = True
            dropped = MeshConvert.prune_glb_unreferenced_textures(session)
        self.assertEqual(dropped, {"textures": 0, "images": 0, "bytes": 0})
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 1)
        self.assertEqual(len(gltf["bufferViews"]), 2)

    def test_prune_unreferenced_textures_bails_when_a_view_lives_outside_buffer_0(self):
        """The BIN rebuild re-slices every kept view out of the embedded
        buffer; a view on an external buffer would keep ``buffer`` but get a
        byteOffset into the rebuilt BIN. Bail whole, like the extension case."""
        path, geometry = _converted_orm_glb(self.tmp)
        with MeshConvert.open_glb(path) as session:
            gltf = session.gltf
            gltf.setdefault("buffers", []).append({"uri": "ext.bin", "byteLength": 4})
            gltf["bufferViews"].append({"buffer": 1, "byteOffset": 0, "byteLength": 4})
            del gltf["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"]
            del gltf["materials"][0]["occlusionTexture"]
            session.dirty = True
            dropped = MeshConvert.prune_glb_unreferenced_textures(session)
        self.assertEqual(dropped, {"textures": 0, "images": 0, "bytes": 0})
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 1)
        self.assertEqual(len(gltf["bufferViews"]), 3)

    def test_metallic_roughness_shared_sources_pack_once(self):
        """N materials naming the same maps must cost one pack + one embed."""
        from PIL import Image

        rough = os.path.join(self.tmp, "r2.png")
        Image.new("L", (4, 4), 90).save(rough)
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [{"name": "A"}, {"name": "B"}],
            }
        )
        spec = {"roughnessTexture": "r2.png"}
        with MeshConvert.open_glb(path) as session:
            records = MeshConvert.apply_glb_materials(
                session,
                *stated_record(
                    self._maps(rough), dict(spec, name="A"), dict(spec, name="B")
                ),
            )
        self.assertEqual([r["orm"] for r in records], [True, True])
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 1, "shared sources embedded twice")

    def test_an_out_of_range_index_is_skipped_not_wrapped(self):
        """A negative glTF index must be rejected, not resolved from the end.

        Indices are non-negative by spec, so a negative one means a malformed
        file — but Python indexing would quietly hand back the *last* texture
        and report a finding against a material that has nothing wrong with it.
        """
        path = self._write_glb(
            {
                "asset": {"version": "2.0"},
                "materials": [
                    {
                        "name": "Bad",
                        "alphaMode": "BLEND",
                        "pbrMetallicRoughness": {"baseColorTexture": {"index": -1}},
                    }
                ],
                "textures": [{"source": 0}],
                "images": [{"name": "real.png"}],
            }
        )
        self.assertEqual(MeshConvert.check_glb_materials(path), [])

    def test_a_raising_body_writes_nothing(self):
        """A half-applied edit must not reach disk."""
        path = self._write_glb(
            {"asset": {"version": "2.0"}, "materials": [{"name": "Body"}]}
        )
        before = open(path, "rb").read()
        with self.assertRaises(RuntimeError):
            with MeshConvert.open_glb(path) as session:
                MeshConvert.apply_glb_materials(
                    session, self._record(Body={"emissiveFactor": [1.0, 1.0, 1.0]})
                )
                raise RuntimeError("interrupted mid-edit")
        self.assertEqual(open(path, "rb").read(), before)


class TestEncodeOrder(unittest.TestCase):
    """The GLB texture pass starts its longest encode first (2026-09-14): a
    pool fed in insertion order ran a 260 s UASTC+RDO encode alone at the
    tail of a 302 s pass."""

    def test_jobs_are_ordered_largest_source_first(self):
        jobs = {
            ("a", False, None): b"x" * 10,
            ("b", False, None): b"x" * 300,
            ("c", True, None): b"x" * 20,
        }
        self.assertEqual(
            MeshConvert._largest_first(jobs),
            [("b", False, None), ("c", True, None), ("a", False, None)],
        )
        self.assertEqual(MeshConvert._largest_first({}), [])


class TestDedupeGlbImages(unittest.TestCase):
    """One copy per distinct payload, however many materials wired the file.

    FBX2glTF embeds per MATERIAL, so two Maya materials sharing one normal map
    arrive as two byte-identical images. The session's own ``image_digests``
    dedupe is write-side only -- it stops a WRITER appending a copy of
    something already embedded, and cannot see a pair the converter itself
    produced. Measured on a production assembly: 2 duplicate pairs, 154.7 KB of
    the delivered 5.9 MB, and the texture pass paid to decode, resize and
    re-encode both copies of each.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_dedupe_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _twinned(self, name="twins.glb", second_payload=None):
        """A GLB whose two materials each sample their OWN copy of one map."""
        import io as iolib

        from PIL import Image

        buf = iolib.BytesIO()
        Image.new("RGB", (4, 4), (10, 200, 30)).save(buf, format="PNG")
        blob = buf.getvalue()
        twin = second_payload if second_payload is not None else blob
        pad = lambda b: b + b"\x00" * ((4 - (len(b) % 4)) % 4)  # noqa: E731
        first, second = pad(blob), pad(twin)
        gltf = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": len(first) + len(second)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": len(blob)},
                {"buffer": 0, "byteOffset": len(first), "byteLength": len(twin)},
            ],
            "images": [
                {"name": "N_OpenGL.png", "bufferView": 0, "mimeType": "image/png"},
                {"name": "N_OpenGL.png", "bufferView": 1, "mimeType": "image/png"},
            ],
            "textures": [{"source": 0}, {"source": 1}],
            "materials": [
                {
                    "name": "cabinet",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                },
                {
                    "name": "cabinet1",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}},
                },
            ],
        }
        return _write_glb_file(os.path.join(self.tmp, name), gltf, first + second)

    def test_identical_payloads_collapse_onto_one_image(self):
        path = self._twinned()
        dropped = MeshConvert.dedupe_glb_images(path)
        self.assertEqual(dropped["images"], 1)
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 1)
        # Both materials still render: their textures now share the survivor.
        sources = {t["source"] for t in gltf["textures"]}
        self.assertEqual(sources, {0}, "both textures must sample the survivor")
        self.assertEqual(len(gltf["textures"]), 2, "textures themselves are kept")

    def test_the_reclaimed_bytes_are_reported(self):
        path = self._twinned()
        before = os.path.getsize(path)
        dropped = MeshConvert.dedupe_glb_images(path)
        self.assertGreater(dropped["bytes"], 0)
        self.assertLess(os.path.getsize(path), before)

    def test_images_that_merely_share_a_name_are_left_alone(self):
        """Content-addressed, never name-addressed: two DIFFERENT maps exported
        from materials with colliding names must both survive."""
        import io as iolib

        from PIL import Image

        buf = iolib.BytesIO()
        Image.new("RGB", (4, 4), (200, 10, 30)).save(buf, format="PNG")
        path = self._twinned(second_payload=buf.getvalue())
        self.assertEqual(MeshConvert.dedupe_glb_images(path)["images"], 0)
        self.assertEqual(len(MeshConvert._read_glb(path).gltf["images"]), 2)

    def test_it_refuses_where_the_PRUNE_would_refuse(self):
        """The rebind is only sound if the orphans it creates can be collected.

        `prune_glb_unreferenced_textures` bails whole on a file whose images
        are referenced from outside the material tree. Rebinding into that
        leaves payload nothing can ever reach -- worse than the duplication it
        set out to remove -- and reports having removed nothing while having
        rewritten the file.
        """
        path = self._twinned(name="foreign.glb")
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["extensionsUsed"] = ["EXT_lights_image_based"]
            edit.dirty = True
        before = open(path, "rb").read()

        self.assertEqual(MeshConvert.dedupe_glb_images(path), {"images": 0, "bytes": 0})
        self.assertEqual(
            open(path, "rb").read(), before, "a refusal must not rewrite the file"
        )
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(
            [t["source"] for t in gltf["textures"]],
            [0, 1],
            "both textures must still sample their own image",
        )

    def test_a_file_with_nothing_to_collapse_is_not_rewritten(self):
        path, _ = _converted_orm_glb(self.tmp)
        before = open(path, "rb").read()
        self.assertEqual(MeshConvert.dedupe_glb_images(path), {"images": 0, "bytes": 0})
        self.assertEqual(open(path, "rb").read(), before)

    def test_the_conversion_collapses_what_the_converter_duplicated(self):
        """The pass has to run where every deliverable goes through, not only
        where a caller remembers it: both the preview and the exporters reach
        the GLB via ``fbx_to_glb``."""
        path = self._twinned(name="conv.glb")

        def _run(cmd, **kw):
            import shutil as sh

            sh.copyfile(path, cmd[cmd.index("-o") + 1] + ".glb")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        src = os.path.join(self.tmp, "in.fbx")
        open(src, "wb").write(b"fbx")
        with (
            unittest.mock.patch.object(
                MeshConvert, "resolve_binary", return_value="FBX2glTF"
            ),
            unittest.mock.patch("subprocess.run", side_effect=_run),
        ):
            out = MeshConvert.fbx_to_glb(src, overwrite=True, prompt=False)
        self.assertEqual(len(MeshConvert._read_glb(out).gltf["images"]), 1)


@unittest.skipUnless(
    os.environ.get("PYTHONTK_INTEGRATION_TESTS") == "1",
    "Set PYTHONTK_INTEGRATION_TESTS=1 to run network/install integration tests.",
)
class TestRealInstall(unittest.TestCase):
    """End-to-end install. Downloads FBX2glTF (~3.7 MB)."""

    def test_install_and_invoke_help(self):
        binary = MeshConvert.resolve_binary(auto_install=True, prompt=False)
        self.assertTrue(os.path.isfile(binary), f"binary missing: {binary}")
        result = subprocess.run(
            [binary, "--help"], capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("FBX2glTF", result.stdout)


class TestGlbHandoff(unittest.TestCase):
    """A GLB's ``extras.handoff``: the standalone-reader contract.

    The block every carrier publishes (``HandoffRecord``; what it says is
    ``test_handoff_record``'s), as a GLB writes it: named against the
    root-``extras`` blocks the file carries, and for a GLB built from an FBX
    derived from that FBX's own record. It replaced the
    ``extras.scene_sidecar`` envelope on 2026-10-20.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_handoff_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _glb(self, extras=None, name="scene.glb"):
        """A JSON-chunk-only GLB with one material and *extras* at its root."""
        gltf = {"asset": {"version": "2.0"}, "materials": [{"name": "m"}]}
        if extras:
            gltf["extras"] = extras
        return _write_glb_file(os.path.join(self.tmp, name), gltf)

    @staticmethod
    def _embedded(glb, source=None):
        """Embed the GLB's handoff block in *glb*; the block read back."""
        with MeshConvert.open_glb(glb) as edit:
            MeshConvert.embed_glb_handoff(edit, HandoffRecord.block("glb", [], source))
        return MeshConvert.read_glb_handoff(glb)

    @staticmethod
    def _parts(block):
        return [row["part"] for row in block["assembly"]]

    def _converted_with(self, channels, name="converted.glb"):
        """What FBX2glTF hands over for an FBX whose ``data_export`` carrier
        holds *channels*: each a user property under the node's
        ``extras.fromFBX.userProperties``, beside a mesh node of its own."""
        props = {
            key: {"type": "eFbxString", "value": json.dumps(value)}
            for key, value in channels.items()
        }
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0, 1]}],
            "nodes": [
                {
                    "name": "box",
                    "extras": {
                        "fromFBX": {
                            "userProperties": {
                                "currentUVSet": {"type": "eFbxString", "value": "map1"}
                            }
                        }
                    },
                },
                {"name": "data_export", "extras": {"fromFBX": {"userProperties": props}}},
            ],
        }
        return _write_glb_file(os.path.join(self.tmp, name), gltf)

    def _native(self, scene_extras, channels=None, name="native.glb"):
        """A GLB as a host's own glTF exporter writes it: each block of ours
        JSON text in its first scene's extras (*scene_extras*), and the
        carrier's records (*channels*) at the top of the carrier node's
        extras, each as JSON text."""
        records = {key: json.dumps(value) for key, value in (channels or {}).items()}
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0, 1], "extras": dict(scene_extras)}],
            "nodes": [
                {"name": "box", "mesh": 0},
                {"name": "data_export", "extras": records},
            ],
            "meshes": [{"primitives": [{"attributes": {}, "material": 0}]}],
            "materials": [{"name": "m"}],
        }
        return _write_glb_file(os.path.join(self.tmp, name), gltf)

    # ------------------------------------------------------------ embedding
    def test_embedding_names_exactly_the_blocks_the_file_carries(self):
        """Each block's row is taken from the file -- never a block it lacks,
        nor one it has -- between the file's own content and the lighting,
        whose rows read the lightmaps where the file holds them; a pass that
        changes the blocks after the handoff is written finishes it again,
        idempotently."""
        from pythontk import SceneRecords

        lightmaps = SceneRecords.LIGHTMAPS
        shadows = SceneRecords.SHADOWS
        shots = SceneRecords.SHOTS
        # Each row states what its block calls for: the blocks themselves.
        held = {
            lightmaps.key: _lightmaps_record("m"),
            shadows.key: {"version": shadows.version, "unit_scale": 1.0, "planes": []},
        }
        glb = self._glb(held)
        block = self._embedded(glb, {"application": "maya", "version": "2025"})
        self.assertEqual(
            block["extras"], {"source": {"application": "maya", "version": "2025"}}
        )
        self.assertEqual(block["assembly"], HandoffRecord.assembly("glb", held))
        self.assertEqual(
            self._parts(block)[:5],
            [
                "geometry",
                SceneRecords.MATERIALS.key,
                SceneRecords.VISIBILITY.key,
                lightmaps.key,
                shadows.key,
            ],
        )
        rows = {row["part"]: row for row in block["assembly"]}
        self.assertEqual(rows[shadows.key]["data"], [f"extras.{shadows.key}"])
        # Shadows v3 rides as the record itself: its row is the record's.
        self.assertEqual(
            rows[shadows.key]["spec"], shadows.assembly_for(held[shadows.key])
        )
        self.assertEqual(
            rows["environment"]["data"],
            ["rendering.environment", f"extras.{lightmaps.key}"],
        )
        # The same keys as the FBX's record, each in its place.
        fbx = HandoffRecord.block("fbx", list(held), source={"application": "maya"})
        self.assertEqual(set(block), set(fbx))
        self.assertEqual(set(block["extras"]), set(fbx["extras"]))

        animation = {"version": shots.version, "fps": 30.0, "shots": []}
        with MeshConvert.open_glb(glb) as edit:
            extras = edit.gltf["extras"]
            extras[shots.key] = animation
            del extras[shadows.key]
            self.assertTrue(MeshConvert._finish_glb_handoff(edit))
            finished = json.dumps(extras[HandoffRecord.KEY], sort_keys=True)
            MeshConvert._finish_glb_handoff(edit)
            self.assertEqual(
                json.dumps(extras[HandoffRecord.KEY], sort_keys=True), finished
            )
        block = MeshConvert.read_glb_handoff(glb)
        self.assertEqual(
            block["assembly"],
            HandoffRecord.assembly(
                "glb", {lightmaps.key: held[lightmaps.key], shots.key: animation}
            ),
        )
        self.assertNotIn(shadows.key, self._parts(block))

    def test_a_file_carrying_no_block_names_none(self):
        """Nothing carried and no source given: the file's own content and the
        lighting alone, and no ``extras``; a file with no handoff has nothing
        to finish or read back."""
        block = self._embedded(self._glb())
        self.assertNotIn("extras", block)
        self.assertEqual(block["assembly"], HandoffRecord.assembly("glb", []))
        bare = self._glb(name="bare.glb")
        with MeshConvert.open_glb(bare) as edit:
            self.assertFalse(MeshConvert._finish_glb_handoff(edit))
        self.assertIsNone(MeshConvert.read_glb_handoff(bare))

    # -------------------------------------------------- built from an FBX
    def test_a_converted_glb_says_what_its_fbx_says(self):
        """No envelope -- every export's case: the handoff is the one the
        FBX's own record implies, its source and each lighting value it
        states other than the policy, named against the GLB's own blocks;
        and the record itself leaves with the carrier it rode."""
        chosen = {"renderer": {"toneMappingExposure": 1.4}}
        record = HandoffRecord.block(
            "fbx",
            HandoffRecord.carried(["animation"]),
            source={"application": "maya", "version": "2025"},
            rendering=chosen,
        )
        out = _convert_prebuilt(self.tmp, self._converted_with({"handoff": record}))
        block = MeshConvert.read_glb_handoff(out)
        self.assertEqual(block["version"], HandoffRecord.VERSION)
        self.assertEqual(
            block["instructions"],
            HandoffRecord.instructions("glb"),
            "the GLB's own contract, not the FBX's",
        )
        self.assertEqual(
            block["extras"], {"source": {"application": "maya", "version": "2025"}}
        )
        # As this file uses the recipe: it carries no block.
        self.assertEqual(block["rendering"], HandoffRecord.rendering_policy(chosen, {}))
        self.assertEqual(block["rendering"]["renderer"]["toneMappingExposure"], 1.4)
        self.assertEqual(
            block["assembly"],
            HandoffRecord.assembly("glb", {}),
            "the FBX's records are not this file's blocks",
        )
        with MeshConvert.open_glb(out) as edit:
            self.assertEqual([n.get("name") for n in edit.gltf["nodes"]], ["box"])
            self.assertEqual(edit.gltf["scenes"][0]["nodes"], [0])
            self.assertIsNone(
                MeshConvert.data_export_channel(edit.gltf, HandoffRecord.KEY)
            )

    def test_a_handoff_record_of_another_version_is_refused_and_said(self):
        """The FBX's handoff record is read in the version and shape it
        declares (``_current_record``): a version-2 record -- its source at
        its top, prose beside its numbers -- is no record this reader reads,
        never read through a branch for its shape. The GLB claims no approved
        scene, as a raw export's does, and the log says why: the cure is a
        re-export. Added: 2026-10-20"""
        record = {
            "version": 2,
            "source": {"application": "blender", "version": "4.2"},
            "rendering": {"keyLight": {"color": "#ffffff", "intensity": 0.9}},
        }
        with self.assertLogs("pythontk.file_utils.mesh_convert", level="WARNING") as logs:
            out = _convert_prebuilt(
                self.tmp, self._converted_with({HandoffRecord.KEY: record})
            )
        self.assertIsNone(MeshConvert.read_glb_handoff(out))
        self.assertTrue(
            any(
                f"{HandoffRecord.KEY} record not applied -- {HandoffRecord.KEY}: v2 "
                in line
                for line in logs.output
            ),
            logs.output,
        )

    def test_a_converted_glb_carries_none_of_the_converters_notes(self):
        """End to end: the conversion's last step leaves the file only its own
        content and our declared blocks. Added: 2026-10-20"""
        record = HandoffRecord.block("fbx", [], {"application": "maya"})
        converted = self._converted_with({HandoffRecord.KEY: record})
        with MeshConvert.open_glb(converted) as edit:
            edit.gltf["materials"] = [
                {
                    "name": "m",
                    "extras": {
                        "fromFBX": {"shadingModel": "lambert", "isTruePBR": False}
                    },
                }
            ]
            edit.dirty = True
        out = _convert_prebuilt(self.tmp, converted)
        with MeshConvert.open_glb(out) as edit:
            gltf = edit.gltf
        self.assertNotIn("fromFBX", json.dumps(gltf))
        self.assertEqual(sorted(gltf["extras"]), [HandoffRecord.KEY])
        self.assertTrue(MeshConvert.read_glb_handoff(out))

    def test_a_file_with_no_handoff_record_gets_none(self):
        """A raw export claims no approved scene, so nothing declares one."""
        out = _convert_prebuilt(self.tmp, self._converted_with({}))
        self.assertIsNone(MeshConvert.read_glb_handoff(out))

    def test_the_exports_lighting_choice_lays_over_the_fbxs(self):
        """``rendering=`` -- a preview's own rows -- wins where it speaks, and
        the FBX's other choices stand."""
        record = HandoffRecord.block(
            "fbx",
            HandoffRecord.carried(["animation"]),
            rendering={
                "keyLight": {"intensity": 0.6},
                "renderer": {"toneMappingExposure": 1.4},
            },
        )
        out = _convert_prebuilt(
            self.tmp,
            self._converted_with({"handoff": record}),
            rendering={"keyLight": {"intensity": 0.4}},
        )
        rendering = MeshConvert.read_glb_handoff(out)["rendering"]
        self.assertEqual(rendering["keyLight"]["intensity"], 0.4)
        self.assertEqual(rendering["renderer"]["toneMappingExposure"], 1.4)

    def test_a_handoff_handed_in_is_embedded_in_place_of_the_fbxs(self):
        """``sidecar=`` is a handoff block to embed instead of the one the
        FBX's record implies."""
        record = HandoffRecord.block(
            "fbx",
            HandoffRecord.carried(["animation"]),
            source={"application": "maya"},
        )
        out = _convert_prebuilt(
            self.tmp,
            self._converted_with({"handoff": record}),
            sidecar=HandoffRecord.block("glb", [], {"application": "preview"}),
        )
        self.assertEqual(
            MeshConvert.read_glb_handoff(out)["extras"]["source"],
            {"application": "preview"},
        )

    # ------------------------------------------------- a host's own export
    def test_finish_glb_states_a_native_export_as_every_deliverable(self):
        """What the FBX route's build does at its end, for a GLB a host's own
        exporter wrote: its blocks move from the first scene's extras to the
        root's as objects -- a scene property of its own staying put -- the
        handoff its carrier's record implies is embedded and named against
        them, and the records leave with their carrier. A second run finds
        nothing to do. Added: 2026-10-20"""
        from pythontk import SceneRecords

        lightmaps = _lightmaps_record("box")
        record = HandoffRecord.block(
            "fbx",
            HandoffRecord.carried(["lightmaps"]),
            source={"application": "blender", "version": "4.2"},
            rendering={"keyLight": {"intensity": 0.4}},
        )
        glb = self._native(
            {SceneRecords.LIGHTMAPS.key: json.dumps(lightmaps), "theirs": "kept"},
            channels={HandoffRecord.KEY: record, "lightmaps": lightmaps},
        )
        self.assertTrue(MeshConvert.finish_glb(glb))
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
            before = json.dumps(gltf, sort_keys=True)
        self.assertEqual(gltf["scenes"][0]["extras"], {"theirs": "kept"})
        self.assertEqual(gltf["extras"][SceneRecords.LIGHTMAPS.key], lightmaps)
        handoff = gltf["extras"][HandoffRecord.KEY]
        self.assertEqual(handoff["instructions"], HandoffRecord.instructions("glb"))
        self.assertEqual(
            handoff["extras"], {"source": {"application": "blender", "version": "4.2"}}
        )
        held = {SceneRecords.LIGHTMAPS.key: lightmaps}
        self.assertEqual(
            handoff["rendering"],
            HandoffRecord.rendering_policy({"keyLight": {"intensity": 0.4}}, held),
        )
        self.assertEqual(handoff["assembly"], HandoffRecord.assembly("glb", held))
        self.assertEqual([n.get("name") for n in gltf["nodes"]], ["box"])
        self.assertEqual(gltf["scenes"][0]["nodes"], [0])

        self.assertFalse(MeshConvert.finish_glb(glb), "idempotent")
        with MeshConvert.open_glb(glb) as edit:
            self.assertEqual(json.dumps(edit.gltf, sort_keys=True), before)

    def test_finish_glb_names_what_it_lifts_in_a_handoff_already_there(self):
        """A file whose handoff a host embedded before its blocks were lifted
        has it finished against them; one whose carrier holds no handoff
        record gets none -- a raw export claims no approved scene -- and its
        blocks are lifted all the same. Added: 2026-10-20"""
        from pythontk import SceneRecords

        shadows = {
            "version": SceneRecords.SHADOWS.version,
            "unit_scale": 1.0,
            "planes": [],
        }
        scene = {SceneRecords.SHADOWS.key: json.dumps(shadows)}
        raw = self._native(scene, name="raw.glb")
        self.assertTrue(MeshConvert.finish_glb(raw))
        self.assertIsNone(MeshConvert.read_glb_handoff(raw))
        with MeshConvert.open_glb(raw) as edit:
            self.assertEqual(edit.gltf["extras"], {SceneRecords.SHADOWS.key: shadows})
            self.assertNotIn("extras", edit.gltf["scenes"][0])

        stated = self._native(scene, name="stated.glb")
        with MeshConvert.open_glb(stated) as edit:
            MeshConvert.embed_glb_handoff(edit, HandoffRecord.block("glb", []))
        self.assertTrue(MeshConvert.finish_glb(stated))
        self.assertEqual(
            MeshConvert.read_glb_handoff(stated)["assembly"],
            HandoffRecord.assembly("glb", {SceneRecords.SHADOWS.key: shadows}),
        )

    def test_lifting_moves_only_our_blocks_under_their_current_key(self):
        """``_lift_scene_blocks``: each block of ours in the first scene's
        extras -- JSON text or an object -- moves to the root's as an object,
        under the record's key. A scene property that is no block of ours --
        not an object stating its version, whatever its name -- stays where
        it is, and a second run moves nothing. Added: 2026-10-20"""
        from pythontk import SceneRecords

        shadows = {
            "version": SceneRecords.SHADOWS.version,
            "unit_scale": 1.0,
            "planes": [],
        }
        rig = {"version": 1, "rigs": []}
        theirs = {
            SceneRecords.SHOTS.key: json.dumps({"clips": "a property of theirs"}),
            SceneRecords.LIGHTMAPS.key: "not json",
            "note": "theirs",
        }
        gltf = {
            "scenes": [
                {
                    "nodes": [],
                    "extras": {
                        SceneRecords.SHADOWS.key: json.dumps(shadows),
                        SceneRecords.ARTICULATION.key: rig,
                        **theirs,
                    },
                }
            ],
            "extras": {"root": 1},
        }
        self.assertTrue(MeshConvert._lift_scene_blocks(gltf))
        self.assertEqual(
            gltf["extras"],
            {
                "root": 1,
                SceneRecords.SHADOWS.key: shadows,
                SceneRecords.ARTICULATION.key: rig,
            },
        )
        self.assertEqual(gltf["scenes"][0]["extras"], theirs)
        self.assertFalse(MeshConvert._lift_scene_blocks(gltf), "idempotent")

        emptied = {
            "scenes": [{"nodes": [], "extras": {SceneRecords.SHADOWS.key: shadows}}]
        }
        self.assertTrue(MeshConvert._lift_scene_blocks(emptied))
        self.assertEqual(
            emptied,
            {
                "scenes": [{"nodes": []}],
                "extras": {SceneRecords.SHADOWS.key: shadows},
            },
        )
        self.assertFalse(MeshConvert._lift_scene_blocks({"scenes": []}))


class TestSceneSidecar(unittest.TestCase):
    """The retired scene-sidecar envelope (2026-10-20).

    What a host that predates the ``materials`` record still sends: the file
    carries the handoff its source and lighting make (``extras.handoff``),
    never the envelope; its sections are read no more (their appliers were
    removed 2026-10-31 -- the FBX's ``materials`` record states every
    material).
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_sidecar_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write_glb(self, name="scene.glb", materials=None):
        """A JSON-chunk-only GLB with named materials (no BIN needed)."""
        gltf = {"asset": {"version": "2.0"}}
        if materials is not None:
            gltf["materials"] = materials
        payload = json.dumps(gltf).encode("utf-8")
        payload += b" " * ((4 - (len(payload) % 4)) % 4)
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(
                b"glTF"
                + struct.pack("<I", 2)
                + struct.pack("<I", 12 + 8 + len(payload))
                + struct.pack("<I", len(payload))
                + b"JSON"
                + payload
            )
        return path

    def _envelope(self, sections):
        with self.assertWarns(DeprecationWarning):
            return MeshConvert.build_scene_sidecar(
                sections,
                source={"application": "test", "version": "0"},
                asset="scene.fbx",
            )

    def _assert_embeds(self, glb, envelope):
        """The GLB carries the envelope's handoff and nothing else of it --
        named against the file's blocks (none here) as every handoff is.

        The sections are written into the materials, not carried, and the
        per-section outcome is the build's report, not the file's; the
        caller's envelope is left as built.
        """
        with MeshConvert.open_glb(glb) as edit:
            extras = edit.gltf.get("extras") or {}
        self.assertNotIn("scene_sidecar", extras)
        self.assertNotIn("scene_sidecar_applied", extras)
        self.assertEqual(
            extras[HandoffRecord.KEY],
            HandoffRecord.block("glb", {}, envelope["source"]),
        )
        self.assertEqual(
            set(envelope),
            {"version", "source", "asset", "sections", "handoff"},
            "the caller's envelope was mutated",
        )
        return extras[HandoffRecord.KEY]

    def test_build_scene_sidecar_wraps_the_glb_handoff(self):
        """What a host that predates the materials record builds: its
        sections, and the GLB handoff its source and lighting make."""
        envelope = self._envelope({"emissive": {"m": {"color": [1, 0, 0]}}})
        self.assertEqual(
            set(envelope), {"version", "source", "asset", "sections", "handoff"}
        )
        self.assertEqual(envelope["version"], 2)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(envelope["version"], MeshConvert.SIDECAR_VERSION)
        self.assertEqual(envelope["asset"], "scene.fbx")
        # The file's blocks are not known when it is built: the block states
        # them whole, for the file it is embedded in to name.
        self.assertEqual(
            envelope["handoff"], HandoffRecord.block("glb", None, envelope["source"])
        )

    def test_the_retired_reader_reads_the_handoff_alone(self):
        """``read_scene_sidecar`` (retired 2026-10-20) answers with
        ``extras.handoff``; the envelope a file built before it carries is
        read by no reader -- such a file is re-exported."""
        glb = self._write_glb(materials=[{"name": "m"}])
        with MeshConvert.open_glb(glb) as edit:
            edit.gltf["extras"] = {"scene_sidecar": {"version": 2, "sections": {}}}
            edit.dirty = True
        with self.assertWarns(DeprecationWarning):
            self.assertIsNone(MeshConvert.read_scene_sidecar(glb))
        self.assertIsNone(MeshConvert.read_glb_handoff(glb))

        current = self._write_glb("current.glb", materials=[{"name": "m"}])
        with MeshConvert.open_glb(current) as edit:
            MeshConvert.embed_glb_handoff(
                edit, HandoffRecord.block("glb", None, self._envelope({})["source"])
            )
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(
                MeshConvert.read_scene_sidecar(current),
                MeshConvert.read_glb_handoff(current),
            )

    def test_an_envelopes_sections_are_read_no_more(self):
        """An older host's envelope, on a file carrying no materials record:
        its handoff is embedded and its sections repair nothing -- their
        appliers were removed 2026-10-31. Added: 2026-10-31"""
        src = os.path.join(self.tmp, "model.fbx")
        with open(src, "wb") as fh:
            fh.write(b"fake-fbx")
        dst = os.path.join(self.tmp, "model.glb")
        envelope = self._envelope({"emissive": {"m": {"color": [0, 1, 0]}}})

        def fake_run(cmd, **kwargs):
            # FBX2glTF writes where `-o` points; fbx_to_glb moves it onto dst.
            self._write_glb("converted.glb", materials=[{"name": "m"}])
            shutil.copyfile(
                os.path.join(self.tmp, "converted.glb"),
                cmd[cmd.index("-o") + 1] + ".glb",
            )

            class R:
                returncode = 0
                stdout = ""
                stderr = ""

            return R()

        report = {}
        with (
            patch("shutil.which", return_value=os.path.join(self.tmp, "bin")),
            patch("subprocess.run", side_effect=fake_run),
        ):
            out = MeshConvert.fbx_to_glb(
                src, dst, overwrite=True, sidecar=envelope, report=report
            )

        self._assert_embeds(out, envelope)
        self.assertNotIn("sidecar", report, "no section applied, none reported")
        with MeshConvert.open_glb(out) as edit:
            self.assertNotIn("emissiveFactor", edit.gltf["materials"][0])

    def test_a_file_that_states_its_materials_takes_only_the_envelopes_handoff(self):
        """The ``materials`` record states every material, so an envelope's
        sections repair nothing there: its handoff is embedded, its sections
        are not applied, and the record leaves the file with its carrier."""
        envelope = self._envelope({"emissive": {"m": {"color": [0, 1, 0]}}})
        converted = _write_glb_file(
            os.path.join(self.tmp, "stated.glb"),
            {
                "asset": {"version": "2.0"},
                "scene": 0,
                "scenes": [{"nodes": [0]}],
                "nodes": [
                    {
                        "name": "data_export",
                        "extras": {
                            MeshConvert.MATERIALS_KEY: json.dumps(
                                {"version": 1, "materials": []}
                            )
                        },
                    }
                ],
                "materials": [{"name": "m"}],
            },
        )
        report = {}
        out = _convert_prebuilt(self.tmp, converted, sidecar=envelope, report=report)
        self._assert_embeds(out, envelope)
        self.assertNotIn("sidecar", report)
        with MeshConvert.open_glb(out) as edit:
            self.assertNotIn("emissiveFactor", edit.gltf["materials"][0])
            self.assertEqual(edit.gltf.get("nodes") or [], [])


class TestFbxHandoff(unittest.TestCase):
    """The FBX's records in a converted GLB, and the names its handoff went
    by before ``HandoffRecord`` declared the block for every carrier (what the
    block says is ``test_handoff_record``'s)."""

    def test_the_retired_handoff_names_still_answer(self):
        """``RENDERING_POLICY``, ``rendering_policy``, ``HANDOFF_INSTRUCTIONS``,
        ``HANDOFF_ASSEMBLY``, ``FBX_HANDOFF_VERSION``, ``FBX_HANDOFF_CHANNEL``,
        ``FBX_HANDOFF_CHANNELS``, ``FBX_HANDOFF_INSTRUCTIONS``,
        ``build_fbx_handoff``, ``LIGHTING_ASSEMBLY`` and ``RENDERING_CARRIERS``
        (retired 2026-10-20 for ``HandoffRecord``, removed in 0.18.0) warn, and
        answer with what ``HandoffRecord`` states now -- the policy carrying
        no prose for a carrier to lay over it."""
        from pythontk import SceneRecords

        retired = {
            "RENDERING_POLICY": HandoffRecord.RENDERING_POLICY,
            "HANDOFF_INSTRUCTIONS": HandoffRecord.instructions("glb"),
            "HANDOFF_ASSEMBLY": HandoffRecord.CARRIERS["glb"].own,
            "FBX_HANDOFF_VERSION": HandoffRecord.VERSION,
            "FBX_HANDOFF_CHANNEL": HandoffRecord.KEY,
            "FBX_HANDOFF_INSTRUCTIONS": HandoffRecord.instructions("fbx"),
            "FBX_HANDOFF_CHANNELS": {
                spec.key: spec.description
                for spec in SceneRecords.deliverable()
                if spec is not SceneRecords.HANDOFF
            },
            "RENDERING_CARRIERS": {"glb": {}, "fbx": {}},
        }
        for name, now in retired.items():
            with self.subTest(name=name):
                with self.assertWarns(DeprecationWarning):
                    value = getattr(MeshConvert, name)
                self.assertEqual(value, now)
        with self.assertWarns(DeprecationWarning):
            rows = MeshConvert.LIGHTING_ASSEMBLY
        self.assertEqual(list(rows), HandoffRecord.lighting("glb"))

        chosen = {"lightmappedMaterials": {"envMapIntensity": 0.5}}
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(
                MeshConvert.rendering_policy(chosen, carrier="fbx"),
                HandoffRecord.rendering_policy(chosen),
            )
        with self.assertWarns(DeprecationWarning), self.assertRaises(ValueError):
            MeshConvert.rendering_policy(carrier="usdz")

        channels = {
            "lightmaps": "{}",
            "emissiveGroup_Neon": 0.5,
            HandoffRecord.KEY: "{}",
        }
        with self.assertWarns(DeprecationWarning):
            block = MeshConvert.build_fbx_handoff(channels, {"application": "maya"})
        self.assertEqual(
            block, HandoffRecord.block("fbx", channels, {"application": "maya"})
        )
        for empty in ([], [HandoffRecord.KEY], {"emissiveGroup_Neon": 1.0}):
            with self.subTest(channels=empty), self.assertWarns(DeprecationWarning):
                self.assertEqual(
                    MeshConvert.build_fbx_handoff(empty),
                    {},
                    "a carrier holding no record has nothing to describe",
                )

    def test_the_retired_record_names_still_answer(self):
        """Every constant that restated a record's declaration (retired
        2026-10-20, removed in 0.18.0) warns, and serves what ``SceneRecords``
        declares -- the block a record rides a GLB as taking the record's own
        key; the take list and the folder hints, which no record carries any
        more, their old values."""
        records = SceneRecords
        retired = {
            "LIGHTMAP_METADATA_KEY": records.LIGHTMAPS.key,
            "LIGHTMAP_METADATA_VERSION": records.LIGHTMAPS.version,
            "LIGHTMAP_WEB_KEY": records.LIGHTMAPS.key,
            "SHOT_METADATA_KEY": records.SHOTS.key,
            "ANIMATION_WEB_KEY": records.SHOTS.key,
            "SHADOW_METADATA_KEY": records.SHADOWS.key,
            "SHADOW_METADATA_VERSION": records.SHADOWS.version,
            "SHADOW_WEB_KEY": records.SHADOWS.key,
            "ARTICULATION_KEY": records.ARTICULATION.key,
            "ARTICULATION_VERSION": records.ARTICULATION.version,
            "ARTICULATION_WEB_KEY": records.ARTICULATION.key,
            "VISIBILITY_TRACKS_KEY": records.VISIBILITY.key,
            "VISIBILITY_TRACKS_VERSION": records.VISIBILITY.version,
            "FBX_TAKES_KEY": "fbx_takes",
            "LOCATE_HINT_KEY": "dir",
            "LOCATE_HINT_KEYS": ("dir", "dirs"),
        }
        for name, now in retired.items():
            with self.subTest(name=name):
                with self.assertWarns(DeprecationWarning):
                    value = getattr(MeshConvert, name)
                self.assertEqual(value, now)
        # The ones that read a record's GLB block outlived the block machinery
        # (2026-11-08), held until 0.22.0 (their window closes 2026-11-23): a
        # GLB carries each record as it stands, its version the record's and no
        # field bound by node index (``TestRetiredUntil022``).
        import pythontk

        for name in (
            "ANIMATION_WEB_VERSION",
            "SHADOW_WEB_NODE_FIELDS",
            "ARTICULATION_WEB_NODE_FIELDS",
        ):
            with self.subTest(name=name):
                self.assertIn(f"MeshConvert.{name}", pythontk.HELD_UNTIL_0_22)
                self.assertNotIn(f"MeshConvert.{name}", pythontk.MOVED_NAMES)

    def test_every_record_the_fbx_carried_leaves_a_converted_glb(self):
        """A GLB must not carry the FBX's account of itself.

        FBX2glTF transcribes every user property into node extras (a native
        glTF export writes them at the top of the node's extras), so the FBX's
        records arrived in the GLB a second time: the carrier's channels --
        its handoff describing the FBX, not this file -- and each baked
        object's ``lightmapInfo`` marker. The build states each in glTF's own
        terms, so the copies leave -- and the converter's own notes with them
        (``fromFBX``: nothing reads them once the conversion's passes ran);
        another program's extras stay, and a carrier holding no content goes
        -- every index renumbered.
        """
        uv_set = {"type": "eFbxString", "value": "map1"}
        gltf = {
            "scenes": [{"nodes": [0, 1, 2, 3, 4]}],
            "nodes": [
                {
                    "name": "box",
                    "extras": {
                        "fromFBX": {
                            "userProperties": {
                                "currentUVSet": dict(uv_set),
                                "lightmapInfo": {"type": "eFbxString", "value": "{}"},
                            }
                        }
                    },
                },
                {
                    "name": "data_export",
                    "extras": {
                        "fromFBX": {
                            "userProperties": {
                                "lightmaps": {
                                    "type": "eFbxString",
                                    "value": "{}",
                                },
                                HandoffRecord.KEY: {
                                    "type": "eFbxString",
                                    "value": "{}",
                                },
                            }
                        }
                    },
                },
                {
                    "name": "NS:data_export",
                    "mesh": 0,
                    "extras": {"lightmaps": "{}", "note": "another tool's"},
                },
                {"name": "data_export1", "extras": {"isNull": True}},
                {"name": "plain"},
            ],
        }
        # Every node that changed, the carriers it removed included.
        self.assertEqual(MeshConvert.strip_fbx_records(gltf), 4)
        self.assertEqual(
            [node["name"] for node in gltf["nodes"]], ["box", "NS:data_export", "plain"]
        )
        self.assertEqual(gltf["scenes"][0]["nodes"], [0, 1, 2])
        self.assertNotIn("extras", gltf["nodes"][0], "the converter's note too")
        self.assertEqual(
            gltf["nodes"][1]["extras"],
            {"note": "another tool's"},
            "a declared record leaves a carrier with content; another program's stays",
        )
        self.assertEqual(MeshConvert.strip_fbx_records(gltf), 0, "idempotent")

    def test_the_converters_notes_leave_a_finished_glb(self):
        """FBX2glTF's own notes -- a node's host attributes (Maya's current UV
        set, 3ds Max's handle) and converter flags, a material's (its shading
        model, ``isTruePBR``) -- leave with the records: nothing reads them
        once the conversion's passes ran. An attribute the artist authored is content
        and stays, as do another program's extras; an extras object left
        empty goes. Added: 2026-10-20"""
        uv_set = {"type": "eFbxString", "value": "map1"}
        gltf = {
            "nodes": [
                {
                    "name": "box",
                    "mesh": 0,
                    "extras": {
                        "fromFBX": {
                            "userProperties": {
                                "currentUVSet": dict(uv_set),
                                "MaxHandle": {"type": "eFbxInt", "value": 7},
                            },
                            "isNull": False,
                        }
                    },
                },
                {
                    "name": "lamp",
                    "mesh": 0,
                    "extras": {
                        "fromFBX": {
                            "userProperties": {
                                "currentUVSet": dict(uv_set),
                                "interactable": {"type": "eFbxBool", "value": True},
                            }
                        },
                        "note": "theirs",
                    },
                },
                {"name": "plain", "mesh": 0},
            ],
            "materials": [
                {"name": "m", "extras": {"fromFBX": {"shadingModel": "lambert"}}},
                {
                    "name": "n",
                    "extras": {
                        "fromFBX": {"shadingModel": "phong", "isTruePBR": False},
                        "kept": 1,
                    },
                },
            ],
            "animations": [
                {"name": "S", "extras": {"theirs": True}},
                {"name": "T"},
            ],
        }
        # Every object that changed: two nodes, two materials.
        self.assertEqual(MeshConvert.strip_fbx_records(gltf), 4)
        self.assertNotIn("extras", gltf["nodes"][0])
        self.assertEqual(
            gltf["nodes"][1]["extras"],
            {
                "fromFBX": {
                    "userProperties": {
                        "interactable": {"type": "eFbxBool", "value": True}
                    }
                },
                "note": "theirs",
            },
        )
        self.assertNotIn("extras", gltf["materials"][0])
        self.assertEqual(gltf["materials"][1]["extras"], {"kept": 1})
        self.assertEqual(gltf["animations"][0]["extras"], {"theirs": True})
        self.assertEqual(MeshConvert.strip_fbx_records(gltf), 0, "idempotent")

    def test_the_retired_strip_drops_the_handoff_alone(self):
        """``strip_fbx_handoff`` (retired 2026-10-20): the one record
        ``strip_fbx_records`` now drops with every other the FBX carried."""
        gltf = {
            "nodes": [
                {
                    "name": "box",
                    "extras": {
                        "fromFBX": {
                            "userProperties": {
                                "lightmapInfo": {"type": "eFbxString", "value": "{}"},
                            }
                        }
                    },
                },
                {
                    "name": "data_export",
                    "extras": {
                        "fromFBX": {
                            "userProperties": {
                                "lightmaps": {
                                    "type": "eFbxString",
                                    "value": "{}",
                                },
                                HandoffRecord.KEY: {
                                    "type": "eFbxString",
                                    "value": "{}",
                                },
                            }
                        }
                    },
                },
            ]
        }
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(MeshConvert.strip_fbx_handoff(gltf), 1)
        carrier = gltf["nodes"][1]["extras"]["fromFBX"]["userProperties"]
        self.assertNotIn(HandoffRecord.KEY, carrier)
        self.assertIn("lightmaps", carrier)
        self.assertIn(
            "lightmapInfo", gltf["nodes"][0]["extras"]["fromFBX"]["userProperties"]
        )
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(MeshConvert.strip_fbx_handoff(gltf), 0, "idempotent")


class TestCurrentRecord(unittest.TestCase):
    """``_current_record``: the one gate every pass that applies a record
    reads it through -- the lightmaps, shadows, articulation and render
    effects passes. A reader reads a record in the version and shape it
    declares alone (``SceneRecords.problems``): one of any other version, an
    older one too, or one its shape refuses, is refused and said, never read
    through a branch for its shape; the cure is a re-export.
    Added: 2026-10-20"""

    GATED = (
        SceneRecords.LIGHTMAPS,
        SceneRecords.SHADOWS,
        SceneRecords.ARTICULATION,
        SceneRecords.VISIBILITY,
    )
    LOGGER = "pythontk.file_utils.mesh_convert"

    @staticmethod
    def _gltf(key, payload):
        return {
            "nodes": [{"name": "data_export", "extras": {key: json.dumps(payload)}}]
        }

    @staticmethod
    def _version_alone():
        """The version gate with no record's shape in play: each shape is
        held to its producers by its own tests."""
        return patch.object(SceneRecords, "shape", return_value=None)

    def test_the_declared_version_is_read(self):
        with self._version_alone():
            for spec in self.GATED:
                with self.subTest(record=spec.key):
                    payload = {"version": spec.version, "entries": []}
                    self.assertEqual(
                        MeshConvert._current_record(
                            self._gltf(spec.key, payload), spec
                        ),
                        payload,
                    )

    def test_every_other_version_is_refused_and_said(self):
        with self._version_alone():
            for spec in self.GATED:
                for version in (spec.version - 1, spec.version + 1, None):
                    with self.subTest(record=spec.key, version=version):
                        payload = {"entries": []}
                        if version is not None:
                            payload["version"] = version
                        with self.assertLogs(self.LOGGER, level="WARNING") as logs:
                            self.assertIsNone(
                                MeshConvert._current_record(
                                    self._gltf(spec.key, payload), spec
                                )
                            )
                        self.assertIn(
                            f"{spec.key} record not applied -- {spec.key}: "
                            f"v{version} is not the v{spec.version} a reader "
                            "reads; re-export the file.",
                            logs.output[0],
                        )

    def test_a_record_its_shape_refuses_is_refused_and_said(self):
        """In its version, a record its declared shape refuses is no record a
        reader reads either: the first problems are said, and how many more."""
        spec = SceneRecords.VISIBILITY
        current = {"version": spec.version, "tracks": [{"node": "a", "channels": ["opacity"]}]}
        self.assertEqual(
            MeshConvert._current_record(self._gltf(spec.key, current), spec),
            current,
        )
        nameless = dict(current, tracks=[{"channels": ["opacity"]}] * 5)
        with self.assertLogs(self.LOGGER, level="WARNING") as logs:
            self.assertIsNone(
                MeshConvert._current_record(self._gltf(spec.key, nameless), spec)
            )
        said = logs.output[0]
        self.assertIn(f"{spec.key} record not applied -- {spec.key}.tracks[0]", said)
        self.assertIn(f"{spec.key}.tracks[2]", said)
        self.assertNotIn(f"{spec.key}.tracks[3]", said)
        self.assertIn("(+2 more); re-export the file.", said)

    def test_a_file_carrying_none_has_none_and_says_nothing(self):
        with self.assertNoLogs(self.LOGGER, level="WARNING"):
            for spec in self.GATED:
                self.assertIsNone(MeshConvert._current_record({"nodes": []}, spec))


class TestVerifyGlb(unittest.TestCase):
    """The reader a RECIPIENT runs against a delivered GLB.

    What the file says of itself (``extras.handoff``), whether it is valid
    glTF, and the one ORM finding that is destructive -- checkable from the
    file alone, never raising on the shape it reports on.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_verify_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _delivered(self, name="scene.glb"):
        """A GLB carrying its handoff: one material."""
        path = os.path.join(self.tmp, name)
        _write_glb_file(
            path, {"asset": {"version": "2.0"}, "materials": [{"name": "m"}]}
        )
        with MeshConvert.open_glb(path) as edit:
            MeshConvert.embed_glb_handoff(
                edit,
                HandoffRecord.block(
                    "glb", [], {"application": "maya", "version": "2025"}
                ),
            )
        return path

    def _handoff(self, path, **changes):
        """Edit *path*'s handoff in place (``None`` removes a key)."""
        with MeshConvert.open_glb(path) as edit:
            block = edit.gltf["extras"][HandoffRecord.KEY]
            for key, value in changes.items():
                if value is None:
                    block.pop(key, None)
                else:
                    block[key] = value
            edit.dirty = True

    def test_a_clean_deliverable_verifies(self):
        """The pass case has to be reachable, or every report reads as noise."""
        report = MeshConvert.verify_glb(self._delivered())

        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["problems"], [])
        self.assertEqual(
            report["handoff"],
            {
                "version": HandoffRecord.VERSION,
                "source": {"application": "maya", "version": "2025"},
            },
        )
        self.assertIn("pythontk", report["generator"])
        for retired in ("envelope", "textures", "sections"):
            self.assertNotIn(retired, report)

    def test_the_report_names_what_a_reader_must_SUPPORT_to_open_the_file(self):
        """`extensionsRequired` is not advice -- the spec says a reader that
        does not implement one of these must refuse the file. The deliverable
        most likely to reach a third party is precisely the one with a hard
        prerequisite (a web-delivery GLB requires `EXT_texture_webp`), and the
        report a recipient runs did not mention it."""
        path = self._delivered()
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["extensionsUsed"] = ["EXT_texture_webp"]
            edit.gltf["extensionsRequired"] = ["EXT_texture_webp"]
            edit.dirty = True
        report = MeshConvert.verify_glb(path)
        self.assertEqual(report["extensions"]["required"], ["EXT_texture_webp"])
        self.assertTrue(
            any("EXT_texture_webp" in n for n in report["notes"]),
            f"the prerequisite must be stated: {report['notes']}",
        )
        self.assertTrue(report["ok"], "a correct requirement is not a defect")

    def test_a_requirement_the_file_never_declares_is_invalid_gltf(self):
        """glTF 2.0 defines `extensionsRequired` as a SUBSET of
        `extensionsUsed`. A file demanding a capability it does not declare is
        rejected by stock validators, so this is a failure, not a note."""
        path = self._delivered()
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["extensionsRequired"] = ["EXT_texture_webp"]
            edit.gltf.pop("extensionsUsed", None)
            edit.dirty = True
        report = MeshConvert.verify_glb(path)
        self.assertFalse(report["ok"])
        self.assertTrue(any("invalid glTF" in p for p in report["problems"]))

    def test_a_handoff_of_another_schema_is_a_problem(self):
        """Another schema is the case worth saying out loud -- an older one as
        a newer one: this reader would skip whatever the other says and report
        a clean bill on a file it only partly understood. Said as the
        declaration says it (``SceneRecords.problems``)."""
        current = HandoffRecord.VERSION
        for version in (current - 1, current + 1):
            with self.subTest(version=version):
                path = self._delivered(f"v{version}.glb")
                self._handoff(path, version=version)
                report = MeshConvert.verify_glb(path)
                self.assertFalse(report["ok"])
                self.assertEqual(report["handoff"]["version"], version)
                self.assertIn(
                    f"extras.{HandoffRecord.KEY}: v{version} is not the "
                    f"v{current} a reader reads",
                    report["problems"],
                )

    def test_a_handoff_missing_a_fundamental_is_a_problem(self):
        """What a reader must read sits at the top: a block without it does
        not describe the file."""
        path = self._delivered()
        self._handoff(path, rendering=None)
        report = MeshConvert.verify_glb(path)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("'rendering'" in p for p in report["problems"]), report["problems"]
        )

    def test_every_block_is_held_to_its_declaration(self):
        """Each block of ours the file carries is held to its declaration as
        the handoff is (``SceneRecords.problems``, said at ``extras.<key>``): one of another
        version, or one its declared shape refuses, is a problem -- a reader
        would otherwise skip it and call the file clean. Added: 2026-10-20"""
        lightmaps, articulation = SceneRecords.LIGHTMAPS, SceneRecords.ARTICULATION
        stale = lightmaps.version + 1
        path = self._delivered()
        with MeshConvert.open_glb(path) as edit:
            extras = edit.gltf["extras"]
            extras[lightmaps.key] = _lightmaps_record("m", version=stale)
            extras[articulation.key] = {
                "version": articulation.version,
                "rigs": "none",
            }
            edit.dirty = True
        report = MeshConvert.verify_glb(path)
        self.assertFalse(report["ok"])
        self.assertIn(
            f"extras.{lightmaps.key}: v{stale} is not the "
            f"v{lightmaps.version} a reader reads",
            report["problems"],
        )
        where = f"extras.{articulation.key}.rigs"
        self.assertTrue(
            any(p.startswith(where) for p in report["problems"]), report["problems"]
        )
        # In its version and shape, a block is no problem.
        with MeshConvert.open_glb(path) as edit:
            del edit.gltf["extras"][lightmaps.key]
            edit.gltf["extras"][articulation.key]["rigs"] = []
            edit.dirty = True
        self.assertEqual(MeshConvert.verify_glb(path)["problems"], [])

    def test_only_a_destructive_orm_is_reported(self):
        """Every ORM is decoded, and only metallic everywhere -- black under a
        lightmap -- is said: a channel layout no reader can vouch for is
        neither a problem nor a note, so a clean report stays readable."""
        from io import BytesIO
        from PIL import Image

        blobs = []
        for color in ((40, 190, 90), (255, 255, 255)):
            buf = BytesIO()
            Image.new("RGB", (4, 4), color).save(buf, format="PNG")
            blobs.append(buf.getvalue())
        path = os.path.join(self.tmp, "orm.glb")
        with open(path, "wb") as fh:
            fh.write(
                TestCheckGlbMaterials._build_glb(
                    materials=[
                        {
                            "name": name,
                            "pbrMetallicRoughness": {
                                "metallicRoughnessTexture": {"index": index}
                            },
                        }
                        for index, name in enumerate(("packed", "white"))
                    ],
                    images=[
                        {"bufferView": i, "mimeType": "image/png", "name": f"img{i}"}
                        for i in range(len(blobs))
                    ],
                    textures=[{"source": i} for i in range(len(blobs))],
                    image_blobs=blobs,
                )
            )
        with MeshConvert.open_glb(path) as edit:
            MeshConvert.embed_glb_handoff(edit, HandoffRecord.block("glb", []))

        report = MeshConvert.verify_glb(path)

        self.assertFalse(report["ok"])
        self.assertEqual(list(report["orm"]), ["white"])
        self.assertEqual(
            report["orm"]["white"]["finding"], MeshConvert.ORM_FINDING_METALLIC_FULL
        )
        self.assertTrue(any("white" in p for p in report["problems"]))
        self.assertFalse(any("packed" in p for p in report["problems"]))
        self.assertEqual(
            [n for n in report["notes"] if "ORM" in n or "packed" in n], []
        )

    def test_a_handoff_that_is_not_an_object_is_reported_not_raised(self):
        """A report-only method must not raise on the shape it is reporting on,
        and a block it could not parse must never come back ``ok``: a
        hand-edited (or future-schema) file is exactly what a recipient runs
        this to find out about."""
        path = self._delivered()
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["extras"][HandoffRecord.KEY] = ["not", "an", "object"]
            edit.dirty = True
        report = MeshConvert.verify_glb(path)
        self.assertFalse(report["ok"])
        self.assertIsNone(report["handoff"])

    def test_a_handoff_whose_extras_is_not_an_object_is_reported_not_raised(self):
        """Same rule one level down: the block's ``extras`` was dereferenced
        with ``.get`` -- the shape the retired envelope's ``sections`` and
        ``textures`` blocks once raised on."""
        path = self._delivered()
        self._handoff(path, extras=["source"])
        report = MeshConvert.verify_glb(path)
        self.assertEqual(report["handoff"]["version"], HandoffRecord.VERSION)
        self.assertIsNone(report["handoff"]["source"])
        self.assertIn(
            f"extras.{HandoffRecord.KEY}.extras: expected a mapping for nested schema",
            report["problems"],
        )

    def test_a_file_carrying_only_the_retired_envelope_declares_nothing(self):
        """Its ``extras.scene_sidecar`` envelope is read by no reader: the file
        states no handoff, as a foreign GLB does -- re-exported, it states
        one."""
        path = os.path.join(self.tmp, "older.glb")
        _write_glb_file(
            path,
            {"asset": {"version": "2.0"}, "extras": {"scene_sidecar": {"version": 2}}},
        )
        report = MeshConvert.verify_glb(path)
        self.assertFalse(report["ok"])
        self.assertIsNone(report["handoff"])
        self.assertIn(
            "no extras.handoff: nothing here declares itself", report["problems"]
        )
        self.assertFalse(any("scene_sidecar" in n for n in report["notes"]))

    def test_a_foreign_glb_is_reported_not_raised(self):
        """Pointing this at someone else's asset is a legitimate thing to do."""
        path = os.path.join(self.tmp, "foreign.glb")
        _write_glb_file(path, {"asset": {"version": "2.0"}})

        report = MeshConvert.verify_glb(path)

        self.assertFalse(report["ok"])
        self.assertIsNone(report["handoff"])
        self.assertTrue(any("no extras.handoff" in p for p in report["problems"]))


class TestSuspectOrmMaterials(unittest.TestCase):
    """The detector for the ORM the sidecar did NOT repair.

    FBX2glTF white-fills a grayscale ("L"-mode) PBR source, and glTF reads
    metallic from the ORM's blue channel -- so the converter's packing renders
    metallic=1: no diffuse response, and pure black under a lightmap (which
    contributes to diffuse alone). `apply_glb_materials` repacks the materials
    the record states; these tests cover the ones it does not reach, which
    shipped that failure with nothing said.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_orm_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    @staticmethod
    def _png(color, size=(4, 4)):
        from io import BytesIO
        from PIL import Image

        buf = BytesIO()
        Image.new("RGB", size, color).save(buf, format="PNG")
        return buf.getvalue()

    def _glb(self, materials, blobs, name="scene.glb"):
        """A GLB binding one image per texture, in order."""
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(
                TestCheckGlbMaterials._build_glb(
                    materials=materials,
                    images=[
                        {"bufferView": i, "mimeType": "image/png", "name": f"img{i}"}
                        for i in range(len(blobs))
                    ],
                    textures=[{"source": i} for i in range(len(blobs))],
                    image_blobs=blobs,
                )
            )
        return path

    @staticmethod
    def _orm_material(name, texture, **pbr):
        mat = {"name": name, "pbrMetallicRoughness": dict(pbr)}
        mat["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": texture}
        return mat

    def test_flags_only_the_material_whose_metallic_channel_is_full(self):
        """White ORM -> flagged; a real packing -> not. The rest is noise."""
        path = self._glb(
            materials=[
                self._orm_material("white", 0),
                self._orm_material("packed", 1),
                {"name": "no_orm"},
            ],
            blobs=[self._png((255, 255, 255)), self._png((255, 180, 20))],
        )

        found = MeshConvert.suspect_orm_materials(path)

        self.assertEqual(list(found), ["white"])
        self.assertEqual(
            found["white"]["finding"], MeshConvert.ORM_FINDING_METALLIC_FULL
        )
        self.assertEqual(found["white"]["image"], "img0")

    def test_a_constant_metal_over_a_white_filled_channel_is_not_flagged(self):
        """A material with a roughness map and a constant metal packs its blue
        channel white (an absent channel's fill: its factor carries it), so a
        white B is metal everywhere only where the factor is 1 or more.
        Added: 2026-11-05"""
        path = self._glb(
            materials=[self._orm_material("cast", 0, metallicFactor=0.4)],
            blobs=[self._png((255, 90, 255))],
        )

        self.assertEqual(MeshConvert.suspect_orm_materials(path), {})

    def test_verify_glb_judges_no_pack_this_build_wrote(self):
        """``verify_glb`` takes what the build knows: the materials whose
        packing it wrote are no finding of its, whatever their blue channel
        (a fully metal surface packs it white). Added: 2026-11-05"""
        path = self._glb(
            materials=[self._orm_material("chrome", 0), self._orm_material("raw", 0)],
            blobs=[self._png((255, 40, 255))],
        )
        with MeshConvert.open_glb(path) as edit:
            MeshConvert.embed_glb_handoff(
                edit, HandoffRecord.block("glb", [], {"application": "maya"})
            )

        report = MeshConvert.verify_glb(path, described={"chrome"})

        self.assertEqual(list(report.get("orm") or {}), ["raw"])

    def test_a_zero_metallic_factor_cancels_the_texture(self):
        """metallicFactor 0 multiplies the white channel to nothing.

        The material renders as a non-metal, so flagging it would send an
        artist after a file that is behaving exactly as authored.
        """
        path = self._glb(
            materials=[self._orm_material("inert", 0, metallicFactor=0)],
            blobs=[self._png((255, 255, 255))],
        )

        self.assertEqual(MeshConvert.suspect_orm_materials(path), {})

    def test_excluded_names_are_not_probed(self):
        """A material the repair covers is the packer's business, not this.

        Its ORM comes from the authoring maps rather than from the converter,
        and `pack_orm_texture` has already reported per map. Excluding by name
        is also what keeps a fully-covered export from decoding anything.
        """
        path = self._glb(
            materials=[self._orm_material("covered", 0)],
            blobs=[self._png((255, 255, 255))],
        )

        self.assertEqual(
            MeshConvert.suspect_orm_materials(path, described={"covered"}), {}
        )

    def test_a_described_name_covers_that_material_alone(self):
        """A material is described by its own name: ``covered~lmap`` is not
        ``covered``, and exempting it would hide it from every finding. (The
        lightmap pass once cloned a material per instance under ``~lm<N>``
        names, which this exempted; a GLB carries no clone now.)"""
        path = self._glb(
            materials=[self._orm_material("covered~lmap", 0)],
            blobs=[self._png((40, 190, 90))],
        )

        found = MeshConvert.suspect_orm_materials(path, described={"covered"})

        self.assertEqual(
            found["covered~lmap"]["finding"], MeshConvert.ORM_FINDING_UNVALIDATED
        )

    def test_an_undescribed_binding_is_reported_even_when_it_looks_ordinary(self):
        """The case a whiteness test is blind to.

        A mask map packed for another engine (Unity's is R=Metallic,
        G=Occlusion, B=Detail) that reaches the GLB unrepaired is read channel
        for channel as ORM and misinterpreted -- while looking like perfectly
        ordinary image data. Nothing about its pixels says so; what says so is
        that the envelope never described the binding.
        """
        path = self._glb(
            materials=[self._orm_material("foreign", 0)],
            blobs=[self._png((40, 190, 90))],  # nothing uniform, nothing white
        )

        found = MeshConvert.suspect_orm_materials(path, described=set())

        self.assertEqual(
            found["foreign"]["finding"], MeshConvert.ORM_FINDING_UNVALIDATED
        )

    def test_nothing_is_unvalidated_when_the_caller_says_nothing(self):
        """`described=None` must not report every material in the file.

        Only the caller knows what was described; absent that, "unvalidated"
        is not a claim this can make, and making it anyway would fire on every
        material of every asset from another producer.
        """
        path = self._glb(
            materials=[self._orm_material("foreign", 0)],
            blobs=[self._png((40, 190, 90))],
        )

        self.assertEqual(MeshConvert.suspect_orm_materials(path), {})

    def test_glb_orm_layout_matches_the_registry(self):
        """The spec constant and the authored taxonomy must not drift apart.

        `GLTF_ORM_CHANNELS` is held as glTF's own constant rather than looked
        up from `MapRegistry`, so an edit to the registry cannot silently
        change what the check tests. This is the tie that makes the two
        visible to each other -- if the registry's ORM layout ever stops
        matching the spec, that is a finding, not a silent divergence.
        """
        from pythontk import MapRegistry

        self.assertEqual(
            MapRegistry().get("ORM").channels, MeshConvert.GLTF_ORM_CHANNELS
        )
        # And the channel the harm test reads really is the metallic one.
        self.assertEqual(
            MeshConvert.GLTF_ORM_CHANNELS[MeshConvert._ORM_HARMFUL_CHANNEL], "Metallic"
        )

    def test_the_build_warns_of_what_its_repairs_left_behind(self):
        """The warning has to fire where the deliverable is actually built.

        A repair that covers only some materials is the measured production
        case -- the covered room looks right, the omitted material ships
        black -- so the build's check is told the names it repaired (the
        materials record's packed ORMs, ``fbx_to_glb``) and names the rest.
        """
        path = self._glb(
            materials=[
                self._orm_material("repaired", 0),
                self._orm_material("omitted", 0),
            ],
            blobs=[self._png((255, 255, 255))],
        )
        with (
            self.assertLogs("pythontk.file_utils.mesh_convert", level="WARNING") as caught,
            MeshConvert.open_glb(path) as edit,
        ):
            MeshConvert._sidecar._warn_destructive_orms(edit, described=["repaired"])

        flagged = [m for m in caught.output if "Metallic=1 everywhere" in m]
        self.assertEqual(len(flagged), 1)
        self.assertIn("omitted", flagged[0])
        # The excluded name must not be listed -- deliberately not "covered",
        # which is a word the message itself uses.
        self.assertNotIn("repaired", flagged[0])

    def test_the_export_time_warning_stays_off_the_unvalidated_finding(self):
        """Coverage information is not a defect, and must not warn at export.

        An artist cannot act on "nothing described this binding" mid-export,
        and a warning nobody can act on is how the actionable ones stop being
        read.
        """
        path = self._glb(
            materials=[self._orm_material("undescribed", 0)],
            blobs=[self._png((40, 190, 90))],
        )
        with (
            self.assertLogs("pythontk.file_utils.mesh_convert", level="WARNING") as caught,
            MeshConvert.open_glb(path) as edit,
        ):
            logging.getLogger("pythontk.file_utils.mesh_convert").warning(
                "sentinel so assertLogs always has a record"
            )
            MeshConvert._sidecar._warn_destructive_orms(edit, described=["other"])

        self.assertEqual(
            [m for m in caught.output if "Metallic=1" in m or "unvalidated" in m], []
        )
        # Not vacuous: the finding really is there, and only the LOG is quiet.
        with MeshConvert.open_glb(path) as edit:
            found = MeshConvert.suspect_orm_materials(edit, described=set())
        self.assertEqual(
            found["undescribed"]["finding"], MeshConvert.ORM_FINDING_UNVALIDATED
        )


class TestAssetGeneratorStamp(unittest.TestCase):
    """`asset.generator` is the one provenance field glTF itself defines.

    Every viewer and inspector displays it, so it reaches a recipient who opens
    the deliverable in a tool that is not ours and reads nothing else we write.
    Stamped when the handoff is embedded, from the handoff's own source.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_generator_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _glb(self, gltf, name="scene.glb"):
        path = os.path.join(self.tmp, name)
        _write_glb_file(path, gltf)
        return path

    @staticmethod
    def _stamp(path, source=None):
        """Embed a handoff from *source*; the generator it leaves."""
        if source is None:
            source = {"application": "maya", "version": "2025"}
        with MeshConvert.open_glb(path) as edit:
            MeshConvert.embed_glb_handoff(edit, HandoffRecord.block("glb", [], source))
        with MeshConvert.open_glb(path) as edit:
            return edit.gltf["asset"]["generator"]

    def test_stamp_names_the_authoring_app_and_this_package(self):
        """Coarse on purpose: the app the handoff already names, and us.

        No host, no user, no paths -- a generator string travels to whoever
        gets the file.
        """
        from pythontk import __version__ as ptk_version

        path = self._glb({"asset": {"version": "2.0"}, "materials": [{"name": "m"}]})
        generator = self._stamp(path)

        self.assertIn("maya 2025", generator)
        self.assertIn(f"pythontk {ptk_version}", generator)
        self.assertNotIn(os.getcwd(), generator)

    def test_an_application_without_a_version_stamps_no_literal_None(self):
        """An f-string over a missing version writes "None" mid-string.

        `.strip()` cannot reach it there. Every in-repo producer supplies a
        version (`cmds.about(version=True)` / `bpy.app.version_string`), so this
        reaches only a caller of the PUBLIC `HandoffRecord.block(..., source)` -- which
        is exactly the caller who would then ship "someApp None + pythontk
        0.9.24" in the one provenance field glTF itself defines.
        """
        path = self._glb({"asset": {"version": "2.0"}, "materials": [{"name": "m"}]})
        generator = self._stamp(path, {"application": "someApp"})

        self.assertNotIn("None", generator)
        self.assertIn("someApp", generator)

    def test_a_source_naming_no_application_stamps_this_package_alone(self):
        """A handoff's source names the application it is stated by, or there
        is none (``HandoffSource``, 2026-10-20): a version with no application
        is no source, so the generator names this package alone -- never a
        bare number, nor "None"."""
        from pythontk import __version__ as ptk_version

        path = self._glb({"asset": {"version": "2.0"}, "materials": [{"name": "m"}]})
        generator = self._stamp(path, {"version": "2025"})

        self.assertEqual(generator, f"pythontk {ptk_version}")

    def test_the_converters_own_claim_is_kept(self):
        """FBX2glTF really did produce the geometry.

        Replacing that claim would lose the fact that matters most when a mesh
        arrives wrong, so ours is appended to it.
        """
        path = self._glb(
            {
                "asset": {"version": "2.0", "generator": "FBX2glTF"},
                "materials": [{"name": "m"}],
            }
        )
        self.assertTrue(self._stamp(path).startswith("FBX2glTF via "))

    def test_reapplying_stacks_nothing_when_there_was_no_converter_claim(self):
        """A GLB whose only generator claim is a previous stamp of ours.

        There is no separator to split on in that case, so the whole string
        comes back as the "prior" claim and the next pass appends ours to
        itself. Reachable from any producer that writes no generator of its
        own and re-embeds a handoff.
        """
        path = self._glb({"asset": {"version": "2.0"}, "materials": [{"name": "m"}]})
        self._stamp(path)
        generator = self._stamp(path)

        self.assertEqual(generator.count("pythontk"), 1, generator)
        self.assertNotIn(" via ", generator)

    def test_reapplying_refreshes_the_stamp_rather_than_stacking(self):
        """Passes compose: a GLB can be re-opened and its handoff re-embedded.

        A stamp that appended every time would grow the generator string one
        entry per pass, and a reader could not tell which was current.
        """
        path = self._glb(
            {
                "asset": {"version": "2.0", "generator": "FBX2glTF"},
                "materials": [{"name": "m"}],
            }
        )
        self._stamp(path)
        generator = self._stamp(path)

        self.assertEqual(generator.count(" via "), 1)


class TestGlbLightmapMedia(unittest.TestCase):
    """One layout, two containers: the GLB carries the ``lightmaps`` record
    as the FBX holds it, and every file it names as the bufferView named by
    that file (``SceneRecords.GLB_MEDIA``) -- no material bound, none cloned.

    The room: ``floor`` and a namespaced ``NS:prop`` share ``roomMat`` and
    are baked (the prop into a corner of an atlas); ``stool`` wears its own
    material and is not. Added: 2026-11-07
    """

    MEDIA = SceneRecords.GLB_MEDIA

    @classmethod
    def setUpClass(cls):
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cv2 unavailable; HDR encode untestable")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_lightmap_media_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _hdr(self, name, value=0.5, shape=(8, 8)):
        import cv2

        path = os.path.join(self.tmp, name)
        cv2.imwrite(path, np.full(shape + (3,), value, dtype=np.float32))
        return path

    def _record(self, objects=None, probe=None, lights=None):
        record = {
            "version": SceneRecords.LIGHTMAPS.version,
            "objects": [
                _baked(entry)
                for entry in objects
                or [
                    {"name": "floor", "map": "floor_LM.exr", "hierarchy": ["room", "floor"]},
                    {
                        "name": "prop",
                        "map": "atlas_LM.exr",
                        "scaleOffset": [0.5, 0.5, 0.5, 0.0],
                        "hierarchy": ["room", "NS:prop"],
                    },
                ]
            ],
        }
        if probe is not None or lights is not None:
            record["unit_scale"] = 0.01
        if probe is not None:
            record["probe"] = probe
        if lights is not None:
            record["lights"] = lights
        return record

    def _room(self, record):
        attrs = {"POSITION": 0, "TEXCOORD_0": 1, "TEXCOORD_1": 2}
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0, 4]}],
            "nodes": [
                {"name": "room", "children": [1, 2, 3]},
                {"name": "floor", "mesh": 0},
                {"name": "NS:prop", "mesh": 1},
                {"name": "stool", "mesh": 2},
                TestGlbLightmaps._carrier_node(record),
            ],
            "meshes": [
                {"primitives": [{"attributes": dict(attrs), "material": m}]}
                for m in (0, 0, 1)
            ],
            "materials": [{"name": "roomMat"}, {"name": "stoolMat"}],
        }
        return _write_glb_file(os.path.join(self.tmp, "room.glb"), gltf)

    def _built(self, record, **kwargs):
        self._hdr("floor_LM.exr", 0.5)
        self._hdr("atlas_LM.exr", 0.25)
        path = self._room(record)
        bound = MeshConvert.apply_glb_lightmaps(path, search_dirs=[self.tmp], **kwargs)
        return path, bound

    def _view_bytes(self, path, name):
        edit = MeshConvert._read_glb(path)
        index = self.MEDIA.view(edit.gltf, name)
        self.assertIsNotNone(index, f"no bufferView named {name}")
        view = edit.gltf["bufferViews"][index]
        start = int(view.get("byteOffset", 0))
        return view, bytes(edit.bin_data[start : start + int(view["byteLength"])])

    def test_the_glb_carries_the_record_as_the_fbx_holds_it(self):
        record = self._record()
        path, bound = self._built(record)
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(gltf["extras"][SceneRecords.LIGHTMAPS.key], record)
        # No material bound or cloned, no mesh copied: the viewer makes what
        # it needs at load from the record.
        self.assertEqual([m["name"] for m in gltf["materials"]], ["roomMat", "stoolMat"])
        self.assertEqual(len(gltf["meshes"]), 3)
        for material in gltf["materials"]:
            self.assertNotIn("occlusionTexture", material)
        self.assertEqual(sorted(r["object"] for r in bound), ["floor", "prop"])

    def test_each_map_rides_as_the_view_named_by_its_file(self):
        """Today's web encoding, unchanged: 8-bit sRGB whose decoded texels
        times ``extras.scale`` are the map (the golden constant: 0.5 -> 255
        at scale 0.5)."""
        from PIL import Image

        path, _bound = self._built(self._record())
        view, data = self._view_bytes(path, "floor_LM.exr")
        self.assertEqual(view["extras"], self.MEDIA.extras("image/png", 0.5))
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.getpixel((0, 0))[:3], (255, 255, 255))
        view, _data = self._view_bytes(path, "ATLAS_lm.EXR")  # joined without case
        self.assertAlmostEqual(view["extras"]["scale"], 0.25)
        self.assertFalse(MeshConvert._read_glb(path).gltf.get("images"))

    def test_a_map_only_objects_outside_the_file_wear_does_not_ride(self):
        self._hdr("ghost_LM.exr")
        record = self._record()
        record["objects"].append(_baked({"name": "ghost", "map": "ghost_LM.exr"}))
        path, _bound = self._built(record)
        gltf = MeshConvert._read_glb(path).gltf
        self.assertIsNone(self.MEDIA.view(gltf, "ghost_LM.exr"))
        self.assertIsNotNone(self.MEDIA.view(gltf, "floor_LM.exr"))

    def test_the_views_survive_bin_compaction(self):
        """A media view is held by its name, which no index references."""
        path, _bound = self._built(self._record())
        with MeshConvert.open_glb(path) as edit:
            MeshConvert._compact_bin(edit)
            edit.dirty = True
        gltf = MeshConvert._read_glb(path).gltf
        for name in ("floor_LM.exr", "atlas_LM.exr"):
            self.assertIsNotNone(self.MEDIA.view(gltf, name), name)

    def test_a_second_pass_adds_nothing(self):
        path, _bound = self._built(self._record())
        before = MeshConvert._read_glb(path).gltf["bufferViews"]
        MeshConvert.apply_glb_lightmaps(path, search_dirs=[self.tmp])
        self.assertEqual(MeshConvert._read_glb(path).gltf["bufferViews"], before)

    def test_the_transport_joins_by_file_name_and_never_guesses_a_tie(self):
        """``GlbMedia``: a record's name joins a view's by file name, no folder,
        lower-cased on both sides; two views that join one name hold neither;
        a scale is stated only on an 8-bit web encoding of an HDR map, and
        only a finite positive one. Added: 2026-11-07"""
        media = self.MEDIA
        gltf = {"bufferViews": [
            {"name": "Floor_LM.exr"},
            {"name": "maps/Atlas.EXR"},
            {"name": "atlas.exr"},
            {"byteLength": 4},
        ]}
        self.assertEqual(media.fold("C:\\bake\\Floor_LM.EXR"), "floor_lm.exr")
        self.assertEqual(media.view(gltf, "bake/FLOOR_lm.exr"), 0)
        self.assertEqual(media.views(gltf, "Atlas.exr"), [1, 2])
        self.assertIsNone(media.view(gltf, "Atlas.exr"), "a tie is never guessed")
        self.assertIsNone(media.view(gltf, "ghost.exr"))
        self.assertEqual(media.views({}, "floor_LM.exr"), [])
        for mime in media.hdr_web:
            self.assertEqual(media.extras(mime, 2), {"mimeType": mime, "scale": 2.0})
        self.assertEqual(media.extras(media.radiance), {"mimeType": media.radiance})
        with self.assertRaises(ValueError):
            media.extras(media.radiance, 1.0)
        for bad in (0, -1.0, float("nan"), float("inf")):
            with self.subTest(scale=bad), self.assertRaises(ValueError):
                media.extras("image/png", bad)

    def test_a_reader_holds_the_files_to_the_transport(self):
        """``media_findings`` -- the media gate ``verify_glb`` runs: a clean
        build passes; a map's view stating another type or no scale, a probe
        not Radiance, a probe a lit file does not carry and a name two views
        hold are problems; a map an object here reads that rides nothing is a
        note (that object renders unlit). Added: 2026-11-07"""
        self._hdr("room_Probe.hdr", 0.75, shape=(16, 32))
        probe = {"map": "room_Probe.hdr", "position": [0.0, 150.0, 0.0], "box": None}
        path, _bound = self._built(self._record(probe=probe))
        where = f"extras.{SceneRecords.LIGHTMAPS.key}"

        def findings(change=None):
            with MeshConvert.open_glb(path) as edit:
                gltf = copy.deepcopy(edit.gltf)
            if change:
                change(gltf)
            return MeshConvert._lightmaps.media_findings(gltf)

        self.assertEqual(findings(), ([], []))
        with MeshConvert.open_glb(path) as edit:  # a file that declares itself
            MeshConvert.embed_glb_handoff(
                edit, HandoffRecord.block("glb", [], {"application": "maya", "version": "2025"})
            )
        verified = MeshConvert.verify_glb(path)["problems"]
        self.assertFalse([p for p in verified if where in p], verified)

        def view(gltf, name):
            return gltf["bufferViews"][self.MEDIA.view(gltf, name)]

        def retype(gltf):
            view(gltf, "floor_LM.exr")["extras"] = {"mimeType": "image/jpeg", "scale": 0.5}

        def unscale(gltf):
            view(gltf, "atlas_LM.exr")["extras"].pop("scale")

        def scaled_probe(gltf):
            view(gltf, "room_Probe.hdr")["extras"]["scale"] = 1.0

        def drop_probe(gltf):
            view(gltf, "room_Probe.hdr").pop("name")

        def tie(gltf):
            gltf["bufferViews"].append(dict(view(gltf, "floor_LM.exr"), name="FLOOR_lm.EXR"))

        def drop_map(gltf):
            view(gltf, "atlas_LM.exr").pop("name")

        for change, wanted in (
            (retype, "the view of floor_LM.exr states 'image/jpeg'"),
            (unscale, "the view of atlas_LM.exr states no positive extras.scale"),
            (scaled_probe, "the view of room_Probe.hdr is not Radiance"),
            (drop_probe, "names room_Probe.hdr, which rides no bufferView"),
            (tie, "2 bufferViews hold floor_LM.exr"),
        ):
            with self.subTest(change=change.__name__):
                problems, _notes = findings(change)
                self.assertTrue(any(wanted in p for p in problems), problems)
        problems, notes = findings(drop_map)
        self.assertEqual(problems, [])
        self.assertEqual(len(notes), 1)
        self.assertIn("atlas_LM.exr", notes[0])
        # verify_glb gates on them: the file as written, broken the same way.
        with MeshConvert.open_glb(path) as edit:
            retype(edit.gltf)
            edit.dirty = True
        verified = MeshConvert.verify_glb(path)["problems"]
        self.assertTrue(any("states 'image/jpeg'" in p for p in verified), verified)

    def test_a_second_pass_reads_the_maps_the_file_holds(self):
        """The bake's folders gone (the delivery moved), a second pass binds
        what the file already holds, from the views, with no word of a
        missing map. Added: 2026-11-07"""
        path, first = self._built(self._record())
        for name in ("floor_LM.exr", "atlas_LM.exr"):
            os.remove(os.path.join(self.tmp, name))
        with self.assertNoLogs("pythontk", level="WARNING"):
            again = MeshConvert.apply_glb_lightmaps(path, search_dirs=[self.tmp])
        self.assertEqual(again, first)

    def test_a_map_held_twice_is_named_a_tie(self):
        """Two views that join one name hold neither: the pass says so, never
        that the buffer is external, and adds no third. Added: 2026-11-07"""
        self._hdr("floor_LM.exr", 0.5)
        self._hdr("atlas_LM.exr", 0.25)
        path = self._room(self._record())
        with MeshConvert.open_glb(path) as edit:
            for spelling in ("floor_LM.exr", "FLOOR_lm.exr"):
                (index,) = MeshConvert._append_bin_views(edit, [b"\x89PNG"])
                edit.gltf["bufferViews"][index].update(
                    {self.MEDIA.name: spelling, "extras": self.MEDIA.extras("image/png", 1.0)}
                )
            edit.dirty = True
        with self.assertLogs("pythontk", level="WARNING") as logs:
            bound = MeshConvert.apply_glb_lightmaps(path, search_dirs=[self.tmp])
        text = "\n".join(logs.output)
        self.assertIn("2 bufferViews", text)
        self.assertNotIn("external", text)
        self.assertEqual(sorted(r["object"] for r in bound), ["prop"])
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(self.MEDIA.views(gltf, "floor_LM.exr")), 2)

    def test_a_build_without_lightmaps_names_no_map(self):
        """``fbx_to_glb(lightmaps=False)`` ships neither the maps nor the
        record naming them: a file never names maps it does not hold (as the
        build without them carried nothing before). Added: 2026-11-07"""
        self._hdr("floor_LM.exr")
        self._hdr("atlas_LM.exr")
        converted = os.path.join(self.tmp, "converted.glb")
        shutil.copyfile(self._room(self._record()), converted)
        out = _convert_prebuilt(self.tmp, converted, lightmaps=False, lightmap_dirs=[self.tmp])
        gltf = MeshConvert._read_glb(out).gltf
        self.assertNotIn(SceneRecords.LIGHTMAPS.key, gltf.get("extras") or {})
        self.assertFalse([v for v in gltf.get("bufferViews") or [] if v.get("name")])
        self.assertEqual(MeshConvert.lit_nodes(gltf), {})

    def test_the_probe_rides_as_radiance_under_its_name_in_the_records_units(self):
        """No light to draw: the bake's own probe, Radiance bytes under its
        file name; its point and box stay in the record's units beside its
        ``unit_scale``, as the record states them."""
        self._hdr("room_Probe.hdr", 0.75, shape=(16, 32))
        probe = {"map": "room_Probe.hdr", "position": [0.0, 150.0, 0.0], "box": None}
        path, _bound = self._built(self._record(probe=probe))
        view, data = self._view_bytes(path, "room_Probe.hdr")
        self.assertEqual(view["extras"], self.MEDIA.extras(self.MEDIA.radiance))
        self.assertTrue(data.startswith(b"#?"))
        carried = MeshConvert._read_glb(path).gltf["extras"][SceneRecords.LIGHTMAPS.key]
        self.assertEqual(carried["probe"], probe)
        self.assertEqual(carried["unit_scale"], 0.01)

    def test_a_file_nothing_lights_carries_no_probe(self):
        """The probe lights what the bake lit: a file none of whose objects
        the record lights ships the record and no probe, as it shipped no
        block before."""
        self._hdr("room_Probe.hdr", 0.75, shape=(16, 32))
        probe = {"map": "room_Probe.hdr", "position": [0.0, 150.0, 0.0], "box": None}
        record = self._record(
            objects=[{"name": "ghost", "map": "floor_LM.exr"}], probe=probe
        )
        path, bound = self._built(record)
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(bound, [])
        self.assertIsNone(self.MEDIA.view(gltf, "room_Probe.hdr"))
        self.assertEqual(gltf["extras"][SceneRecords.LIGHTMAPS.key], record)

    def test_baked_lights_leave_a_file_the_record_lights(self):
        record = self._record()
        self._hdr("floor_LM.exr")
        self._hdr("atlas_LM.exr")
        path = self._room(record)
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["nodes"][0]["extensions"] = {"KHR_lights_punctual": {"light": 0}}
            edit.gltf["extensions"] = {"KHR_lights_punctual": {"lights": [{"type": "point"}]}}
            edit.gltf["extensionsUsed"] = ["KHR_lights_punctual"]
            edit.dirty = True
        MeshConvert.apply_glb_lightmaps(path, search_dirs=[self.tmp])
        self.assertEqual(MeshConvert.strip_glb_baked_lights(path), ["room"])
        self.assertNotIn("extensions", MeshConvert._read_glb(path).gltf)


class TestGlbLightmaps(unittest.TestCase):
    """The self-feeding lightmap pass: committed bake -> GLB deliverable.

    The record travels INSIDE the GLB (FBX user properties -> node extras via
    FBX2glTF's --user-properties, probe-verified), so the pass takes no scene
    knowledge from the caller -- these tests hand-build that GLB shape and only
    ever pass file paths. What a node reads is what every reader resolves
    (``MeshConvert.lit_nodes``): the record's entry, its map riding the file
    as the bufferView of its name.
    """

    #: Cross-implementation golden value: a constant-0.5 linear EXR must encode
    #: with scalar 0.5 in BOTH this encoder and blendertk's bpy-I/O twin
    #: (test_web_export pins the same number), so the two cannot drift.
    GOLDEN_CONSTANT = 0.5

    @classmethod
    def setUpClass(cls):
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cv2 unavailable; HDR encode untestable")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_lightmap_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ helpers
    def _exr(self, name="room_Lightmap.exr", value=GOLDEN_CONSTANT):
        import cv2
        import numpy as np

        path = os.path.join(self.tmp, name)
        cv2.imwrite(path, np.full((8, 8, 3), value, dtype=np.float32))
        return path

    def _glb(self, gltf, name="scene.glb"):
        json_bytes = json.dumps(gltf).encode("utf-8")
        json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
        blob = (
            struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_bytes))
            + struct.pack("<I4s", len(json_bytes), b"JSON")
            + json_bytes
        )
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(blob)
        return path

    def _scene(
        self,
        manifest,
        objects=("room",),
        texcoord1=True,
        material=0,
        nested=True,
    ):
        """A minimal lit-scene GLB: mesh nodes + the data_export carrier node.

        *nested* picks which of the two real on-disk carrier shapes the manifest
        rides: the FBX2glTF one (``extras.fromFBX.userProperties``, wrapped as
        ``{"type", "value"}``) or a native glTF export's top-level node extras.
        ``None`` writes no carrier at all -- a GLB whose per-node markers
        survived but whose manifest did not.
        """
        attrs = {"POSITION": 0, "TEXCOORD_0": 1}
        if texcoord1:
            attrs["TEXCOORD_1"] = 2
        nodes = [{"name": n, "mesh": i} for i, n in enumerate(objects)]
        nodes.append(self._carrier_node(manifest, nested))
        return {
            "asset": {"version": "2.0"},
            "nodes": nodes,
            "meshes": [
                {"primitives": [{"attributes": dict(attrs), "material": material}]}
                for _ in objects
            ],
            "materials": [{"name": "roomMat"}, {"name": "propMat"}],
        }

    def _manifest(self, entries, version=None):
        """The bake's ``lightmaps`` record over *entries*, in the schema the
        applier reads unless *version* says otherwise. It names each map by
        file name alone: the applier finds it beside the GLB, or through
        ``search_dirs``."""
        return {
            "version": SceneRecords.LIGHTMAPS.version if version is None else version,
            "objects": [_baked(entry) for entry in entries],
        }

    @staticmethod
    def _carrier_node(manifest, nested=True):
        """The ``data_export`` node carrying *manifest* (see :meth:`_scene`)."""
        if nested is None:
            extras = {}
        elif nested:
            extras = {
                "fromFBX": {
                    "userProperties": {
                        "lightmaps": {
                            "type": "eFbxString",
                            "value": json.dumps(manifest),
                        }
                    }
                }
            }
        else:
            extras = {"lightmaps": json.dumps(manifest)}
        return {"name": "data_export", "extras": extras}

    def _tree_scene(self, manifest, paths, prims=None, share_mesh=None, materials=3):
        """A GLB whose mesh nodes sit at *paths* -- name tuples, root first.

        The FBX2glTF shape: a group node per distinct prefix (two paths sharing
        ancestors share those nodes), children listed on the parent. *prims*
        maps a path to its primitives' material indices (default ``[0]``);
        *share_mesh* maps a path to the path whose MESH it reuses (instancing).
        Returns ``(gltf, {path: node_index})``.
        """
        prims, share_mesh = prims or {}, share_mesh or {}
        attrs = {"POSITION": 0, "TEXCOORD_0": 1, "TEXCOORD_1": 2}
        nodes, meshes, index_of = [], [], {}

        def node_for(prefix):
            if prefix not in index_of:
                nodes.append({"name": prefix[-1]})
                index_of[prefix] = len(nodes) - 1
                if len(prefix) > 1:
                    parent = node_for(prefix[:-1])
                    nodes[parent].setdefault("children", []).append(index_of[prefix])
            return index_of[prefix]

        for path in paths:
            node_for(tuple(path))
        for path in paths:
            path = tuple(path)
            if path in share_mesh:
                continue
            meshes.append(
                {
                    "primitives": [
                        {"attributes": dict(attrs), "material": m}
                        for m in prims.get(path, [0])
                    ]
                }
            )
            nodes[index_of[path]]["mesh"] = len(meshes) - 1
        for path, owner in share_mesh.items():
            nodes[index_of[tuple(path)]]["mesh"] = nodes[index_of[tuple(owner)]]["mesh"]
        nodes.append(self._carrier_node(manifest))
        children = {c for n in nodes for c in n.get("children", ())}
        roots = [i for i in range(len(nodes)) if i not in children]
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": roots}],
            "nodes": nodes,
            "meshes": meshes,
            "materials": [{"name": f"mat{i}"} for i in range(materials)],
        }
        return gltf, {p: index_of[tuple(p)] for p in map(tuple, paths)}

    @staticmethod
    def _bound_map(gltf, node, prim=0):
        """The lightmap a mesh node's primitive reads, as every reader resolves
        the record (``MeshConvert.lit_nodes``: its entry, its map riding the
        file as the bufferView of its name) -- the record's file name, or
        ``None`` for a node the record does not light here or a primitive
        with no second UV set to read it through."""
        entry = MeshConvert.lit_nodes(gltf).get(node)
        mesh = gltf["meshes"][gltf["nodes"][node]["mesh"]]
        prims = mesh["primitives"]
        if not entry or "TEXCOORD_1" not in (prims[prim].get("attributes") or {}):
            return None
        return SceneRecords.GLB_MEDIA.file_name(entry["map"])

    def _view(self, glb, name):
        """``(view, bytes)`` of the bufferView of *glb* holding the file
        *name* (``SceneRecords.GLB_MEDIA``), or ``(None, None)``."""
        with MeshConvert.open_glb(glb) as edit:
            index = SceneRecords.GLB_MEDIA.view(edit.gltf, name)
            if index is None:
                return None, None
            view = edit.gltf["bufferViews"][index]
            start = int(view.get("byteOffset", 0))
            return view, bytes(edit.bin_data[start : start + int(view["byteLength"])])

    @staticmethod
    def _carried(glb):
        """The ``lightmaps`` record *glb* carries in its root extras."""
        with MeshConvert.open_glb(glb) as edit:
            return SceneRecords.glb_block(edit.gltf, SceneRecords.LIGHTMAPS)

    def _native_export(self, web_holder="scene"):
        """A GLB shaped like blendertk's native glTF export of one baked floor.

        Every copy is the scene's: the ``floor`` marker names the EXR with the
        authoring folder as ``dir``, the carrier holds the record. *web_holder*
        is where the record rides as well: ``"scene"`` (a native export's
        first scene extras, as a JSON string), ``"root"`` (as ``finish_glb``
        leaves it) or ``None``.
        """
        key = SceneRecords.LIGHTMAPS.key
        marker = {
            "map": "floor_mat_Lightmap.exr",
            "dir": self.tmp,
            "uv_set": "lightmap",
            "intensity": 1.0,
        }
        manifest = self._manifest(
            [{"name": "floor", "map": "floor_mat_Lightmap.exr", "hierarchy": ["floor"]}]
        )
        gltf = {
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [0, 1]}],
            "nodes": [
                {
                    "name": "floor",
                    "mesh": 0,
                    "extras": {"lightmapInfo": json.dumps(marker)},
                },
                {
                    "name": "data_export",
                    "extras": {"lightmaps": json.dumps(manifest)},
                },
            ],
            "meshes": [
                {"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}
            ],
            "materials": [{"name": "floor_mat"}],
        }
        if web_holder == "scene":
            gltf["scenes"][0]["extras"] = {key: json.dumps(manifest)}
        elif web_holder == "root":
            gltf["extras"] = {key: manifest}
        return self._glb(gltf, name="native.glb")

    # ------------------------------------------------------------------ encode
    def test_encode_golden_constant(self):
        """The cross-implementation pin: constant 0.5 -> scalar 0.5, texel 255."""
        import cv2
        import numpy as np

        from pythontk import ImgUtils

        png, scalar = ImgUtils.encode_hdr_for_web(self._exr())
        self.assertAlmostEqual(scalar, self.GOLDEN_CONSTANT, places=5)
        arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(int(arr.max()), 255)
        self.assertEqual(int(arr.min()), 255)  # constant in, constant out

    def test_encode_is_unbiased_between_codes(self):
        """A level between two 8-bit codes averages to itself, not to the nearer code.

        Rounding every texel of a smooth wall to its nearest code terraced its
        gradients into contour bands 0.5-1% apart (a production room at 2K,
        2026-10-02); stochastic rounding keeps the mean, so the bands go and a
        fraction of a code of fine grain stays.
        """
        import cv2
        import numpy as np

        from pythontk import ImgUtils

        # Encoded against a bright texel, the field sits at sRGB code 127.3.
        target = 127.3 / 255.0
        level = ((target + 0.055) / 1.055) ** 2.4
        img = np.full((64, 64, 3), level, dtype=np.float32)
        img[0, 0] = 1.0
        path = os.path.join(self.tmp, "between.exr")
        cv2.imwrite(path, img)
        png, scalar = ImgUtils.encode_hdr_for_web(path, percentile=100.0)
        self.assertAlmostEqual(scalar, 1.0, places=4)
        arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        field = arr[1:, 1:].astype(np.float64)
        self.assertAlmostEqual(float(field.mean()), 127.3, delta=0.05)
        self.assertLessEqual(int(field.max()) - int(field.min()), 1)
        # Grey stays grey: the channels share one dither.
        self.assertTrue((field[..., 0] == field[..., 1]).all())
        # Deterministic: the same map encodes to the same bytes.
        self.assertEqual(png, ImgUtils.encode_hdr_for_web(path, percentile=100.0)[0])

    def test_encode_keeps_a_lit_pool_brighter_than_the_percentile(self):
        """A pool of light wider than 0.5% of the map keeps its peak.

        Divided by the 99.5th percentile, every texel of a lit pool above it
        clipped: measured on a production room, the backsplash and the floor
        pools went flat grey, 58-68 screen codes off. The divisor is now the
        map's peak, held between that percentile and HDR_WEB_HEADROOM times it.
        """
        import cv2
        import numpy as np

        from pythontk import ImgUtils

        img = np.full((100, 100, 3), 0.1, dtype=np.float32)
        img[:2, :] = np.linspace(0.5, 2.0, 100, dtype=np.float32)[None, :, None]
        path = os.path.join(self.tmp, "pool.exr")
        cv2.imwrite(path, img)
        png, scalar = ImgUtils.encode_hdr_for_web(path)
        self.assertAlmostEqual(scalar, 2.0, places=4)
        arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(int(arr[:2].max()), 255)  # only the peak reaches full scale
        self.assertLess(int(arr[:2, :50].max()), 250)

    def test_encode_holds_a_firefly_at_the_headroom(self):
        """One wild texel cannot crush the map: the divisor stops at the cap."""
        import cv2
        import numpy as np

        from pythontk import ImgUtils

        img = np.full((40, 40, 3), 0.1, dtype=np.float32)
        img[3, 7] = 100.0
        path = os.path.join(self.tmp, "firefly.exr")
        cv2.imwrite(path, img)
        _png, scalar = ImgUtils.encode_hdr_for_web(path)
        self.assertAlmostEqual(scalar, ImgUtils.HDR_WEB_HEADROOM * 0.1, places=4)

    def test_encode_clips_a_texel_above_the_cap_without_changing_its_hue(self):
        """A warm light past the cap stays warm: the triple is scaled, not clipped per channel."""
        import cv2
        import numpy as np

        from pythontk import ImgUtils

        img = np.full((40, 40, 3), 0.1, dtype=np.float32)
        img[5, 5] = (5.0, 10.0, 20.0)  # BGR: R 20, G 10, B 5 -- 50x the cap
        path = os.path.join(self.tmp, "warm.exr")
        cv2.imwrite(path, img)
        png, _scalar = ImgUtils.encode_hdr_for_web(path)
        bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)[5, 5]
        linear = ImgUtils.srgb_to_linear(bgr.astype(np.float32) / 255.0)
        self.assertEqual(int(bgr[2]), 255)
        self.assertAlmostEqual(float(linear[1] / linear[2]), 0.5, delta=0.02)
        self.assertAlmostEqual(float(linear[0] / linear[2]), 0.25, delta=0.02)

    def test_encode_pixels_is_the_file_encode_without_the_file(self):
        """The pixel form a host without cv2 (Blender) calls: the same codes and scalar."""
        import cv2
        import numpy as np

        from pythontk import ImgUtils

        rng = np.random.default_rng(3)
        img = (rng.random((16, 24, 3)) ** 3 * 3).astype(np.float32)
        path = os.path.join(self.tmp, "pixels.exr")
        cv2.imwrite(path, img)
        png, scalar = ImgUtils.encode_hdr_for_web(path)
        codes, pixel_scalar = ImgUtils.encode_hdr_pixels_for_web(img)
        self.assertAlmostEqual(pixel_scalar, scalar, places=5)
        np.testing.assert_array_equal(
            codes, cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        )
        flat, flat_scalar = ImgUtils.encode_hdr_pixels_for_web(img.reshape(-1, 3))
        np.testing.assert_array_equal(flat.reshape(codes.shape), codes)
        self.assertEqual(flat_scalar, pixel_scalar)

    def test_encode_percentile_ignores_zero_texels(self):
        """Unbaked (zero) texels must not drag the divisor toward black."""
        import cv2
        import numpy as np

        from pythontk import ImgUtils

        img = np.zeros((8, 8, 3), dtype=np.float32)
        img[0, 0] = 2.0  # one lit texel in a sea of gutter
        path = os.path.join(self.tmp, "sparse.exr")
        cv2.imwrite(path, img)
        _png, scalar = ImgUtils.encode_hdr_for_web(path)
        self.assertAlmostEqual(scalar, 2.0, places=4)

    def test_radiance_encode_keeps_the_range_and_drops_what_rgbe_cannot_hold(self):
        """encode_hdr_radiance: a light fifty times its wall survives (RGBE's
        shared exponent), and alpha, inf, NaN and negatives -- which RGBE has
        no bits for -- come back as 0, never as garbage exponents."""
        import cv2
        import numpy as np

        from pythontk import ImgUtils

        img = np.full((4, 8, 4), 0.5, dtype=np.float32)
        img[0, 0, :3] = 25.0
        img[1, 0, :3] = np.inf
        img[1, 1, :3] = np.nan
        img[1, 2, :3] = -3.0
        path = os.path.join(self.tmp, "probe.exr")
        cv2.imwrite(path, img)
        raw = ImgUtils.encode_hdr_radiance(path)
        self.assertTrue(raw.startswith(b"#?"), "a Radiance file")
        out = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(out.shape, (4, 8, 3))
        self.assertAlmostEqual(float(out[0, 0, 0]), 25.0, delta=0.25)
        self.assertAlmostEqual(float(out[3, 7, 1]), 0.5, delta=0.005)
        self.assertEqual(out[1, :3].tolist(), [[0.0, 0.0, 0.0]] * 3)

    # ------------------------------------------------------------------ applier
    def test_search_dirs_are_tried_in_the_hosts_order(self):
        """The bug this cost a production deliverable: a map is located by
        BASENAME, and the workspace's texture folder routinely holds the atlas
        of that name from an EARLIER bake. Binding it pairs this bake's rects
        with that bake's pixels -- every object sampling someone else's
        lighting, silently (measured on the production room: 46 objects on a
        17-day-old 512px atlas while the fresh one sat in the folder the
        markers named). The record names maps by file name alone; the host
        hands the bake's own folders first (``LightmapRecords.search_dirs``),
        and the first that holds a map is the one it binds."""
        import cv2
        import numpy as np

        fresh_dir = os.path.join(self.tmp, "bake")
        stale_dir = os.path.join(self.tmp, "workspace_sourceimages")
        for folder, value in ((fresh_dir, self.GOLDEN_CONSTANT), (stale_dir, 9.0)):
            os.makedirs(folder, exist_ok=True)
            # Same basename, unmistakably different content -- the encode
            # scalar is what the viewer multiplies back, so binding the wrong
            # file shows as a brightness error, not just a sharpness one.
            cv2.imwrite(
                os.path.join(folder, "room_Lightmap.exr"),
                np.full((8, 8, 3), value, dtype=np.float32),
            )
        glb = self._glb(
            self._scene(self._manifest([{"name": "room", "map": "room_Lightmap.exr"}]))
        )
        records = MeshConvert.apply_glb_lightmaps(
            glb, search_dirs=[fresh_dir, stale_dir]
        )
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(
            records[0]["scale"],
            self.GOLDEN_CONSTANT,
            places=4,
            msg="bound the stale same-named map from a later search dir",
        )

    def test_a_record_names_no_folder_to_look_in(self):
        """No record has named a folder since 0.11.0 (the maps' folders ride
        the private ``lightmap_dirs``): a ``dir`` or ``dirs`` a record still
        carries is no key of its shape, and a map is looked for in the host's
        search folders and beside the file alone -- never where an older
        record pointed. Added: 2026-10-20"""
        import cv2
        import numpy as np

        elsewhere = os.path.join(self.tmp, "named_by_the_record")
        os.makedirs(elsewhere)
        cv2.imwrite(
            os.path.join(elsewhere, "room_Lightmap.exr"),
            np.full((8, 8, 3), self.GOLDEN_CONSTANT, dtype=np.float32),
        )
        manifest = self._manifest([{"name": "room", "map": "room_Lightmap.exr"}])
        manifest["dir"], manifest["dirs"] = elsewhere, [elsewhere]
        glb = self._glb(self._scene(manifest))
        with self.assertLogs("pythontk.file_utils.mesh_convert", level="WARNING"):
            self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])
        self.assertEqual(
            len(MeshConvert.apply_glb_lightmaps(glb, search_dirs=[elsewhere])), 1
        )

    # ------------------------------------------------------------ the probe
    def _probe_exr(self, name="room_Probe.exr"):
        """An HDR probe: a 'light' of 35 on its top rows, 0.5 elsewhere, with alpha."""
        import cv2
        import numpy as np

        img = np.full((16, 32, 4), 0.5, dtype=np.float32)
        img[:4, :, :3] = 35.0
        img[..., 3] = 1.0
        path = os.path.join(self.tmp, name)
        cv2.imwrite(path, img)
        return path

    def _probe_manifest(self, probe_map, **probe):
        """The bake's record with its reflection probe, every length in
        centimetres -- the unit the record states once (``unit_scale``)."""
        manifest = self._manifest(
            [{"name": "room", "map": os.path.basename(self._exr())}]
        )
        manifest["unit_scale"] = 0.01
        manifest["probe"] = {
            "map": probe_map,
            "position": [100.0, 200.0, -300.0],
            "box": {"min": [-400.0, 0.0, -400.0], "max": [400.0, 394.0, 400.0]},
            **probe,
        }
        return manifest

    def _probe_pixels(self, glb, name):
        """The probe the file carries under *name*, decoded (float BGR)."""
        import cv2
        import numpy as np

        view, data = self._view(glb, name)
        self.assertIsNotNone(view, f"no bufferView named {name}")
        self.assertEqual(
            view["extras"], SceneRecords.GLB_MEDIA.extras(SceneRecords.GLB_MEDIA.radiance)
        )
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)

    def test_a_manifest_probe_ships_as_radiance_hdr_in_the_records_units(self):
        """The bake's probe rides the GLB whole: its light at 35, not clipped at
        1 as an 8-bit map would hold it, as Radiance bytes under its file name
        -- and the record states its point and box as the bake did, in its own
        units beside ``unit_scale`` (a reader multiplies them into metres)."""
        manifest = self._probe_manifest(os.path.basename(self._probe_exr()))
        glb = self._glb(self._scene(manifest))
        self.assertEqual(len(MeshConvert.apply_glb_lightmaps(glb)), 1)
        record = self._carried(glb)
        self.assertEqual(record["probe"], manifest["probe"])
        self.assertEqual(record["unit_scale"], 0.01)
        pixels = self._probe_pixels(glb, "room_Probe.exr")
        self.assertEqual(pixels.shape, (16, 32, 3), "alpha is dropped")
        # RGBE: 8-bit mantissas over a shared exponent -- within 1%.
        self.assertAlmostEqual(float(pixels[0, 0, 0]), 35.0, delta=0.35)
        self.assertAlmostEqual(float(pixels[-1, -1, 1]), 0.5, delta=0.005)

    def test_the_bakes_lights_never_ship_as_lights(self):
        """The bake's lights reach the GLB only as data -- the record's
        ``lights``, which a reader lights none of while it has a probe -- and
        as highlights drawn into the probe: a viewer that lit them live paid
        for every light on every pixel, and one that does not know they are
        baked lights the room twice. No ``KHR_lights_punctual`` rides."""
        manifest = self._probe_manifest(os.path.basename(self._probe_exr()))
        manifest["lights"] = [
            {"type": "point", "position": [0.0, 300.0, 0.0],
             "color": [1.0, 1.0, 1.0], "intensity": 40000.0},
        ]
        glb = self._glb(self._scene(manifest))
        self.assertEqual(len(MeshConvert.apply_glb_lightmaps(glb)), 1)
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        record = gltf["extras"][SceneRecords.LIGHTMAPS.key]
        self.assertEqual(record["lights"], manifest["lights"])
        self.assertNotIn("KHR_lights_punctual", gltf.get("extensions") or {})
        self.assertNotIn("KHR_lights_punctual", gltf.get("extensionsUsed") or [])
        # The probe the record names rides the file, under its own name.
        self.assertIsNotNone(SceneRecords.GLB_MEDIA.view(gltf, record["probe"]["map"]))

    def test_a_malformed_light_refuses_the_record_before_anything_is_rebound(self):
        """REGRESSION: a light entry a host wrote wrong (a coordinate that is
        no number, a scalar colour) raised out of the pass AFTER the materials
        were rebound, so the GLB kept its maps in the occlusion slot with no
        ``lightmaps`` block to read them by. The record is read in its declared
        shape now (2026-10-20): such a record is refused whole, and said,
        before anything is touched -- the file stays as it was. The lights a
        record does carry are read as ever (``lights_in_metres``)."""
        manifest = self._probe_manifest(os.path.basename(self._probe_exr()))
        manifest["lights"] = [
            {"name": "NO_NUMBER", "type": "point", "position": [None, 1.0, 2.0], "intensity": 1.0},
            {"name": "SCALAR", "type": "point", "position": [0.0, 1.0, 2.0], "color": 0.5, "intensity": 1.0},
            {"name": "KEY", "type": "point", "position": [0.0, 1.0, 2.0], "intensity": 1.0},
        ]
        glb = self._glb(self._scene(manifest))
        with open(glb, "rb") as fh:
            before = fh.read()
        with self.assertLogs("pythontk.file_utils.mesh_convert", level="WARNING") as logs:
            self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])
        said = "\n".join(logs.output)
        self.assertIn(f"{SceneRecords.LIGHTMAPS.key} record not applied -- ", said)
        self.assertIn(f"{SceneRecords.LIGHTMAPS.key}.lights[0]", said)
        with open(glb, "rb") as fh:
            self.assertEqual(fh.read(), before, "nothing rebound, nothing written")
        self.assertEqual(
            [light["name"] for light in MeshConvert.lights_in_metres(manifest["lights"])], ["KEY"]
        )

    def test_the_probe_survives_bin_compaction(self):
        """The probe's view is held by its name, which no index references:
        compaction keeps it."""
        manifest = self._probe_manifest(os.path.basename(self._probe_exr()))
        glb = self._glb(self._scene(manifest))
        MeshConvert.apply_glb_lightmaps(glb)
        with MeshConvert.open_glb(glb) as edit:
            MeshConvert._compact_bin(edit)
            edit.dirty = True
        pixels = self._probe_pixels(glb, "room_Probe.exr")
        self.assertAlmostEqual(float(pixels[0, 0, 2]), 35.0, delta=0.35)

    def test_a_probe_it_cannot_find_is_warned_and_the_lightmaps_still_ship(self):
        manifest = self._probe_manifest("gone_Probe.exr")
        glb = self._glb(self._scene(manifest))
        with self.assertLogs(
            "pythontk.file_utils.mesh_convert._lightmaps", "WARNING"
        ) as logs:
            self.assertEqual(len(MeshConvert.apply_glb_lightmaps(glb)), 1)
        self.assertTrue(any("gone_Probe.exr" in line for line in logs.output))
        self.assertIsNone(self._view(glb, "gone_Probe.exr")[0])
        self.assertIsNotNone(self._view(glb, "room_Lightmap.exr")[0])
        # The record as the bake wrote it: a reader finds no probe to read.
        self.assertEqual(self._carried(glb)["probe"], manifest["probe"])

    def _native_glb(self):
        """A GLB as a DCC's own glTF exporter leaves one once ``finish_glb``
        has stated it: the record (a probe and a baked ``room``) lifted into
        the root extras, no file riding yet."""
        manifest = self._probe_manifest(os.path.basename(self._probe_exr()))
        manifest["unit_scale"] = 1.0
        gltf = self._scene(manifest, nested=None)
        gltf["extras"] = {SceneRecords.LIGHTMAPS.key: manifest}
        return self._glb(gltf, name="native.glb"), manifest

    def test_a_scene_without_a_probe_embeds_none(self):
        """A bake that captured no probe: its maps ride, and no Radiance view."""
        glb = self._glb(
            self._scene(self._manifest([{"name": "room", "map": os.path.basename(self._exr())}]))
        )
        self.assertEqual(len(MeshConvert.apply_glb_lightmaps(glb)), 1)
        with MeshConvert.open_glb(glb) as edit:
            media = [
                (v.get("extras") or {}).get(SceneRecords.GLB_MEDIA.mime_type)
                for v in edit.gltf.get("bufferViews") or []
            ]
        self.assertEqual(media, ["image/png"])

    class _HostCodec:
        """A host's own HDR codec (``hdr_codec``) -- what Blender hands the
        pipeline in place of cv2: records what it encoded, returns marked bytes."""

        def __init__(self, png):
            self.png, self.encoded = png, []

        def encode_hdr_for_web(self, path, percentile=None):
            self.encoded.append(("web", os.path.basename(path)))
            return self.png, 2.5

        def encode_hdr_radiance(self, path):
            self.encoded.append(("radiance", os.path.basename(path)))
            return b"#?RADIANCE\nhost-encoded"

    @staticmethod
    def _png_bytes():
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (4, 4), (128, 128, 128)).save(buf, format="PNG")
        return buf.getvalue()

    def test_a_host_without_cv2_hands_its_own_radiance_encoder(self):
        """Blender ships no cv2: the GLB its own exporter wrote -- the record
        lifted into the root extras by ``finish_glb`` -- takes its files through
        Blender's own encoders, and the bytes ride as given. Added: 2026-10-17"""
        glb, _manifest = self._native_glb()
        codec = self._HostCodec(self._png_bytes())
        records = MeshConvert.apply_glb_lightmaps(glb, hdr_codec=codec)
        self.assertEqual([r["object"] for r in records], ["room"])
        self.assertEqual(
            sorted(codec.encoded),
            [("radiance", "room_Probe.exr"), ("web", "room_Lightmap.exr")],
        )
        self.assertEqual(self._view(glb, "room_Probe.exr")[1], b"#?RADIANCE\nhost-encoded")

    def test_a_host_codec_binds_the_lightmaps_and_the_probe_with_no_cv2(self):
        """REGRESSION (2026-10-17): a GLB built inside Blender -- whose Python has
        no cv2 -- warned every map "not encoded" and shipped unlit, its probe
        with it. The host's codec encodes every float image the pass reads."""
        exr = self._exr()
        manifest = self._probe_manifest(os.path.basename(self._probe_exr()))
        manifest["objects"][0]["map"] = os.path.basename(exr)
        glb = self._glb(self._scene(manifest))
        codec = self._HostCodec(self._png_bytes())
        with unittest.mock.patch.dict(sys.modules, {"cv2": None}):
            records = MeshConvert.apply_glb_lightmaps(glb, hdr_codec=codec)
        self.assertEqual(len(records), 1, "the map bound without cv2")
        self.assertEqual(
            sorted(codec.encoded),
            [("radiance", "room_Probe.exr"), ("web", os.path.basename(exr))],
        )
        view, data = self._view(glb, os.path.basename(exr))
        self.assertEqual(view["extras"], SceneRecords.GLB_MEDIA.extras("image/png", 2.5))
        self.assertEqual(data, codec.png)
        self.assertEqual(self._view(glb, "room_Probe.exr")[1], b"#?RADIANCE\nhost-encoded")

    def test_without_cv2_or_a_host_codec_the_maps_are_warned_not_encoded(self):
        """The default codec is cv2's: absent, every map says so -- the warning
        that was the only sign a Blender-built GLB shipped unlit."""
        exr = self._exr()
        glb = self._glb(
            self._scene(
                self._manifest([{"name": "room", "map": os.path.basename(exr)}])
            )
        )
        with (
            unittest.mock.patch.dict(sys.modules, {"cv2": None}),
            self.assertLogs(
                "pythontk.file_utils.mesh_convert._lightmaps", "WARNING"
            ) as logs,
        ):
            self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])
        self.assertTrue(any("cv2" in line for line in logs.output), logs.output)

    def test_a_manifest_entry_is_matched_to_its_node_namespace_tolerantly(self):
        """An entry is read through every reader's match rule: a record naming
        "room" lights the GLB node "NS:room" -- comparing names would miss on
        exactly the namespaced scenes that tolerance exists for, and miss
        silently -- and the record rides as written, its names kept."""
        exr = self._exr()
        entry = {"name": "room", "map": os.path.basename(exr)}
        scene = self._scene(self._manifest([entry]))
        scene["nodes"][0]["name"] = "NS:room"
        glb = self._glb(scene)

        self.assertEqual(len(MeshConvert.apply_glb_lightmaps(glb)), 1)
        with MeshConvert.open_glb(glb) as edit:
            self.assertEqual(self._bound_map(edit.gltf, 0), "room_Lightmap.exr")
        (shipped,) = self._carried(glb)["objects"]
        self.assertEqual(shipped, _baked(entry), "published names are kept")

    def test_a_native_export_states_its_lightmaps_once(self):
        """blendertk's native glTF export writes the record twice -- the
        first scene's extras and its carrier -- and each baked object's
        ``lightmapInfo`` marker, with the absolute authoring folder as ``dir``
        (measured in Blender 5.1). ``finish_glb`` lifts the record into the
        root extras, where every reader looks, and the copies on its nodes --
        the marker and the carrier's record -- leave it, authoring folder and
        all."""
        glb = self._native_export()
        self.assertTrue(MeshConvert.finish_glb(glb))
        with MeshConvert.open_glb(glb) as edit:
            floor = edit.gltf["nodes"][0]
            raw = json.dumps(edit.gltf)
            self.assertIsNotNone(
                SceneRecords.glb_block(edit.gltf, SceneRecords.LIGHTMAPS)
            )
        self.assertNotIn("lightmapInfo", floor.get("extras") or {})
        self.assertIsNone(_carried_lightmaps(glb))
        self.assertNotIn(
            self.tmp.replace("\\", ""),
            raw.replace("\\", ""),
            "the authoring directory shipped inside the GLB",
        )
        self.assertFalse(MeshConvert.finish_glb(glb), "idempotent")
        self.assertTrue(MeshConvert.verify_glb(glb)["lightmap"])

    def test_the_retired_fix_drops_the_copies_beside_a_block(self):
        """``fix_glb_lightmap_metadata`` (retired 2026-10-20 for ``finish_glb``,
        removed in 0.18.0) warns, and serves: a file whose root extras carry
        the record loses the copies on its nodes; one carrying none there is
        left as found -- the copies may be all it says of them."""
        glb = self._native_export(web_holder="root")
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(MeshConvert.fix_glb_lightmap_metadata(glb), 2)
        with MeshConvert.open_glb(glb) as edit:
            self.assertNotIn("lightmapInfo", edit.gltf["nodes"][0].get("extras") or {})
        self.assertIsNone(_carried_lightmaps(glb))

        bare = self._native_export(web_holder=None)
        with MeshConvert.open_glb(bare) as edit:
            before = json.dumps(edit.gltf, sort_keys=True)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(MeshConvert.fix_glb_lightmap_metadata(bare), 0)
        with MeshConvert.open_glb(bare) as edit:
            self.assertEqual(json.dumps(edit.gltf, sort_keys=True), before)

    def test_a_top_level_manifest_carrier_is_read(self):
        """Both on-disk carrier shapes are read: FBX2glTF nests a user
        property under ``extras.fromFBX.userProperties``, a native glTF export
        writes it at the top of the node's extras. Probing only the nested one
        no-ops the whole applier on a natively exported GLB, so this is
        asserted end to end -- the record carried and its map riding."""
        exr = self._exr()
        manifest = self._manifest([{"name": "room", "map": os.path.basename(exr)}])
        glb = self._glb(self._scene(manifest, nested=False))
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(
            len(records),
            1,
            "a natively exported GLB carries its record as a TOP-LEVEL node "
            "extra; probing only the FBX2glTF shape no-ops the whole applier",
        )
        self.assertEqual(self._carried(glb), manifest)
        self.assertIsNotNone(self._view(glb, os.path.basename(exr))[0])

    def test_markers_without_a_manifest_are_a_clean_no_op(self):
        """The documented limit of what this applier can repair.

        A deliverable can arrive carrying per-node markers and NO manifest at
        all -- blendertk's native glTF export writes exactly that today (checked
        on ``ROOM_ENV_both.glb``: 7 nodes with a top-level ``lightmapInfo``,
        no ``lightmaps`` anywhere). There is nothing to bind and nothing
        authoritative to correct the markers AGAINST, so they are left as found
        rather than guessed at or stripped, and the repair belongs in the
        producer. Pinned so the no-op is a stated contract rather than an
        accident of the early return.
        """
        exr = self._exr()
        stale = {"map": os.path.basename(exr), "uv_set": "lightmap", "intensity": 1.0}
        scene = self._scene(
            self._manifest([{"name": "room", "map": os.path.basename(exr)}]),
            nested=None,
        )
        scene["nodes"][0]["extras"] = {"lightmapInfo": json.dumps(stale)}
        glb = self._glb(scene)

        self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])
        self.assertIsNone(_carried_lightmaps(glb))
        with MeshConvert.open_glb(glb) as edit:
            self.assertNotIn(SceneRecords.LIGHTMAPS.key, edit.gltf.get("extras") or {})
            self.assertEqual(
                json.loads(edit.gltf["nodes"][0]["extras"]["lightmapInfo"]), stale
            )

    def test_the_retired_manifest_reader_reads_the_carriers_record(self):
        """``read_glb_lightmap_manifest`` (retired 2026-10-20, removed in
        0.18.0: a finished GLB carries no record -- its lightmaps are its
        ``extras.lightmaps`` block) warns, and serves the carrier's record
        while a build still holds one."""
        glb = self._glb(
            self._scene(self._manifest([{"name": "room", "map": "room_Lightmap.exr"}]))
        )
        with self.assertWarns(DeprecationWarning):
            served = MeshConvert.read_glb_lightmap_manifest(glb)
        self.assertEqual(served, _carried_lightmaps(glb))
        self.assertEqual(served["objects"][0]["name"], "room")

    def test_the_retired_hint_scrub_hands_a_snapshot_back_as_given(self):
        """``without_locate_hints`` (retired 2026-10-20, removed in 0.18.0)
        warns, and serves: no record has carried a folder hint since 0.11.0
        (the maps' folders ride the private ``lightmap_dirs``), so a snapshot
        comes back as it was given."""
        snapshot = {
            SceneRecords.LIGHTMAPS.key: {
                "version": SceneRecords.LIGHTMAPS.version,
                "objects": [{"name": "room", "map": "room_Lightmap.exr"}],
            },
            SceneRecords.SHOTS.key: {"shots": [1]},
        }
        with self.assertWarns(DeprecationWarning):
            self.assertIs(MeshConvert.without_locate_hints(snapshot), snapshot)

    def test_no_manifest_is_a_clean_noop(self):
        glb = self._glb(
            {
                "asset": {"version": "2.0"},
                "nodes": [{"name": "room", "mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "materials": [{"name": "roomMat"}],
            }
        )
        with open(glb, "rb") as f:
            before = f.read()
        self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])
        with open(glb, "rb") as f:
            self.assertEqual(f.read(), before, "no-op must not rewrite")

    def test_namespaced_node_binds_a_stripped_manifest_name(self):
        """Namespace tolerance, the direction that shipped black racks: an
        older publisher stripped ``NS:`` from the manifest while the FBX kept
        it on the node -- exact matching bound nothing and every referenced
        object lost its lighting. The stripped leaf is unambiguous here, so
        it must bind."""
        exr = self._exr()
        glb = self._glb(
            self._scene(
                self._manifest([{"name": "room", "map": os.path.basename(exr)}]),
                objects=("PROPS_DA:room",),
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["object"], "room")

    def test_manifest_namespace_binds_a_stripped_node(self):
        """The reverse direction: a namespaced manifest against an exporter
        that flattened namespaces out of the node names."""
        exr = self._exr()
        glb = self._glb(
            self._scene(
                self._manifest(
                    [{"name": "PROPS_DA:room", "map": os.path.basename(exr)}]
                ),
                objects=("room",),
            )
        )
        self.assertEqual(len(MeshConvert.apply_glb_lightmaps(glb)), 1)

    def test_ambiguous_leaf_is_not_guessed(self):
        """Two nodes stripping to the same leaf: binding either would put one
        object's lighting on the other -- skip loudly instead."""
        exr = self._exr()
        glb = self._glb(
            self._scene(
                self._manifest([{"name": "room", "map": os.path.basename(exr)}]),
                objects=("NS_A:room", "NS_B:room"),
            )
        )
        self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])

    def test_exact_match_outranks_leaf_fallback(self):
        """When the record's name exists verbatim, a same-leaf sibling under a
        namespace must not attract the lightmap."""
        exr = self._exr()
        glb = self._glb(
            self._scene(
                self._manifest(
                    [
                        {
                            "name": "room",
                            "map": os.path.basename(exr),
                            "scaleOffset": [0.5, 0.5, 0.0, 0.0],
                        }
                    ]
                ),
                objects=("room", "NS_A:room"),
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(len(records), 1)
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        self.assertEqual(self._bound_map(gltf, 0), "room_Lightmap.exr")
        self.assertIsNone(self._bound_map(gltf, 1), "the namespaced sibling")

    # -- Same-named objects. FBX carries leaf names only, so a scene that reuses
    #    one (``|GRP|BODY|MACHINE_B|BODY|BODY`` beside ``|GRP|BODY|MACHINE_A|BODY``,
    #    the production room) publishes two ``BODY`` entries that the
    #    GLB can tell apart only by where each node SITS.
    LX = ("RootNode", "GRP", "BODY", "MACHINE_B", "BODY", "BODY")
    AL = ("RootNode", "GRP", "BODY", "MACHINE_A", "BODY")

    def _two_bodies(self, hierarchy=True):
        lx_exr = self._exr("LX_Lightmap.exr", value=(0.5, 0.25, 0.125))
        al_exr = self._exr("AL_Lightmap.exr", value=(0.125, 0.25, 0.5))
        entries = [
            {"name": "BODY", "map": os.path.basename(lx_exr)},
            {"name": "BODY", "map": os.path.basename(al_exr)},
        ]
        if hierarchy:
            # The Maya DAG path: no synthetic FBX root above it.
            entries[0]["hierarchy"] = list(self.LX[1:])
            entries[1]["hierarchy"] = list(self.AL[1:])
        gltf, index = self._tree_scene(
            self._manifest(entries),
            [self.LX, self.AL],
            prims={self.LX: [0], self.AL: [1]},
        )
        return self._glb(gltf), index

    def test_same_named_objects_bind_their_own_maps_by_hierarchy(self):
        """Reported from the WebXR preview: machine A rendered near-black.
        Both machine bodies are ``BODY``, the exact-name match returned BOTH
        nodes, and the first entry bound machine B's map onto machine A
        too -- its own was then refused as a conflict. Each entry's hierarchy
        now names the one node it belongs to."""
        glb, index = self._two_bodies()
        MeshConvert.apply_glb_lightmaps(glb)
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        self.assertEqual(self._bound_map(gltf, index[self.LX]), "LX_Lightmap.exr")
        self.assertEqual(self._bound_map(gltf, index[self.AL]), "AL_Lightmap.exr")

    def test_same_named_objects_without_a_hierarchy_are_not_guessed(self):
        """A manifest from before the hierarchy key cannot tell the two apart:
        binding either map to both nodes puts one object's lighting on the
        other, so neither binds -- and it says so."""
        glb, _index = self._two_bodies(hierarchy=False)
        with self.assertLogs(
            "pythontk.file_utils.mesh_convert", level="WARNING"
        ) as caught:
            records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(records, [])
        self.assertTrue(
            any("'BODY'" in m and "ambiguous" in m for m in caught.output),
            caught.output,
        )

    def test_a_stale_same_named_entry_loses_to_the_one_that_matches(self):
        """A marker left on a hidden ``|TEMP|PROPS|TABLE`` by an old bake
        publishes a second ``TABLE`` entry beside the exported
        ``|STATIC|PROPS|TABLE``'s. Scene ORDER decided which map the table
        wore (and the report called both bound); the hierarchy decides it now,
        and the stale entry counts as out of scope, not as a miss."""
        stale = self._exr("stale_Lightmap.exr", value=(0.5, 0.5, 0.125))
        live = self._exr("table_Lightmap.exr", value=(0.125, 0.5, 0.5))
        table = ("RootNode", "STATIC", "PROPS", "TABLE")
        manifest = self._manifest(
            [
                {
                    "name": "TABLE",
                    "map": os.path.basename(stale),
                    "hierarchy": ["TEMP", "PROPS", "TABLE"],
                },
                {
                    "name": "TABLE",
                    "map": os.path.basename(live),
                    "hierarchy": ["STATIC", "PROPS", "TABLE"],
                },
            ]
        )
        gltf, index = self._tree_scene(manifest, [table])
        glb = self._glb(gltf)
        coverage = MeshConvert.lightmap_manifest_coverage(glb)
        records = MeshConvert.apply_glb_lightmaps(glb)
        with MeshConvert.open_glb(glb) as edit:
            self.assertEqual(
                self._bound_map(edit.gltf, index[table]), "table_Lightmap.exr"
            )
        self.assertEqual(
            MeshConvert.lightmap_report(coverage, records),
            {"expected": 1, "bound": 1, "unbound": [], "out_of_scope": 1},
        )

    def test_an_exact_name_outranks_a_namespace_twin_the_file_does_not_carry(self):
        """The file carries ``PROPS_DA:prop352``; the manifest also lists the
        unpushed ``PROPS_RF:prop352``. Grouped with the exact match on the one
        node, the namespace-stripped twin tied it -- by name, or by a hierarchy
        that reads the same once namespaces are ignored -- and the object that
        WAS exported shipped unlit. The exact name wins wherever the hierarchy
        cannot tell the two apart, in either manifest order, and the twin is
        out of scope."""
        da = self._exr("DA_Lightmap.exr", value=(0.5, 0.25, 0.125))
        rf = self._exr("RF_Lightmap.exr", value=(0.125, 0.25, 0.5))
        node = ("RootNode", "PROPS", "PROPS_DA:prop352")
        for hierarchy in (False, True):
            for twin_first in (True, False):
                with self.subTest(hierarchy=hierarchy, twin_first=twin_first):
                    entries = [
                        {"name": "PROPS_DA:prop352", "map": os.path.basename(da)},
                        {"name": "PROPS_RF:prop352", "map": os.path.basename(rf)},
                    ]
                    if hierarchy:
                        entries[0]["hierarchy"] = ["PROPS", "PROPS_DA:prop352"]
                        entries[1]["hierarchy"] = ["PROPS", "PROPS_RF:prop352"]
                    if twin_first:
                        entries.reverse()
                    gltf, index = self._tree_scene(self._manifest(entries), [node])
                    glb = self._glb(gltf, name=f"prop_{hierarchy}_{twin_first}.glb")
                    coverage = MeshConvert.lightmap_manifest_coverage(glb)
                    self.assertEqual(
                        coverage,
                        {
                            "present": ["PROPS_DA:prop352"],
                            "ambiguous": [],
                            "absent": ["PROPS_RF:prop352"],
                        },
                    )
                    records = MeshConvert.apply_glb_lightmaps(glb)
                    with MeshConvert.open_glb(glb) as edit:
                        self.assertEqual(
                            self._bound_map(edit.gltf, index[node]),
                            "DA_Lightmap.exr",
                        )
                    self.assertEqual(
                        MeshConvert.lightmap_report(coverage, records)["unbound"], []
                    )

    def test_overlapping_candidates_never_bind_one_node_twice(self):
        """``a:BODY`` names its node exactly; a stripped ``BODY`` entry could be
        either ``a:BODY`` or ``b:BODY``. Settled as two separate groups, BOTH
        bound ``a:BODY`` -- the second through a copy of its material -- so the
        node wore whichever came last in the manifest. One assignment now
        settles them: the exact entry keeps its node, and the stripped one,
        whose best node that was, is out of scope."""
        a_exr = self._exr("A_Lightmap.exr", value=(0.5, 0.25, 0.125))
        s_exr = self._exr("S_Lightmap.exr", value=(0.125, 0.25, 0.5))
        a_node, b_node = ("RootNode", "a:GRP", "a:BODY"), ("RootNode", "b:BODY")
        manifest = self._manifest(
            [
                {
                    "name": "a:BODY",
                    "map": os.path.basename(a_exr),
                    "hierarchy": ["a:GRP", "a:BODY"],
                },
                {
                    "name": "BODY",
                    "map": os.path.basename(s_exr),
                    "hierarchy": ["GRP", "BODY"],
                },
            ]
        )
        gltf, index = self._tree_scene(
            manifest, [a_node, b_node], prims={a_node: [0], b_node: [1]}
        )
        glb = self._glb(gltf)
        coverage = MeshConvert.lightmap_manifest_coverage(glb)
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual([r["object"] for r in records], ["a:BODY"])
        with MeshConvert.open_glb(glb) as edit:
            self.assertEqual(
                self._bound_map(edit.gltf, index[a_node]), "A_Lightmap.exr"
            )
            self.assertIsNone(self._bound_map(edit.gltf, index[b_node]))
        self.assertEqual(
            coverage, {"present": ["a:BODY"], "ambiguous": [], "absent": ["BODY"]}
        )

    def test_a_same_named_object_outside_the_push_is_out_of_scope(self):
        """A push of two of a room's three ``BODY`` machines. The third
        entry's best nodes tie -- only the name matches either -- and each went
        to its own machine, so the third object is simply not in this file. It
        was reported ambiguous: the preview said one object shipped UNLIT when
        every object in the file was lit."""
        maps = [
            self._exr(f"{tag}_Lightmap.exr", value=value)
            for tag, value in (
                ("LX", (0.5, 0.25, 0.125)),
                ("AL", (0.125, 0.25, 0.5)),
                ("VEND", (0.25, 0.5, 0.125)),
            )
        ]
        hierarchies = [
            list(self.LX[1:]),
            list(self.AL[1:]),
            ["GRP", "BODY", "VENDING", "BODY"],
        ]
        manifest = self._manifest(
            [
                {"name": "BODY", "map": os.path.basename(m), "hierarchy": h}
                for m, h in zip(maps, hierarchies)
            ]
        )
        gltf, index = self._tree_scene(
            manifest, [self.LX, self.AL], prims={self.LX: [0], self.AL: [1]}
        )
        glb = self._glb(gltf)
        coverage = MeshConvert.lightmap_manifest_coverage(glb)
        self.assertEqual(coverage["ambiguous"], [])
        self.assertEqual(coverage["absent"], ["BODY"])
        with self.assertLogs(
            "pythontk.file_utils.mesh_convert", level="DEBUG"
        ) as caught:
            records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertFalse([m for m in caught.output if "ambiguous" in m], caught.output)
        with MeshConvert.open_glb(glb) as edit:
            self.assertEqual(
                self._bound_map(edit.gltf, index[self.LX]), "LX_Lightmap.exr"
            )
            self.assertEqual(
                self._bound_map(edit.gltf, index[self.AL]), "AL_Lightmap.exr"
            )
        report = MeshConvert.lightmap_report(coverage, records)
        self.assertEqual((report["unbound"], report["out_of_scope"]), ([], 1))

    def test_a_lighting_choice_survives_the_blocks_a_build_adds_later(self):
        """A conversion writes the handoff before its passes add their blocks.
        A lightmap-only choice of the export (``lightmappedMaterials.
        envMapIntensity``) gated out then -- the file held no lightmaps yet --
        could never be read back once the lightmaps pass wrote its block: the
        GLB published the policy's 1.0 for the export's 0.5. Added: 2026-10-20"""
        exr = self._exr()
        gltf = self._scene(
            self._manifest([{"name": "room", "map": os.path.basename(exr)}])
        )
        chosen = {"lightmappedMaterials": {"envMapIntensity": 0.5}}
        record = HandoffRecord.block(
            "fbx", [SceneRecords.LIGHTMAPS.key], {"application": "maya"}, chosen
        )
        props = gltf["nodes"][-1]["extras"]["fromFBX"]["userProperties"]
        props[HandoffRecord.KEY] = {"type": "eFbxString", "value": json.dumps(record)}
        out = _convert_prebuilt(
            self.tmp, self._glb(gltf, "prebuilt.glb"), lightmap_dirs=[self.tmp]
        )
        with MeshConvert.open_glb(out) as edit:
            extras = edit.gltf["extras"]
        self.assertIn(SceneRecords.LIGHTMAPS.key, extras)
        rendering = extras[HandoffRecord.KEY]["rendering"]
        self.assertEqual(rendering["lightmappedMaterials"]["envMapIntensity"], 0.5)

    def test_a_malformed_entry_refuses_the_record_and_says_so(self):
        """An entry its shape refuses -- no object, no name, a name that is no
        string -- and the record is none a reader reads: refused whole and
        said, before anything is bound, and nothing raised. A non-dict or
        nameless entry once resolved as the name ``""`` -- the key the
        file's NAMELESS mesh nodes sit under -- and raised mid-pass; until
        2026-10-20 such an entry was skipped and the rest bound. The record is
        read in its declared shape alone now, and the cure is a re-export."""
        exr = self._exr()
        gltf = self._scene(
            self._manifest(
                [
                    "junk",
                    {"map": "orphan.exr"},
                    {"name": 7, "map": "orphan.exr"},
                    {"name": "room", "map": os.path.basename(exr)},
                ]
            ),
            objects=("room", "", ""),
        )
        glb = self._glb(gltf)
        with open(glb, "rb") as fh:
            before = fh.read()
        with self.assertLogs("pythontk.file_utils.mesh_convert", level="WARNING") as logs:
            self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])
        said = "\n".join(logs.output)
        self.assertIn(f"{SceneRecords.LIGHTMAPS.key} record not applied -- ", said)
        self.assertIn(f"{SceneRecords.LIGHTMAPS.key}.objects[0]", said)
        self.assertIn("re-export the file", said)
        with open(glb, "rb") as fh:
            self.assertEqual(fh.read(), before, "nothing bound, nothing written")

    def test_a_material_two_lightmaps_share_is_read_per_object(self):
        """The machines' GLASS is a SECONDARY material on both bodies, each of
        which bakes its own map: each body reads its own map on every
        primitive, the glass included -- the record names the object, not its
        material -- and the file copies no material to say so (a reader binds
        each object its own; the viewer, on a copy it makes at load)."""
        lx, al = ("RootNode", "LX_BODY"), ("RootNode", "AL_BODY")
        lx_exr = self._exr("LX_Lightmap.exr", value=(0.5, 0.25, 0.125))
        al_exr = self._exr("AL_Lightmap.exr", value=(0.125, 0.25, 0.5))
        manifest = self._manifest(
            [
                {"name": "LX_BODY", "map": os.path.basename(lx_exr)},
                {"name": "AL_BODY", "map": os.path.basename(al_exr)},
            ]
        )
        # mat2 is the shared glass: LX_BODY's 2nd primitive, AL_BODY's 1st.
        gltf, index = self._tree_scene(
            manifest, [lx, al], prims={lx: [0, 2], al: [2, 1]}
        )
        glb = self._glb(gltf)
        records = MeshConvert.apply_glb_lightmaps(glb)
        with MeshConvert.open_glb(glb) as edit:
            g = edit.gltf
        self.assertEqual(self._bound_map(g, index[lx], 0), "LX_Lightmap.exr")
        self.assertEqual(self._bound_map(g, index[lx], 1), "LX_Lightmap.exr")
        self.assertEqual(self._bound_map(g, index[al], 0), "AL_Lightmap.exr")
        self.assertEqual(self._bound_map(g, index[al], 1), "AL_Lightmap.exr")
        self.assertEqual(sorted({r["object"] for r in records}), ["AL_BODY", "LX_BODY"])
        self.assertEqual(len(g["materials"]), 3, "no material copied")

    def test_per_object_maps_on_instances_each_bind_their_own(self):
        """Per-Object packing bakes every INSTANCE its own map, while FBX2glTF
        keeps instancing as one mesh -- one material -- behind several nodes.
        Each instance node reads its own map, the mesh still shared: no mesh
        or material entry is copied for it."""
        a, b = ("RootNode", "FLOOR_A"), ("RootNode", "FLOOR_B")
        a_exr = self._exr("FLOOR_A_Lightmap.exr", value=(0.5, 0.25, 0.125))
        b_exr = self._exr("FLOOR_B_Lightmap.exr", value=(0.125, 0.25, 0.5))
        manifest = self._manifest(
            [
                {"name": "FLOOR_A", "map": os.path.basename(a_exr)},
                {"name": "FLOOR_B", "map": os.path.basename(b_exr)},
            ]
        )
        gltf, index = self._tree_scene(manifest, [a, b], share_mesh={b: a})
        glb = self._glb(gltf)
        MeshConvert.apply_glb_lightmaps(glb)
        with MeshConvert.open_glb(glb) as edit:
            g = edit.gltf
        self.assertEqual(self._bound_map(g, index[a]), "FLOOR_A_Lightmap.exr")
        self.assertEqual(self._bound_map(g, index[b]), "FLOOR_B_Lightmap.exr")
        self.assertEqual(g["nodes"][index[a]]["mesh"], g["nodes"][index[b]]["mesh"])
        self.assertEqual((len(g["meshes"]), len(g["materials"])), (1, 3))

    def test_missing_texcoord1_is_skipped_loudly_not_bound(self):
        """No second UV set = the FBX shipped without lightmap UVs; reading the
        map anyway would sample it through the texture UVs -- silently wrong.
        Warned; the map rides nothing, and the record rides as it stands."""
        exr = self._exr()
        manifest = self._manifest([{"name": "room", "map": os.path.basename(exr)}])
        glb = self._glb(self._scene(manifest, texcoord1=False))
        with self.assertLogs("pythontk.file_utils.mesh_convert", "WARNING") as logs:
            self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])
        self.assertTrue(any("TEXCOORD_1" in line for line in logs.output), logs.output)
        self.assertIsNone(self._view(glb, os.path.basename(exr))[0])
        self.assertEqual(self._carried(glb), manifest)
        with MeshConvert.open_glb(glb) as edit:
            self.assertEqual(MeshConvert.lit_nodes(edit.gltf), {})

    def test_unresolvable_map_is_skipped(self):
        glb = self._glb(
            self._scene(self._manifest([{"name": "room", "map": "gone.exr"}]))
        )
        self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])

    def test_a_record_of_another_version_binds_nothing_and_says_so(self):
        """Misreading another schema would bind wrong data -- a newer one's
        fields, or an older one's units and box -- so the pass refuses both,
        and says the file is to be re-exported."""
        exr = self._exr()
        current = SceneRecords.LIGHTMAPS.version
        for version in (current - 1, 99):
            with self.subTest(version=version):
                glb = self._glb(
                    self._scene(
                        self._manifest(
                            [{"name": "room", "map": os.path.basename(exr)}],
                            version=version,
                        )
                    )
                )
                with self.assertLogs(
                    "pythontk.file_utils.mesh_convert", level="WARNING"
                ) as logs:
                    self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])
                self.assertTrue(
                    any("re-export the file" in line for line in logs.output),
                    logs.output,
                )

    def test_shared_material_with_different_maps_are_read_per_object(self):
        """Per-object maps on one material: each object reads its own map at
        its own scale -- the record names the object -- and both still wear
        the material as authored. (The binder this replaces bound the first in
        place and copied the material for the second; a reader binds each
        object its own now.)"""
        # Written for their side effect: the record below names them by file.
        _a, _b = self._exr("a.exr"), self._exr("b.exr", value=0.25)
        glb = self._glb(
            self._scene(
                self._manifest(
                    [
                        {"name": "room", "map": "a.exr"},
                        {"name": "prop", "map": "b.exr"},
                    ]
                ),
                objects=("room", "prop"),
                material=0,  # both meshes share materials[0]
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual([r["object"] for r in records], ["room", "prop"])
        self.assertEqual([r["map"] for r in records], ["a.exr", "b.exr"])
        self.assertEqual([r["scale"] for r in records], [0.5, 0.25])
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        worn = [
            gltf["meshes"][gltf["nodes"][i]["mesh"]]["primitives"][0]["material"]
            for i in (0, 1)
        ]
        self.assertEqual(worn, [0, 0], "the material as authored")
        self.assertEqual([self._bound_map(gltf, i) for i in (0, 1)], ["a.exr", "b.exr"])
        for name, scale in (("a.exr", 0.5), ("b.exr", 0.25)):
            self.assertEqual(self._view(glb, name)[0]["extras"]["scale"], scale)

    def test_shared_atlas_binds_every_object_once_per_material(self):
        """The normal atlas case: two objects, one material, ONE map -- one
        view, two records."""
        exr = self._exr("atlas.exr")
        glb = self._glb(
            self._scene(
                self._manifest(
                    [
                        {"name": "room", "map": os.path.basename(exr)},
                        {"name": "prop", "map": os.path.basename(exr)},
                    ]
                ),
                objects=("room", "prop"),
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(len(records), 2)
        with MeshConvert.open_glb(glb) as edit:
            self.assertEqual(len(edit.gltf["bufferViews"]), 1, "the atlas rides once")

    def test_an_UNBAKED_object_sharing_the_material_keeps_it_unlit(self):
        """A bake is per OBJECT: a prop the bake never saw, sharing the wall's
        material, once wore the wall's lightmap through its own second UV set
        -- and the viewer counted the material as lightmapped, 1/1. The record
        lights the wall alone, and the shared material stays as authored."""
        exr = self._exr()
        glb = self._glb(
            self._scene(
                self._manifest([{"name": "room", "map": os.path.basename(exr)}]),
                objects=("room", "prop"),
                material=0,  # both meshes wear materials[0]; only room is baked
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual([r["object"] for r in records], ["room"])
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        self.assertEqual(self._bound_map(gltf, 0), "room_Lightmap.exr")
        self.assertIsNone(
            self._bound_map(gltf, 1), "the unbaked prop wears the room's lighting"
        )
        prop = gltf["meshes"][gltf["nodes"][1]["mesh"]]["primitives"][0]["material"]
        self.assertEqual(prop, 0, "the prop keeps the material as authored")
        self.assertNotIn("occlusionTexture", gltf["materials"][0])

    def test_an_UNBAKED_instance_of_a_baked_mesh_keeps_it_unlit(self):
        """The same through instancing: FBX2glTF keeps one mesh -- one
        material -- behind every instance node; the record lights the baked
        instance alone, and the mesh stays shared."""
        a, b = ("RootNode", "FLOOR_A"), ("RootNode", "FLOOR_B")
        exr = self._exr("FLOOR_A_Lightmap.exr")
        manifest = self._manifest([{"name": "FLOOR_A", "map": os.path.basename(exr)}])
        gltf, index = self._tree_scene(manifest, [a, b], share_mesh={b: a})
        glb = self._glb(gltf)
        MeshConvert.apply_glb_lightmaps(glb)
        with MeshConvert.open_glb(glb) as edit:
            g = edit.gltf
        self.assertEqual(self._bound_map(g, index[a]), "FLOOR_A_Lightmap.exr")
        self.assertIsNone(self._bound_map(g, index[b]), "FLOOR_B was never baked")
        self.assertEqual(len(g["meshes"]), 1)

    def test_a_baked_object_whose_map_is_missing_is_not_lit_by_its_neighbour(self):
        """An object whose map was not found is reported UNLIT -- and wore the
        neighbour's lighting instead whenever the two share a material,
        because the neighbour bound that material in place. What the report
        says is now what renders."""
        self._exr("room_Lightmap.exr")
        glb = self._glb(
            self._scene(
                self._manifest(
                    [
                        {"name": "room", "map": "room_Lightmap.exr"},
                        {"name": "prop", "map": "gone.exr"},
                    ]
                ),
                objects=("room", "prop"),
                material=0,
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual([r["object"] for r in records], ["room"])
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        self.assertEqual(self._bound_map(gltf, 0), "room_Lightmap.exr")
        self.assertIsNone(self._bound_map(gltf, 1))

    def test_a_baked_object_whose_map_is_UNREADABLE_is_not_lit_by_its_neighbour(
        self,
    ):
        """The same for a map that is found but cannot be read (a truncated
        EXR from an interrupted bake): it counted as lit once found, so its
        failed encode left the material to the neighbour's in-place bind."""
        self._exr("room_Lightmap.exr")
        with open(os.path.join(self.tmp, "prop_Lightmap.exr"), "wb") as f:
            f.write(b"not an exr")
        glb = self._glb(
            self._scene(
                self._manifest(
                    [
                        {"name": "prop", "map": "prop_Lightmap.exr"},
                        {"name": "room", "map": "room_Lightmap.exr"},
                    ]
                ),
                objects=("room", "prop"),
                material=0,
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual([r["object"] for r in records], ["room"])
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        self.assertEqual(self._bound_map(gltf, 0), "room_Lightmap.exr")
        self.assertIsNone(self._bound_map(gltf, 1), "prop wears the room's map")

    def test_baked_objects_on_one_map_share_one_view_beside_an_unbaked_one(self):
        """Three objects baked into one atlas beside an unbaked object on the
        same material: the atlas rides once, each baked object reads it, the
        unbaked one reads nothing, and no material is copied."""
        exr = self._exr("atlas.exr")
        glb = self._glb(
            self._scene(
                self._manifest(
                    [{"name": n, "map": os.path.basename(exr)} for n in ("a", "b", "c")]
                ),
                objects=("a", "b", "c", "prop"),
                material=0,
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual([r["object"] for r in records], ["a", "b", "c"])
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        self.assertEqual(len(gltf["materials"]), 2, "roomMat, propMat: no copy")
        self.assertEqual(len(gltf["bufferViews"]), 1, "one atlas view")
        self.assertEqual([self._bound_map(gltf, i) for i in range(3)], ["atlas.exr"] * 3)
        self.assertIsNone(self._bound_map(gltf, 3), "the unbaked prop stays unlit")

    def test_per_instance_rects_ride_the_record_as_written(self):
        """Instances (one shared glTF mesh, many nodes -- what FBX2glTF emits)
        each read their own patch of one atlas: the rect rides the record as
        the bake wrote it (``scaleOffset``, the file's own V-up UVs) and the
        reader applies it -- the file copies no mesh or material and declares
        no ``KHR_texture_transform``, and the atlas rides once."""
        self._exr("atlas.exr")
        rect_a, rect_b = [0.5, 1.0, 0.0, 0.0], [0.5, 1.0, 0.5, 0.0]
        gltf = self._scene(
            self._manifest(
                [
                    {"name": "wall_a", "map": "atlas.exr", "scaleOffset": rect_a},
                    {"name": "wall_b", "map": "atlas.exr", "scaleOffset": rect_b},
                ]
            ),
            objects=("wall_a", "wall_b"),
        )
        gltf["meshes"] = gltf["meshes"][:1]  # collapse to ONE shared mesh
        for node in gltf["nodes"]:
            if "mesh" in node:
                node["mesh"] = 0
        glb = self._glb(gltf)

        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(
            {r["object"]: tuple(r["scaleOffset"]) for r in records},
            {"wall_a": tuple(rect_a), "wall_b": tuple(rect_b)},
        )
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        self.assertNotIn("extensionsUsed", gltf)
        self.assertEqual((len(gltf["meshes"]), len(gltf["materials"])), (1, 2))
        self.assertEqual(len(gltf["bufferViews"]), 1, "one shared atlas view")
        lit = MeshConvert.lit_nodes(gltf)
        self.assertEqual(
            {gltf["nodes"][i]["name"]: e["scaleOffset"] for i, e in lit.items()},
            {"wall_a": rect_a, "wall_b": rect_b},
        )

    def test_an_authored_occlusion_map_stays_as_authored(self):
        """No material slot is written: an authored occlusion map -- a real AO,
        or the packed-ORM binding -- stays exactly where it was, unwarned, and
        the lightmap rides beside it. The binder's slot parameters
        (``carrier``, ``replace_authored``) are gone in 0.21.0."""
        exr = self._exr()
        gltf = self._scene(
            self._manifest(
                [{"name": "room", "map": os.path.basename(exr)}, {"name": "prop", "map": os.path.basename(exr)}]
            ),
            objects=("room", "prop"),
        )
        gltf["materials"][0].update(
            {
                "pbrMetallicRoughness": {"metallicRoughnessTexture": {"index": 0}},
                "occlusionTexture": {"index": 0},
            }
        )
        gltf["textures"] = [{"source": 0}]
        gltf["images"] = [{"name": "orm.png"}]
        glb = self._glb(gltf)
        with self.assertNoLogs("pythontk.file_utils.mesh_convert", level="WARNING"):
            records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(len(records), 2)
        with MeshConvert.open_glb(glb) as edit:
            material = edit.gltf["materials"][0]
            self.assertEqual(material["occlusionTexture"], {"index": 0})
            self.assertEqual(edit.gltf["images"], [{"name": "orm.png"}])
        for retired in ({"carrier": "emissive"}, {"replace_authored": False}):
            with self.subTest(retired=retired), self.assertRaises(TypeError):
                MeshConvert.apply_glb_lightmaps(glb, **retired)

    def test_stale_locate_hint_binds_through_search_dirs(self):
        """A moved texture folder must not silently ship an unlit deliverable.

        The manifest's ``dir`` is recorded when the bake is COMMITTED, so it is
        history rather than a contract: measured on a delivered room whose maps
        moved from ``production/maya/sourceimages`` to ``production/sourceimages``,
        every EXR lookup missed and the GLB shipped unlit while the bake sat one
        folder away. The host knows where its textures are now, and *search_dirs*
        is how it says so.
        """
        moved = os.path.join(self.tmp, "moved")
        os.makedirs(moved)
        exr = self._exr()
        os.rename(exr, os.path.join(moved, os.path.basename(exr)))
        manifest = self._manifest([{"name": "room", "map": os.path.basename(exr)}])
        manifest["dir"] = os.path.join(self.tmp, "gone")  # the stale hint

        self.assertEqual(
            MeshConvert.apply_glb_lightmaps(self._glb(self._scene(manifest))),
            [],
            "precondition: the stale hint alone finds nothing",
        )
        records = MeshConvert.apply_glb_lightmaps(
            self._glb(self._scene(manifest), name="fixed.glb"), search_dirs=[moved]
        )
        self.assertEqual(len(records), 1, "search_dirs must rescue the stale hint")

    def test_unfindable_map_warns_once_and_reports_the_total(self):
        """The failure has to be legible in an export log, not 48 lines of noise.

        Every object baked into one atlas shares its basename, so warning per
        ENTRY turned one moved folder into 48 identical lines -- which is how
        this got lost in a real export log and shipped a lightmap-less
        deliverable. One line per missing MAP, plus one summary naming the
        total, because "never baked" (a silent no-op) and "baked and none of it
        reached the file" are the same empty list and wildly different
        deliverables.
        """
        manifest = self._manifest(
            [{"name": f"room{i}", "map": "absent_Lightmap.exr"} for i in range(1, 4)]
        )
        glb = self._glb(self._scene(manifest, objects=("room1", "room2", "room3")))
        with self.assertLogs(
            "pythontk.file_utils.mesh_convert", level="WARNING"
        ) as caught:
            self.assertEqual(MeshConvert.apply_glb_lightmaps(glb), [])
        not_found = [m for m in caught.output if "not found in" in m]
        totals = [m for m in caught.output if "Lightmaps NOT wired" in m]
        self.assertEqual(len(not_found), 1, f"one per MAP, got {caught.output}")
        self.assertEqual(len(totals), 1, f"one summary, got {caught.output}")
        self.assertIn("3 object(s)", totals[0])
        # The count is the point of collapsing the lines: one warning that says
        # nothing about how much is at stake is not better than 48 that do.
        self.assertIn("3 object(s)", not_found[0])
        self.assertNotIn(
            "search_dirs",
            not_found[0],
            "every shipped caller already passes them; advising it here is "
            "wrong by the time this fires",
        )

    def test_identity_rects_change_nothing_structural(self):
        """An explicit identity scaleOffset rides as written: no mesh or
        material copied, no extension declared."""
        self._exr("atlas.exr")
        glb = self._glb(
            self._scene(
                self._manifest(
                    [
                        {
                            "name": "room",
                            "map": "atlas.exr",
                            "scaleOffset": [1.0, 1.0, 0.0, 0.0],
                        }
                    ]
                )
            )
        )
        records = MeshConvert.apply_glb_lightmaps(glb)
        self.assertEqual(records[0]["scaleOffset"], [1.0, 1.0, 0.0, 0.0])
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
        self.assertNotIn("extensionsUsed", gltf)
        self.assertEqual(len(gltf["materials"]), 2, "no clones")
        self.assertEqual(len(gltf["meshes"]), 1)


class _FakeKtx2Encoder:
    """Records encode calls and writes a KTX2-magic marker file."""

    MAGIC = b"\xabKTX 20\xbb\r\n\x1a\n"

    def __init__(self):
        self.calls = []

    def encode(
        self,
        source,
        output,
        codec="UASTC",
        srgb=True,
        mipmaps=True,
        quality=None,
        uastc_rdo=None,
        uastc_rdo_dictionary=None,
    ):
        self.calls.append(
            {
                "codec": codec,
                "srgb": srgb,
                "quality": quality,
                "size": getattr(source, "size", None),
                "image": source,
                "rdo": uastc_rdo,
                "rdo_dictionary": uastc_rdo_dictionary,
            }
        )
        with open(output, "wb") as fh:
            fh.write(self.MAGIC + codec.encode("ascii"))
        return output


class TestWebDeliveryDials(unittest.TestCase):
    """The secondary ceiling and the UASTC RDO lambda are policy keys too:
    unspecified takes the policy (off), a value is the caller's decision, and
    ``0`` on the lambda is off rather than a lambda. Added: 2026-09-13"""

    def test_defaults_and_overrides(self):
        policy = MeshConvert.web_delivery_texture_params()
        self.assertEqual(policy["secondary_max_size"], 0)
        self.assertIsNone(policy["uastc_rdo"])
        chosen = MeshConvert.web_delivery_texture_params(
            secondary_max_size=2048, uastc_rdo=1.0
        )
        self.assertEqual(
            (chosen["secondary_max_size"], chosen["uastc_rdo"]), (2048, 1.0)
        )
        self.assertIsNone(
            MeshConvert.web_delivery_texture_params(uastc_rdo=0)["uastc_rdo"]
        )
        self.assertIsNone(MeshConvert._uastc_rdo_for("data", None))
        self.assertEqual(MeshConvert._uastc_rdo_for("data", 2.0), 2.0)
        self.assertEqual(MeshConvert._uastc_rdo_for("normal", 2.0), 0.75)
        self.assertEqual(MeshConvert._uastc_rdo_for("normal", 0.5), 0.5)
        # One cap, owned by the codec: the map optimizer applies the same one.
        self.assertEqual(
            MeshConvert.UASTC_RDO_NORMAL_MAX, Ktx2Encoder.UASTC_RDO_NORMAL_MAX
        )

    def test_the_rdo_dictionary_is_web_delivery_policy(self):
        """A web export runs RDO on a smaller dictionary than toktx's 4096: the
        window is where RDO's time goes (a production set's own maps through
        this pass: 245 s at 4096, 129 s at 1024, +2.2% bytes), and the
        maintainer's call is that those bytes do not justify the time on an
        export. Unspecified takes the policy, a size is the caller's, ``0`` is
        toktx's own. Added: 2026-09-19"""
        policy = MeshConvert.web_delivery_texture_params()
        self.assertEqual(policy["uastc_rdo_dictionary"], 1024)
        self.assertEqual(
            MeshConvert.WEB_DELIVERY_UASTC_RDO_DICTIONARY,
            policy["uastc_rdo_dictionary"],
        )
        chosen = MeshConvert.web_delivery_texture_params(uastc_rdo_dictionary=256)
        self.assertEqual(chosen["uastc_rdo_dictionary"], 256)
        toktx_own = MeshConvert.web_delivery_texture_params(uastc_rdo_dictionary=0)
        self.assertIsNone(toktx_own["uastc_rdo_dictionary"])

    def test_the_pass_description_names_the_dials(self):
        summary = {"images": 3, "bytes_before": 3e6, "bytes_after": 1e6, "resized": 1}
        text = MeshConvert.describe_texture_pass(
            summary, "KTX2", 4096, secondary_max_size=2048, uastc_rdo=1.0
        )
        self.assertIn("data maps capped at 2048px", text)
        self.assertIn("UASTC RDO lambda 1", text)
        self.assertNotIn("dictionary", text)
        self.assertNotIn(
            "capped", MeshConvert.describe_texture_pass(summary, "KTX2", 4096)
        )
        sized = MeshConvert.describe_texture_pass(
            summary, "KTX2", 4096, uastc_rdo=1.0, uastc_rdo_dictionary=1024
        )
        self.assertIn("UASTC RDO lambda 1 (dictionary 1024)", sized)
        # A dictionary with no RDO pass means nothing, so it is not claimed.
        self.assertNotIn(
            "dictionary",
            MeshConvert.describe_texture_pass(
                summary, "KTX2", 4096, uastc_rdo_dictionary=1024
            ),
        )


class TestWebDeliveryTexturePolicy(unittest.TestCase):
    """ONE definition of "finished for the web", read by every producer.

    The preview and the scene exporters both hand `optimize_glb_textures` its
    kwargs, and they disagreed: the preview named a container and inherited the
    method's own `max_size` default, while an exporter whose panel dials were
    untouched ran no pass at all. Measured on one production assembly through
    both legs: 8.71 MB of WebP against 280.13 MB of raw PNG, from the same
    scene, in the same session. An artist approves the first and hands over the
    second.
    """

    def test_the_policy_names_a_container_and_a_ceiling(self):
        params = MeshConvert.web_delivery_texture_params()
        self.assertEqual(params["image_format"], MeshConvert.WEB_DELIVERY_FORMAT)
        self.assertEqual(params["max_size"], MeshConvert.WEB_DELIVERY_MAX_SIZE)

    def test_a_caller_may_override_either_half(self):
        params = MeshConvert.web_delivery_texture_params(
            image_format="KTX2", max_size=1024
        )
        self.assertEqual(
            params,
            {
                "image_format": "KTX2",
                "max_size": 1024,
                "ktx2_fallback": False,
                "secondary_max_size": 0,
                "uastc_rdo": None,
                "uastc_rdo_dictionary": MeshConvert.WEB_DELIVERY_UASTC_RDO_DICTIONARY,
            },
        )

    def test_ktx2_ships_without_core_readable_twins(self):
        """A web deliverable carries no PNG/JPEG copy beside each KTX2 image.

        The twin exists for stock importers, and a web deliverable's consumer is
        a basisu-capable viewer. Measured on the production 4K assembly: 145.8 MB
        of fallbacks beside 123.1 MB of KTX2 -- 48% of a 302.6 MB GLB spent on a
        second copy of every map nobody loads.
        """
        self.assertIs(MeshConvert.web_delivery_texture_params()["ktx2_fallback"], False)

    def test_a_caller_may_still_ask_for_the_twins(self):
        """The third part overrides like the other two: an exporter whose GLB
        must also open in Blender or Unreal passes True, and None stays the
        policy."""
        asked = MeshConvert.web_delivery_texture_params(ktx2_fallback=True)
        unspecified = MeshConvert.web_delivery_texture_params(ktx2_fallback=None)
        self.assertIs(asked["ktx2_fallback"], True)
        self.assertIs(
            unspecified["ktx2_fallback"], MeshConvert.WEB_DELIVERY_KTX2_FALLBACK
        )

    def test_an_unspecified_half_falls_back_to_the_policy_not_to_nothing(self):
        """`None` means "unspecified" on both halves. The trap this pins: an
        empty container reaching `optimize_glb_textures` as an explicit None
        raises there, and a caller's own falsy default must not become one."""
        self.assertEqual(
            MeshConvert.web_delivery_texture_params(image_format="", max_size=None),
            {
                "image_format": MeshConvert.WEB_DELIVERY_FORMAT,
                "max_size": MeshConvert.WEB_DELIVERY_MAX_SIZE,
                "ktx2_fallback": MeshConvert.WEB_DELIVERY_KTX2_FALLBACK,
                "secondary_max_size": MeshConvert.WEB_DELIVERY_SECONDARY_MAX_SIZE,
                "uastc_rdo": MeshConvert.WEB_DELIVERY_UASTC_RDO,
                "uastc_rdo_dictionary": MeshConvert.WEB_DELIVERY_UASTC_RDO_DICTIONARY,
            },
        )

    def test_zero_is_an_explicit_refusal_to_resample(self):
        """Distinct from None: a caller that means "keep every pixel" can say
        so, and only the *absence* of a choice takes the ceiling."""
        self.assertEqual(
            MeshConvert.web_delivery_texture_params(max_size=0)["max_size"], 0
        )

    def test_a_container_is_named_by_its_file_extension_or_its_format_id(self):
        """The Scene Exporter's Texture File Type row speaks file EXTENSIONS
        (``jpg``), while ``optimize_glb_textures`` hands the name to Pillow and
        builds the glTF mime from it -- ``JPG`` raises in Pillow and would be an
        invalid mime if it did not. Both exporters carried a private copy of
        that translation; the policy now owns it, so every producer that asks
        for a container (the WebXR preview included) gets the same answer."""
        for asked, expected in (
            ("jpg", "JPEG"),
            ("jpeg", "JPEG"),
            ("JPEG", "JPEG"),
            (".PNG", "PNG"),
            ("webp", "WEBP"),
            ("WEBP", "WEBP"),
            ("ktx2", "KTX2"),
        ):
            with self.subTest(asked=asked):
                self.assertEqual(
                    MeshConvert.web_delivery_texture_params(image_format=asked)[
                        "image_format"
                    ],
                    expected,
                )

    def test_the_containers_a_glb_can_carry(self):
        """glTF core's own (``IMAGE_MIME_TYPES``) plus the two the texture pass
        declares an extension for. A scene-side container (TGA, EXR) is not one:
        a GLB given one takes the policy's container instead."""
        self.assertEqual(
            MeshConvert.GLB_IMAGE_FORMATS,
            frozenset({"png", "jpg", "jpeg", "webp", "ktx2"}),
        )
        for ext in MeshConvert.IMAGE_MIME_TYPES:
            self.assertIn(ext.lstrip("."), MeshConvert.GLB_IMAGE_FORMATS)

    def test_the_policy_is_what_the_optimizer_itself_defaults_to(self):
        """Belt and braces on the seam that made this necessary: the preview
        used to inherit `optimize_glb_textures`' own default, so a change there
        silently changed the deliverable. They must agree while both exist."""
        import inspect

        signature = inspect.signature(MeshConvert.optimize_glb_textures)
        for name, value in MeshConvert.web_delivery_texture_params().items():
            with self.subTest(name=name):
                self.assertEqual(signature.parameters[name].default, value)

    def test_the_retired_worker_cap_reads_the_shared_encode_policy(self):
        """``OPTIMIZE_WORKERS`` stays one release as an alias of the cap that
        replaced it, and follows that cap rather than a copy of its value."""
        self.assertEqual(MeshConvert.OPTIMIZE_WORKERS, ImgUtils.ENCODE_WORKERS)
        with patch.object(ImgUtils, "ENCODE_WORKERS", 3):
            self.assertEqual(MeshConvert.OPTIMIZE_WORKERS, 3)


class TestOptimizeGlbKtx2(unittest.TestCase):
    """KTX2 mode: per-slot codecs, basisu bindings, POT snap, lightmap carve-out."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_ktx2_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.fake = _FakeKtx2Encoder()
        ImgUtils.register_ktx2_encoder(self.fake)
        self.addCleanup(ImgUtils.register_ktx2_encoder, None)

    # ------------------------------------------------------------------ helpers
    def _png_uri(self, size=(128, 128), color=(200, 60, 40)):
        import base64 as b64

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", size, color).save(buffer, format="PNG")
        return "data:image/png;base64," + b64.b64encode(buffer.getvalue()).decode(
            "ascii"
        )

    def _glb(self, gltf, name="scene.glb"):
        json_bytes = json.dumps(gltf).encode("utf-8")
        json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
        blob = (
            struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_bytes))
            + struct.pack("<I4s", len(json_bytes), b"JSON")
            + json_bytes
        )
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(blob)
        return path

    def _payload(self, edit, image):
        view = edit.gltf["bufferViews"][image["bufferView"]]
        return bytes(
            edit.bin_data[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
        )

    def _pbr_glb(self, size=(256, 256)):
        """Color + normal + metallic-roughness on one material, each its own image."""
        return self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [
                    {"name": "color", "uri": self._png_uri(size, (200, 60, 40))},
                    {"name": "normal", "uri": self._png_uri(size, (127, 127, 255))},
                    {"name": "orm", "uri": self._png_uri(size, (255, 128, 0))},
                ],
                "textures": [{"source": 0}, {"source": 1}, {"source": 2}],
                "materials": [
                    {
                        "pbrMetallicRoughness": {
                            "baseColorTexture": {"index": 0},
                            "metallicRoughnessTexture": {"index": 2},
                        },
                        "normalTexture": {"index": 1},
                    }
                ],
            }
        )

    # -------------------------------------------------------------------- tests
    def test_the_secondary_ceiling_resizes_the_data_maps_only(self):
        """A lower ceiling for the packed data maps: the ORM takes it, color and
        normal keep the primary -- and the primary still bounds it.
        Added: 2026-09-13"""
        MeshConvert.optimize_glb_textures(
            self._pbr_glb(), max_size=256, image_format="KTX2", secondary_max_size=128
        )
        self.assertEqual(
            sorted((c["size"], c["codec"]) for c in self.fake.calls),
            [((128, 128), "UASTC"), ((256, 256), "ETC1S"), ((256, 256), "UASTC")],
        )
        self.fake.calls.clear()
        MeshConvert.optimize_glb_textures(
            self._pbr_glb(), max_size=128, image_format="KTX2", secondary_max_size=512
        )
        self.assertEqual({c["size"] for c in self.fake.calls}, {(128, 128)})

    def test_uastc_rdo_reaches_the_data_maps_and_is_capped_for_normals(self):
        """The dial's lambda rides every UASTC encode -- a normal map capped at
        UASTC_RDO_NORMAL_MAX (toktx's own guidance) -- and never an ETC1S one.
        Off, the keyword is not even passed. Added: 2026-09-13"""
        MeshConvert.optimize_glb_textures(
            self._pbr_glb(), max_size=256, image_format="KTX2", uastc_rdo=2.0
        )
        self.assertEqual(
            sorted({(c["codec"], c["srgb"], c["rdo"]) for c in self.fake.calls}),
            [("ETC1S", True, None), ("UASTC", False, 0.75), ("UASTC", False, 2.0)],
        )
        self.fake.calls.clear()
        MeshConvert.optimize_glb_textures(
            self._pbr_glb(), max_size=256, image_format="KTX2"
        )
        self.assertEqual({c["rdo"] for c in self.fake.calls}, {None})

    def test_the_rdo_dictionary_rides_only_an_rdo_encode(self):
        """The web-delivery dictionary reaches every UASTC encode that runs RDO,
        and nothing else: not ETC1S (no RDO stage), not an encode with RDO off
        (toktx rejects the flag there), and ``None`` leaves toktx to its own.
        Added: 2026-09-19"""
        MeshConvert.optimize_glb_textures(
            self._pbr_glb(), max_size=256, image_format="KTX2", uastc_rdo=1.0
        )
        self.assertEqual(
            sorted({(c["codec"], c["rdo_dictionary"]) for c in self.fake.calls}),
            [("ETC1S", None), ("UASTC", MeshConvert.WEB_DELIVERY_UASTC_RDO_DICTIONARY)],
        )
        for kwargs in ({}, {"uastc_rdo": 1.0, "uastc_rdo_dictionary": None}):
            with self.subTest(**kwargs):
                self.fake.calls.clear()
                MeshConvert.optimize_glb_textures(
                    self._pbr_glb(), max_size=256, image_format="KTX2", **kwargs
                )
                self.assertEqual({c["rdo_dictionary"] for c in self.fake.calls}, {None})

    def test_a_bad_rdo_dial_is_refused_before_any_encode(self):
        """A typo in either dial is the caller's error, raised up front. Caught
        inside the per-image encode it read as one failed encode per image,
        each keeping its original PNG -- a GLB that ships raw maps under a
        pass that "ran". Added: 2026-09-19"""
        for kwargs in (
            {"uastc_rdo": 11},
            {"uastc_rdo": 1.0, "uastc_rdo_dictionary": 63},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    MeshConvert.optimize_glb_textures(
                        self._pbr_glb(), max_size=256, image_format="KTX2", **kwargs
                    )
                self.assertEqual(self.fake.calls, [])

    def test_per_slot_codecs_and_fallback_binding(self):
        """Color -> ETC1S/sRGB, normal -> UASTC/linear; with fallbacks asked
        for, each texture keeps a core-readable ``source`` (JPEG for color, PNG
        for normal), so KHR_texture_basisu stays out of extensionsRequired and
        a stock importer reads the fallbacks."""
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [
                    {"name": "wall_color", "uri": self._png_uri(color=(200, 60, 40))},
                    {
                        "name": "wall_normal",
                        "uri": self._png_uri(color=(127, 127, 255)),
                    },
                ],
                "textures": [{"source": 0}, {"source": 1}],
                "materials": [
                    {
                        "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                        "normalTexture": {"index": 1},
                    }
                ],
            }
        )
        summary = MeshConvert.optimize_glb_textures(
            path, max_size=64, image_format="KTX2", quality=60, ktx2_fallback=True
        )
        self.assertEqual(summary["images"], 2)

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        by_codec = {c["codec"]: c for c in self.fake.calls}
        self.assertTrue(by_codec["ETC1S"]["srgb"])
        self.assertEqual(by_codec["ETC1S"]["quality"], 60)
        self.assertFalse(by_codec["UASTC"]["srgb"])
        self.assertIsNone(by_codec["UASTC"]["quality"], "quality is the ETC1S dial")

        self.assertEqual(len(gltf["images"]), 4, "one appended fallback per encode")
        fallback_magic = {"image/jpeg": b"\xff\xd8", "image/png": b"\x89PN"}
        for index, codec, fb_mime in (
            (0, b"ETC1S", "image/jpeg"),
            (1, b"UASTC", "image/png"),
        ):
            image = gltf["images"][index]
            self.assertEqual(image["mimeType"], "image/ktx2")
            payload = self._payload(edit, image)
            self.assertEqual(payload[:12], _FakeKtx2Encoder.MAGIC)
            self.assertEqual(payload[12:], codec)
            texture = gltf["textures"][index]
            self.assertEqual(
                texture["extensions"]["KHR_texture_basisu"]["source"], index
            )
            fb_index = texture["source"]
            self.assertGreaterEqual(fb_index, 2, "fallback is an appended image")
            fallback = gltf["images"][fb_index]
            self.assertEqual(fallback["mimeType"], fb_mime)
            self.assertEqual(
                self._payload(edit, fallback)[: len(fallback_magic[fb_mime])],
                fallback_magic[fb_mime],
            )
            self.assertEqual(fallback["name"], f"{image['name']}_fallback")
        self.assertIn("KHR_texture_basisu", gltf["extensionsUsed"])
        self.assertNotIn(
            "KHR_texture_basisu",
            gltf.get("extensionsRequired", []),
            "every binding has a core-readable fallback",
        )

    def test_pure_delivery_is_the_default_and_requires_basisu(self):
        """No ``ktx2_fallback`` argument = the web-delivery policy: no fallback
        source, and the extension hard-requires a basisu-capable viewer
        (extensionsRequired)."""
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "c", "uri": self._png_uri()}],
                "textures": [{"source": 0}],
                "materials": [
                    {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path, max_size=64, image_format="KTX2")
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(len(gltf["images"]), 1, "no fallback appended")
        texture = gltf["textures"][0]
        self.assertEqual(texture["extensions"]["KHR_texture_basisu"]["source"], 0)
        self.assertNotIn("source", texture, "no fallback is kept")
        self.assertIn("KHR_texture_basisu", gltf["extensionsRequired"])

    def test_fallback_alpha_color_stays_png(self):
        """A color image with alpha cannot fall back to JPEG — the twin must
        keep the channel, so ETC1S + alpha lands on PNG."""
        import base64 as b64

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGBA", (32, 32), (200, 60, 40, 128)).save(buffer, format="PNG")
        uri = "data:image/png;base64," + b64.b64encode(buffer.getvalue()).decode(
            "ascii"
        )
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "decal", "uri": uri}],
                "textures": [{"source": 0}],
                "materials": [
                    {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                ],
            }
        )
        MeshConvert.optimize_glb_textures(
            path, max_size=0, image_format="KTX2", ktx2_fallback=True
        )
        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(self.fake.calls[0]["codec"], "ETC1S")
        fallback = gltf["images"][gltf["textures"][0]["source"]]
        self.assertEqual(fallback["mimeType"], "image/png")

    def test_shared_image_takes_the_stricter_semantic(self):
        """Bytes sampled as both color and data must encode once, as UASTC."""
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "shared", "uri": self._png_uri()}],
                "textures": [{"source": 0}],
                "materials": [
                    {
                        "pbrMetallicRoughness": {
                            "baseColorTexture": {"index": 0},
                            "metallicRoughnessTexture": {"index": 0},
                        }
                    }
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path, max_size=64, image_format="KTX2")
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(self.fake.calls[0]["codec"], "UASTC")
        self.assertFalse(self.fake.calls[0]["srgb"])

    def test_dimensions_snap_down_to_pot(self):
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "odd", "uri": self._png_uri(size=(96, 48))}],
                "textures": [{"source": 0}],
                "materials": [
                    {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path, max_size=0, image_format="KTX2")
        self.assertEqual(self.fake.calls[0]["size"], (64, 32))

    def test_a_normal_map_is_resampled_as_one(self):
        """The pass resamples a normal-slot image in float and renormalises it
        (``ImgUtils.resize_image(normal_map=True)``), in every container: a
        plain 8-bit Lanczos clips its overshoot at a crease and bends those
        normals (measured on a production 4K atlas halved: 0.07% of texels
        off by more than 3 degrees, the worst 56)."""
        import base64

        from conftest import hard_edge_normal_map, unit_length_error

        buffer = io.BytesIO()
        hard_edge_normal_map((64, 64)).save(buffer, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
            "ascii"
        )

        def glb(name):
            return self._glb(
                {
                    "asset": {"version": "2.0"},
                    "images": [{"name": "normal", "uri": uri}],
                    "textures": [{"source": 0}],
                    "materials": [{"normalTexture": {"index": 0}}],
                },
                name=name,
            )

        MeshConvert.optimize_glb_textures(
            glb("k.glb"), max_size=32, image_format="KTX2"
        )
        (call,) = self.fake.calls
        self.assertEqual(call["size"], (32, 32))
        self.assertLess(unit_length_error(call["image"]), 0.01)

        from PIL import Image

        path = glb("w.glb")
        MeshConvert.optimize_glb_textures(path, max_size=27, image_format="WEBP")
        edit = MeshConvert._read_glb(path)
        with Image.open(io.BytesIO(self._payload(edit, edit.gltf["images"][0]))) as im:
            self.assertEqual(im.size, (27, 27))
            self.assertLess(unit_length_error(im), 0.01)

    def _ktx2_lightmap_glb(self, lightmap_size=(128, 128)):
        """A KTX2-mode scene with one ordinary map and one exempt lightmap --
        found, as every lightmap is, by its texCoord-1 occlusion binding."""
        return self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [
                    {"name": "source.png", "uri": self._png_uri(color=(10, 200, 30))},
                    {
                        "name": "room_Lightmap.png",
                        "uri": self._png_uri(size=lightmap_size),
                    },
                ],
                "textures": [{"source": 0}, {"source": 1}],
                "materials": [
                    {
                        "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                        "occlusionTexture": {"index": 1, "texCoord": 1},
                    }
                ],
            }
        )

    def _lightmap_call(self, size):
        """The one encode the fake saw at *size* -- the lightmap's own edge."""
        calls = [call for call in self.fake.calls if call["size"] == size]
        self.assertEqual(len(calls), 1, self.fake.calls)
        return calls[0]

    def test_a_pot_lightmap_takes_uastc_srgb_at_its_own_size(self):
        """KTX2 mode is the GPU-memory mode, and a lightmap atlas is one of the
        larger resident images: a 2048 atlas is 22.4 MB as RGBA8 + mips, 5.6 MB
        as UASTC. So it takes the Basis codec too -- UASTC, never ETC1S, and
        never RDO (measured on production atlases: ETC1S 2-5x the error of the
        lossless container, RDO 1.0 doubled the dark-quartile error), and still
        exempt from the size ceiling the bake chose its resolution against.
        """
        path = self._ktx2_lightmap_glb()
        MeshConvert.optimize_glb_textures(
            path, max_size=64, image_format="KTX2", uastc_rdo=1.0
        )

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        lightmap = gltf["images"][1]
        self.assertEqual(lightmap["mimeType"], "image/ktx2")
        self.assertEqual(
            self._payload(edit, lightmap)[: len(self.fake.MAGIC)], self.fake.MAGIC
        )
        self.assertEqual(
            gltf["textures"][1]["extensions"]["KHR_texture_basisu"]["source"], 1
        )
        self.assertNotIn("source", gltf["textures"][1])
        self.assertNotIn("EXT_texture_webp", gltf.get("extensionsUsed", []))
        self.assertIn("KHR_texture_basisu", gltf["extensionsRequired"])
        call = self._lightmap_call((128, 128))
        self.assertEqual(call["codec"], "UASTC")
        self.assertTrue(call["srgb"])
        self.assertIsNone(call["rdo"], "a lightmap never takes RDO")
        self.assertIsNone(call["quality"])

    def _media_lightmap_glb(self, size=(128, 128)):
        """A KTX2-mode scene whose lightmap rides as media, as the lightmap
        pass writes it (``SceneRecords.GLB_MEDIA``): one ordinary image, and
        the map's PNG bytes as the bufferView named by its file, with its
        scale."""
        import base64

        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "source.png", "uri": self._png_uri(color=(10, 200, 30))}],
                "textures": [{"source": 0}],
                "materials": [
                    {"name": "m", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                ],
            }
        )
        png = base64.b64decode(self._png_uri(size=size).split(",", 1)[1])
        with MeshConvert.open_glb(path) as edit:
            (index,) = MeshConvert._append_bin_views(edit, [png])
            edit.gltf["bufferViews"][index].update(
                {
                    "name": "room_Lightmap.exr",
                    "extras": SceneRecords.GLB_MEDIA.extras("image/png", 2.5),
                }
            )
            edit.dirty = True
        return path

    def _media_view(self, path, name="room_Lightmap.exr"):
        edit = MeshConvert._read_glb(path)
        view = edit.gltf["bufferViews"][SceneRecords.GLB_MEDIA.view(edit.gltf, name)]
        start = int(view.get("byteOffset", 0))
        return edit.gltf, view, bytes(edit.bin_data[start : start + view["byteLength"]])

    def test_a_pot_lightmap_riding_as_media_takes_uastc_srgb_at_its_own_size(self):
        """A lightmap the record names rides as media, not as an image, and
        KTX2 mode gives it the encode an image lightmap takes -- UASTC + sRGB at
        its own size, no RDO -- its view keeping the file's name and its
        scale, the PNG's collected: the pixels the viewer reads are the ones a
        KTX2 delivery always gave it. Added: 2026-11-07"""
        path = self._media_lightmap_glb()
        MeshConvert.optimize_glb_textures(
            path, max_size=64, image_format="KTX2", uastc_rdo=1.0
        )
        gltf, view, payload = self._media_view(path)
        self.assertEqual(view["extras"], SceneRecords.GLB_MEDIA.extras("image/ktx2", 2.5))
        self.assertEqual(payload[: len(self.fake.MAGIC)], self.fake.MAGIC)
        call = self._lightmap_call((128, 128))
        self.assertEqual(call["codec"], "UASTC")
        self.assertTrue(call["srgb"])
        self.assertIsNone(call["rdo"], "a lightmap never takes RDO")
        self.assertIsNone(call["quality"])
        named = [v for v in gltf["bufferViews"] if v.get("name")]
        self.assertEqual(len(named), 1, "the PNG's view went")

    def test_a_non_pot_lightmap_riding_as_media_stays_lossless(self):
        """Never resampled, so a map off a power of two keeps a lossless
        container, as an image lightmap does: lossless WebP in pure delivery,
        its PNG in the "opens in any reader" mode. Added: 2026-11-07"""
        for fallback, mime, magic in ((False, "image/webp", b"RIFF"), (True, "image/png", b"\x89PNG")):
            with self.subTest(ktx2_fallback=fallback):
                path = self._media_lightmap_glb(size=(96, 64))
                MeshConvert.optimize_glb_textures(
                    path, max_size=0, image_format="KTX2", ktx2_fallback=fallback
                )
                _gltf, view, payload = self._media_view(path)
                self.assertEqual(view["extras"], SceneRecords.GLB_MEDIA.extras(mime, 2.5))
                self.assertTrue(payload.startswith(magic))

    def _media_pixels(self, payload):
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as image:
            return image.format, list(image.convert("RGB").getdata())

    def test_a_webp_delivery_shrinks_its_lightmaps_losslessly(self):
        """A WebP delivery re-encodes each lightmap riding as media as
        lossless WebP, the pixels unchanged and its scale kept, as it
        re-encoded an image lightmap before the maps rode as media; a reader
        decodes it as it decodes the PNG. Added: 2026-11-07"""
        path = self._media_lightmap_glb(size=(96, 64))
        _gltf, _view, png = self._media_view(path)
        report = MeshConvert.optimize_glb_textures(path, max_size=0, image_format="WEBP")
        self.assertEqual(report["media"], 1)
        gltf, view, payload = self._media_view(path)
        self.assertEqual(view["extras"], SceneRecords.GLB_MEDIA.extras("image/webp", 2.5))
        self.assertLess(len(payload), len(png))
        fmt, pixels = self._media_pixels(payload)
        self.assertEqual(fmt, "WEBP")
        self.assertEqual(pixels, self._media_pixels(png)[1])
        self.assertEqual(len([v for v in gltf["bufferViews"] if v.get("name")]), 1)

    def test_a_media_re_encode_is_reported_when_no_image_changes(self):
        """The report counts the media it re-encoded even when every image
        was left as found. Added: 2026-11-07"""
        import base64

        path = self._media_lightmap_glb(size=(96, 64))
        MeshConvert.optimize_glb_textures(path, max_size=0, image_format="WEBP")
        with MeshConvert.open_glb(path) as edit:  # a second map, riding as PNG
            png = base64.b64decode(self._png_uri(size=(32, 16)).split(",", 1)[1])
            (index,) = MeshConvert._append_bin_views(edit, [png])
            edit.gltf["bufferViews"][index].update(
                {"name": "atlas_Lightmap.exr", "extras": SceneRecords.GLB_MEDIA.extras("image/png", 1.5)}
            )
            edit.dirty = True
        report = MeshConvert.optimize_glb_textures(path, max_size=0, image_format="WEBP")
        self.assertEqual(report.get("images"), 0)
        self.assertEqual(report.get("media"), 1)

    def test_a_pot_lightmap_keeps_a_core_readable_twin_when_fallbacks_are_on(self):
        """The "opens in any reader" mode gives the lightmap the same PNG twin
        every other UASTC image gets, bound as the plain ``source``, so a stock
        importer still reads it and basisu stays merely *used*.
        """
        path = self._ktx2_lightmap_glb()
        MeshConvert.optimize_glb_textures(
            path, max_size=64, image_format="KTX2", ktx2_fallback=True
        )

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        self.assertEqual(gltf["images"][1]["mimeType"], "image/ktx2")
        texture = gltf["textures"][1]
        self.assertEqual(texture["extensions"]["KHR_texture_basisu"]["source"], 1)
        twin = gltf["images"][texture["source"]]
        self.assertEqual(twin["mimeType"], "image/png")
        self.assertEqual(self._payload(edit, twin)[:4], b"\x89PNG")
        self.assertNotIn("KHR_texture_basisu", gltf.get("extensionsRequired", []))
        self.assertNotIn("EXT_texture_webp", gltf.get("extensionsUsed", []))

    def test_a_non_pot_lightmap_takes_lossless_webp_in_pure_delivery_mode(self):
        """A lightmap is never resampled, and KHR_texture_basisu wants POT
        edges, so an off-size atlas keeps the smallest lossless container --
        and the file SAYS it needs it.

        Lossy WebP is YUV 4:2:0, which blotches near-black lightmap texels, so
        the encode stays lossless; nothing here is core-readable any more,
        which ``extensionsRequired`` must declare.
        """
        path = self._ktx2_lightmap_glb(lightmap_size=(96, 48))
        MeshConvert.optimize_glb_textures(
            path, max_size=64, image_format="KTX2", ktx2_fallback=False
        )

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        lightmap = gltf["images"][1]
        self.assertEqual(lightmap["mimeType"], "image/webp")
        self.assertEqual(self._payload(edit, lightmap)[:4], b"RIFF")
        self.assertEqual(
            gltf["textures"][1]["extensions"]["EXT_texture_webp"]["source"], 1
        )
        self.assertNotIn(
            "source",
            gltf["textures"][1],
            "no core-readable twin exists, so none may be claimed",
        )
        for ext in ("EXT_texture_webp", "KHR_texture_basisu"):
            self.assertIn(ext, gltf["extensionsUsed"])
            self.assertIn(ext, gltf["extensionsRequired"])
        # The encoder saw only the source image -- never the lightmap.
        self.assertEqual(len(self.fake.calls), 1)

    def test_a_non_pot_lightmap_stays_core_readable_when_fallbacks_are_on(self):
        """An off-size lightmap needs a container of its own -- and in the
        "opens in any reader" mode that container must be one glTF core can
        read.

        This previously shipped the lightmap as WebP and bound it through
        EXT_texture_webp with the plain ``source`` pointing at the SAME image,
        calling that a fallback. glTF 2.0 core permits image/jpeg and image/png
        only, so with the extension merely *used* the file advertised itself as
        readable by anyone while a core reader resolved the texture to WebP.
        Measured on a delivered room: both lightmap textures shipped that way.

        Carrying a real PNG twin alongside the WebP would cost MORE than the
        PNG alone (103 KB + 209 KB vs 209 KB), so ``ktx2_fallback=True`` keeps
        the lightmap on the core path and declares no extension for it at all.
        """
        path = self._ktx2_lightmap_glb(lightmap_size=(96, 48))
        MeshConvert.optimize_glb_textures(
            path, max_size=64, image_format="KTX2", ktx2_fallback=True
        )

        gltf = MeshConvert._read_glb(path).gltf
        self.assertEqual(gltf["images"][0]["mimeType"], "image/ktx2")
        self.assertNotEqual(
            gltf["images"][1].get("mimeType"),
            "image/webp",
            "a fallback-mode lightmap must stay core-readable",
        )
        self.assertNotIn("extensions", gltf["textures"][1])
        self.assertIsNotNone(gltf["textures"][1].get("source"))
        self.assertNotIn("EXT_texture_webp", gltf.get("extensionsUsed", []))
        self.assertNotIn("EXT_texture_webp", gltf.get("extensionsRequired", []))
        # The whole point of the fallback mode: a stock importer opens it.
        self.assertNotIn("KHR_texture_basisu", gltf.get("extensionsRequired", []))
        # The encoder saw only the source image -- never the lightmap.
        self.assertEqual(len(self.fake.calls), 1)

    def test_lightmap_basis_none_keeps_every_lightmap_lossless(self):
        """``LIGHTMAP_BASIS = None`` is the policy's off switch: a POT lightmap
        then ships exactly as an off-size one does, and the encoder never sees
        it."""
        path = self._ktx2_lightmap_glb()
        with patch.object(MeshConvert, "LIGHTMAP_BASIS", None):
            MeshConvert.optimize_glb_textures(
                path, max_size=64, image_format="KTX2", ktx2_fallback=False
            )

        edit = MeshConvert._read_glb(path)
        lightmap = edit.gltf["images"][1]
        self.assertEqual(lightmap["mimeType"], "image/webp")
        self.assertEqual(self._payload(edit, lightmap)[:4], b"RIFF")
        self.assertEqual(len(self.fake.calls), 1)

    def test_webp_then_ktx2_rerun_cleans_the_webp_declaration(self):
        """Switching a deliverable from WebP to KTX2 must not leave a dangling
        EXT_texture_webp binding or declaration behind — a declared extension
        with no user is a validator warning shipped for nothing."""
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "c", "uri": self._png_uri()}],
                "textures": [{"source": 0}],
                "materials": [
                    {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path, max_size=64)  # WebP first
        gltf = MeshConvert._read_glb(path).gltf
        self.assertIn("EXT_texture_webp", gltf["extensionsUsed"])

        MeshConvert.optimize_glb_textures(
            path, max_size=64, image_format="KTX2", ktx2_fallback=True
        )
        gltf = MeshConvert._read_glb(path).gltf
        texture = gltf["textures"][0]
        self.assertNotIn("EXT_texture_webp", texture["extensions"])
        self.assertEqual(texture["extensions"]["KHR_texture_basisu"]["source"], 0)
        self.assertNotIn("EXT_texture_webp", gltf["extensionsUsed"])
        self.assertIn("KHR_texture_basisu", gltf["extensionsUsed"])
        self.assertNotIn("KHR_texture_basisu", gltf.get("extensionsRequired", []))
        self.assertEqual(gltf["images"][0]["mimeType"], "image/ktx2")
        # The WebP bytes the first pass wrote became the decode source for the
        # KTX2 rerun; the plain source now points at the appended core fallback.
        self.assertEqual(gltf["images"][texture["source"]]["mimeType"], "image/jpeg")

    def test_webp_then_ktx2_rerun_drops_the_stranded_webp_requirement(self):
        """The WebP pass REQUIRES its extension (nothing core-readable survives
        it). A KTX2 rerun re-encodes past those bindings, so the requirement
        must go with them: glTF 2.0 makes ``extensionsRequired`` a subset of
        ``extensionsUsed``, and a rerun that clears only the latter ships a file
        demanding a capability it no longer declares -- invalid, not merely
        untidy, and every stock validator rejects it."""
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "c", "uri": self._png_uri()}],
                "textures": [{"source": 0}],
                "materials": [
                    {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path, max_size=64)  # WebP first
        gltf = MeshConvert._read_glb(path).gltf
        # Precondition: the WebP pass has no core-readable fallback to offer, so
        # it escalates to extensionsRequired. Without this the rerun has nothing
        # to strand and the test would pass vacuously.
        self.assertIn("EXT_texture_webp", gltf.get("extensionsRequired", []))

        MeshConvert.optimize_glb_textures(path, max_size=64, image_format="KTX2")
        gltf = MeshConvert._read_glb(path).gltf
        self.assertNotIn("EXT_texture_webp", gltf.get("extensionsRequired", []))
        self.assertLessEqual(
            set(gltf.get("extensionsRequired", [])),
            set(gltf.get("extensionsUsed", [])),
            "extensionsRequired must stay a subset of extensionsUsed",
        )

    def test_ktx2_rerun_keeps_requiring_basisu_when_bindings_have_no_fallback(self):
        """The sweep must not overreach: a pure-delivery KTX2 pass leaves every
        binding without a core-readable ``source``, so the requirement it
        declared is still load-bearing on a rerun."""
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "c", "uri": self._png_uri()}],
                "textures": [{"source": 0}],
                "materials": [
                    {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                ],
            }
        )
        for _ in range(2):
            MeshConvert.optimize_glb_textures(
                path, max_size=64, image_format="KTX2", ktx2_fallback=False
            )
        gltf = MeshConvert._read_glb(path).gltf
        self.assertNotIn("source", gltf["textures"][0])
        self.assertIn("KHR_texture_basisu", gltf.get("extensionsRequired", []))
        self.assertLessEqual(
            set(gltf.get("extensionsRequired", [])),
            set(gltf.get("extensionsUsed", [])),
        )

    def test_unsampled_image_never_gets_a_non_core_mime_type(self):
        """An image no texture samples cannot be rebound through a container
        extension, so re-encoding it strands a non-core mime in a file with
        nothing to enable it — and glTF 2.0 core permits only image/jpeg and
        image/png. Both ways a texture fails to resolve: a ``source`` out of
        range, and no textures at all.

        Both non-core containers, because the invariant is about the CONTAINER
        and not about basisu: ``EXT_texture_webp`` is equally a texture-level
        extension, so a WebP-encoded orphan is exactly as unreachable as a KTX2
        one. Only the KTX2 path was gated, and a WebP delivery re-encoded every
        orphan it found.
        """
        for container, mime in (("KTX2", "image/ktx2"), ("WEBP", "image/webp")):
            for label, extra in (
                ("out-of-range source", {"textures": [{"source": 7}]}),
                ("no textures", {}),
            ):
                with self.subTest(container=container, case=label):
                    path = self._glb(
                        {
                            "asset": {"version": "2.0"},
                            "images": [{"name": "orphan", "uri": self._png_uri()}],
                            **extra,
                        },
                        name=f"{container}_{label.replace(' ', '_')}.glb",
                    )
                    with open(path, "rb") as f:
                        before = f.read()
                    MeshConvert.optimize_glb_textures(
                        path, max_size=64, image_format=container
                    )
                    gltf = MeshConvert._read_glb(path).gltf
                    self.assertNotEqual(
                        gltf["images"][0].get("mimeType"),
                        mime,
                        "non-core mime with no extension declaring it",
                    )
                    for name in MeshConvert.TEXTURE_CONTAINER_EXTENSIONS:
                        self.assertNotIn(name, gltf.get("extensionsUsed", []))
                    with open(path, "rb") as f:
                        self.assertEqual(
                            f.read(), before, "nothing to do, nothing written"
                        )
                    self.assertEqual(self.fake.calls, [], "no encode worth paying for")

    def test_only_a_sampled_image_takes_the_ktx2_path(self):
        """A bound image still re-encodes; its unsampled sibling keeps its
        core-mime bytes instead of riding the declaration the bound one earns.
        """
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [
                    {"name": "bound", "uri": self._png_uri()},
                    {"name": "orphan", "uri": self._png_uri(color=(9, 9, 9))},
                ],
                "textures": [{"source": 0}],
                "materials": [
                    {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                ],
            }
        )
        summary = MeshConvert.optimize_glb_textures(
            path, max_size=64, image_format="KTX2", ktx2_fallback=True
        )
        self.assertEqual(summary["images"], 1)
        self.assertEqual(len(self.fake.calls), 1)

        edit = MeshConvert._read_glb(path)
        gltf = edit.gltf
        self.assertEqual(gltf["images"][0]["mimeType"], "image/ktx2")
        self.assertEqual(
            self._payload(edit, gltf["images"][0])[:12], _FakeKtx2Encoder.MAGIC
        )
        self.assertIn("KHR_texture_basisu", gltf["extensionsUsed"])
        self.assertNotIn("KHR_texture_basisu", gltf.get("extensionsRequired", []))
        orphan = gltf["images"][1]
        self.assertNotEqual(orphan.get("mimeType"), "image/ktx2")
        self.assertTrue(orphan["uri"].startswith("data:image/png"))

    def test_missing_encoder_fails_before_touching_the_file(self):
        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "c", "uri": self._png_uri()}],
                "textures": [{"source": 0}],
            }
        )
        with open(path, "rb") as f:
            before = f.read()
        ImgUtils.register_ktx2_encoder(None)
        with (
            patch.object(Ktx2Encoder, "available", return_value=False),
            patch.object(
                Ktx2Encoder,
                "resolve_toktx",
                side_effect=FileNotFoundError("no toktx"),
            ),
        ):
            with self.assertRaises(FileNotFoundError):
                MeshConvert.optimize_glb_textures(path, image_format="KTX2")
        with open(path, "rb") as f:
            after = f.read()
        self.assertEqual(after, before, "no partial rewrite")


class TestTheGlbCarriesTheAudioRecord(unittest.TestCase):
    """``extras.audio`` -- the FBX's ``audio`` record, carried into the GLB as
    it stands (one layout, two containers; 2026-11-05): its cues per clip,
    joined to the animations by name. The WebXR viewer plays no audio; the
    record rides for any reader of the delivery."""

    RECORD = {
        "version": 3,
        "cues": [
            {"clip": "Intro", "sound": "door.wav", "time": 0.5, "offset": 0.0, "stop": 1.3},
            {"clip": "Outro", "sound": "door.wav", "time": 0.0, "offset": 0.8, "stop": 0.4},
        ],
        "media": {"door.wav": "ab" * 32},
    }

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_audio_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _glb(self, record, nested=True):
        props = {SceneRecords.AUDIO.key: json.dumps(record)}
        if nested:
            props = {k: {"type": "eFbxString", "value": v} for k, v in props.items()}
            extras = {"fromFBX": {"userProperties": props}}
        else:
            extras = props
        return _write_glb_file(
            os.path.join(self.tmp, "audio.glb"),
            {
                "asset": {"version": "2.0"},
                "nodes": [{"name": "box", "mesh": 0}, {"name": "data_export", "extras": extras}],
            },
        )

    def _root(self, path):
        with MeshConvert.open_glb(path) as edit:
            return copy.deepcopy(edit.gltf.get("extras") or {}), [
                n.get("name") for n in edit.gltf.get("nodes") or []
            ]

    def test_the_record_rides_as_it_stands_and_the_carrier_goes(self):
        """As it stands, less ``media``: the digests are of the sound files
        the FBX embeds (``AudioRecord.media``), and a GLB embeds none."""
        carried = {k: v for k, v in self.RECORD.items() if k != "media"}
        for nested in (True, False):
            with self.subTest(nested=nested):
                path = self._glb(self.RECORD, nested=nested)
                MeshConvert.finish_glb(path)
                extras, nodes = self._root(path)
                self.assertEqual(extras.get(SceneRecords.AUDIO.key), carried)
                self.assertNotIn("media", extras[SceneRecords.AUDIO.key])
                self.assertNotIn("data_export", nodes)
                self.assertEqual(
                    SceneRecords.problems(SceneRecords.AUDIO, extras["audio"]), []
                )

    def test_a_record_of_another_version_is_not_carried(self):
        """Version 2 (frames at a rate) is refused, warned: re-export."""
        v2 = {"version": 2, "fps": 30.0, "events": [{"clip": "", "frame": 4, "name": "door"}]}
        path = self._glb(v2)
        with self.assertLogs("pythontk", level="WARNING"):
            MeshConvert.finish_glb(path)
        extras, _nodes = self._root(path)
        self.assertNotIn(SceneRecords.AUDIO.key, extras)


class TestApplyGlbAnimations(unittest.TestCase):
    """``extras.animation`` -- the FBX's ``animation`` record, carried into
    the GLB as it stands (one layout, two containers; 2026-11-05).

    The fixtures are the converted shape of a per-shot FBX: one take per
    shot, named after it, each converted to one glTF animation whose sampler
    times are rebased to zero (SHOT_B, authored at frames 20-30, starts at
    t=0 exactly like SHOT_A at 1-10). The ``animation`` record (version 3)
    states per shot only what a clip cannot hold -- its description and its
    metadata -- and the scene's frame rate, which glTF cannot hold; a reader
    joins each shot to the animation of its clip's NAME.
    """

    FPS = 30.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_anim_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ helpers
    def _animation(self, name, first_frame, last_frame, accessors):
        """One animation whose sampler input spans the take's frames, at t=0."""
        span = (last_frame - first_frame) / self.FPS
        accessors.append({"type": "SCALAR", "min": [0.0], "max": [span]})
        return {
            "name": name,
            "samplers": [{"input": len(accessors) - 1, "output": 99}],
            "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
        }

    def _glb(self, animations, channels=None, nested=True, name="anim.glb"):
        """A GLB with *animations* and the ``data_export`` carrier *channels*.

        *nested* picks the carrier shape: FBX2glTF's
        ``extras.fromFBX.userProperties`` wrapping (the Maya route) or a native
        exporter's top-level node extras.
        """
        accessors = []
        anims = [self._animation(*a, accessors) for a in animations]
        nodes = [{"name": "animCube", "mesh": 0}]
        if channels is not None:
            props = {k: json.dumps(v) for k, v in channels.items()}
            if nested:
                props = {
                    k: {"type": "eFbxString", "value": v} for k, v in props.items()
                }
                extras = {"fromFBX": {"userProperties": props}}
            else:
                extras = props
            nodes.append({"name": "data_export", "extras": extras})
        return _write_glb_file(
            os.path.join(self.tmp, name),
            {
                "asset": {"version": "2.0"},
                "nodes": nodes,
                "accessors": accessors,
                "animations": anims,
            },
        )

    #: The record a per-shot file carries: each shot's clip, and what a clip
    #: cannot hold where the shot has it.
    SHOTS = [
        {
            "clip": "SHOT_A",
            "description": "first",
            "metadata": {"section": "A", "behaviors": ["door"]},
        },
        {"clip": "SHOT_B"},
    ]

    def _shot_shape(self, **kw):
        """The per-shot deliverable: one take per shot, each its own clip."""
        return self._glb(
            [("SHOT_A", 1, 10), ("SHOT_B", 20, 30)],
            channels={
                "animation": {
                    "version": SceneRecords.SHOTS.version,
                    "fps": self.FPS,
                    "shots": copy.deepcopy(self.SHOTS),
                },
            },
            **kw,
        )

    def _extras(self, path):
        with MeshConvert.open_glb(path) as edit:
            return (edit.gltf.get("extras") or {}).get(SceneRecords.SHOTS.key)

    # ------------------------------------------------------------------- tests
    def test_the_glb_carries_the_record_itself(self):
        """The record the FBX carries, verbatim, under its key in the root
        ``extras`` -- its version and shape the record's, with nothing bound
        to the file's animation indices (no ``clips``, ``default_clip`` or
        ``zero_frame``): a reader joins each shot to the animation of its
        clip's name. Fixed: 2026-11-05 (a v3 block of clips by index)."""
        path = self._shot_shape()
        with MeshConvert.open_glb(path) as edit:
            carried = copy.deepcopy(
                MeshConvert.data_export_channel(edit.gltf, SceneRecords.SHOTS.key)
            )
        manifest = MeshConvert.apply_glb_animations(path)

        self.assertEqual(manifest, carried)
        self.assertEqual(manifest["version"], SceneRecords.SHOTS.version)
        self.assertEqual(manifest["fps"], self.FPS)
        self.assertEqual(set(manifest), {"version", "fps", "shots"})
        with MeshConvert.open_glb(path) as edit:
            names = [a["name"] for a in edit.gltf["animations"]]
        self.assertEqual([s["clip"] for s in manifest["shots"]], names)
        self.assertEqual(SceneRecords.problems(SceneRecords.SHOTS, manifest), [])

    def test_a_clip_named_after_its_shot_in_another_case_takes_the_shots_name(self):
        """A shot's clip is unique ignoring case and joins its take by name, so
        an animation whose name differs from it in case alone (a converter or
        a hand respelling it) is the shot's: it is renamed to the record's
        ``clip`` exactly, said, and every reader joins it by name. Added:
        2026-11-05."""
        path = self._shot_shape()
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["animations"][1]["name"] = "shot_b"
            edit.dirty = True
        with self.assertLogs("pythontk", level="INFO") as caught:
            MeshConvert.apply_glb_animations(path)
        with MeshConvert.open_glb(path) as edit:
            names = [a["name"] for a in edit.gltf["animations"]]
        self.assertEqual(names, ["SHOT_A", "SHOT_B"])
        self.assertTrue(any("shot_b" in m and "SHOT_B" in m for m in caught.output))

    def test_the_shots_ride_as_the_record_states_them(self):
        """Description and metadata, whole; the frames are the clips'."""
        manifest = MeshConvert.apply_glb_animations(self._shot_shape())
        self.assertEqual(manifest["shots"], self.SHOTS)
        for shot in manifest["shots"]:
            self.assertFalse({"start", "end", "objects"} & set(shot), shot)

    def test_both_carrier_shapes_are_read(self):
        """Nested (FBX2glTF) and top-level (native export) node extras alike."""
        manifest = MeshConvert.apply_glb_animations(self._shot_shape(nested=False))
        self.assertEqual([s["clip"] for s in manifest["shots"]], ["SHOT_A", "SHOT_B"])

    def test_a_file_with_no_record_carries_none(self):
        """No shots declared, no record: the one whole-timeline clip plays by
        its own name, and nothing is made up for it."""
        path = self._glb([("Take 001", 1, 30)])
        self.assertIsNone(MeshConvert.apply_glb_animations(path))
        self.assertIsNone(self._extras(path))

    def test_a_glb_with_no_animation_is_a_clean_no_op(self):
        """Which is what makes it safe to run after every conversion."""
        path = _write_glb_file(
            os.path.join(self.tmp, "static.glb"),
            {"asset": {"version": "2.0"}, "nodes": [{"name": "cube"}]},
        )
        before = open(path, "rb").read()

        self.assertIsNone(MeshConvert.apply_glb_animations(path))
        self.assertIsNone(self._extras(path))
        self.assertEqual(open(path, "rb").read(), before, "the file was rewritten")

    def test_a_declared_shot_with_no_clip_is_warned(self):
        """A shot the file carries no take of -- the split never reached it."""
        path = self._glb(
            [("Take 001", 1, 30)],
            channels={"animation": _shot_record(["SHOT_A"])},
        )
        with self.assertLogs(
            "pythontk.file_utils.mesh_convert", level="WARNING"
        ) as caught:
            manifest = MeshConvert.apply_glb_animations(path)

        self.assertTrue(any("SHOT_A" in m for m in caught.output), caught.output)
        # Still carried: the record is the scene's, whatever the file holds.
        self.assertEqual([s["clip"] for s in manifest["shots"]], ["SHOT_A"])

    def test_a_version_1_record_is_refused_with_its_fix(self):
        """A file written before one take per shot states its shots as windows
        of a whole-timeline take ('start', 'end', 'fps', 'clip_mode'): read
        as v2 it would mean something else, so it is refused, warned, and the
        cure named -- re-export. The clips still stand."""
        path = self._glb(
            [("Take 001", 1, 30), ("SHOT_A", 1, 10)],
            channels={
                "animation": {
                    "version": 1,
                    "fps": self.FPS,
                    "clip_mode": "both",
                    "shots": [{"clip": "SHOT_A", "start": 1, "end": 10}],
                },
            },
        )
        with self.assertLogs("pythontk", level="WARNING") as caught:
            manifest = MeshConvert.apply_glb_animations(path)

        self.assertTrue(
            any("animation record not applied" in m and "re-export" in m for m in caught.output),
            caught.output,
        )
        # Not carried: a GLB states no record its reader would refuse.
        self.assertIsNone(manifest)
        self.assertIsNone(self._extras(path))

    def _with_empty_shot(self):
        """The measured production shape: a declared shot that carries NOTHING.

        A take split emits an AnimStack per declared range, but bakes no
        curve for a range in which nothing actually moves (a hold, or a shot
        whose motion belongs to objects outside the export). Those clips reach
        the GLB named, listed, and empty -- measured on a 12-shot production
        assembly where Shot_1, Shot_6 and Shot_11 came through with zero
        channels.
        """
        accessors = []
        anims = [
            {"name": "Shot_1", "samplers": [], "channels": []},
            self._animation("Shot_2", 20, 40, accessors),
        ]
        props = {
            k: {"type": "eFbxString", "value": json.dumps(v)}
            for k, v in {"animation": _shot_record(["Shot_1", "Shot_2"], self.FPS)}.items()
        }
        return _write_glb_file(
            os.path.join(self.tmp, "empty_shot.glb"),
            {
                "asset": {"version": "2.0"},
                "nodes": [
                    {"name": "animCube", "mesh": 0},
                    {
                        "name": "data_export",
                        "extras": {"fromFBX": {"userProperties": props}},
                    },
                ],
                "accessors": accessors,
                "animations": anims,
            },
        )

    def test_an_empty_clip_is_listed_and_said(self):
        """A shot that holds still is a real clip in the sequence: carried in
        its place, its emptiness the glTF animation's own (the record states
        no ``empty``), and the reviewer told."""
        with self.assertLogs("pythontk", level="INFO") as caught:
            manifest = MeshConvert.apply_glb_animations(self._with_empty_shot())

        self.assertEqual([s["clip"] for s in manifest["shots"]], ["Shot_1", "Shot_2"])
        self.assertNotIn("empty", manifest["shots"][0])
        self.assertTrue(
            any("carry no keyframes" in m and "Shot_1" in m for m in caught.output),
            caught.output,
        )

    def test_the_block_is_written_to_the_file(self):
        """It is the deliverable's own copy, not a return value the caller keeps."""
        path = self._shot_shape()
        returned = MeshConvert.apply_glb_animations(path)
        written = self._extras(path)
        self.assertEqual(written["version"], SceneRecords.SHOTS.version)
        self.assertEqual(written, returned)

    def test_a_glb_carrying_the_retired_v2_block_is_refused_plainly(self):
        """HEAD's published GLB block rode ``extras.animation`` at version 2;
        the record rides it at 3. ``verify_glb`` refuses the old block
        plainly -- naming its version and the one read -- rather than reading
        it half-way, and no pass reads it. Added: 2026-11-05."""
        path = self._shot_shape()
        with MeshConvert.open_glb(path) as edit:
            edit.gltf.setdefault("extras", {})[SceneRecords.SHOTS.key] = {
                "version": 2,
                "fps": self.FPS,
                "shots": [{"clip": "SHOT_A", "start": 1, "end": 10}],
                "clips": [{"name": "SHOT_A", "animation": 0, "zero_frame": 1}],
                "default_clip": "SHOT_A",
            }
            edit.gltf["extras"][HandoffRecord.KEY] = HandoffRecord.block(
                "glb", [], {"application": "maya"}
            )
            edit.dirty = True
        with MeshConvert.open_glb(path) as edit:
            self.assertIsNone(SceneRecords.glb_block(edit.gltf, SceneRecords.SHOTS))
        report = MeshConvert.verify_glb(path)
        self.assertNotIn("animation", report)
        self.assertTrue(
            any(
                f"extras.{SceneRecords.SHOTS.key}: v2 is not the v3" in failure
                for failure in report["problems"]
            ),
            report["problems"],
        )

    def test_verify_reports_the_clips_to_a_recipient(self):
        path = self._shot_shape()
        MeshConvert.apply_glb_animations(path)
        report = MeshConvert.verify_glb(path)
        self.assertEqual(
            report["animation"], {"clips": 2, "shots": 2, "declared": 2}
        )

    def test_shots_with_no_animation_are_still_stated(self):
        """A file written with animation off keeps its shots' names and
        notes; it has no clip to play."""
        path = self._glb(
            [],
            channels={
                "animation": {
                    "version": SceneRecords.SHOTS.version,
                    "fps": self.FPS,
                    "shots": copy.deepcopy(self.SHOTS),
                }
            },
        )
        manifest = MeshConvert.apply_glb_animations(path)
        self.assertEqual([s["clip"] for s in manifest["shots"]], ["SHOT_A", "SHOT_B"])
        self.assertEqual(SceneRecords.problems(SceneRecords.SHOTS, manifest), [])

    def test_verify_notes_animation_that_nothing_describes(self):
        report = MeshConvert.verify_glb(self._glb([("Take 001", 1, 30)]))
        self.assertNotIn("animation", report)
        block = f"extras.{SceneRecords.SHOTS.key}"
        self.assertTrue(any(block in note for note in report["notes"]), report["notes"])

    def test_the_handoff_names_the_block(self):
        """A standalone reader has to be told the record is there, in the
        artifact: the handoff of a file carrying it names it -- its
        ``assembly`` row, addressed at the record under its own part, saying
        what the FBX's says of it: the takes, one per shot, each joined by
        name."""
        from pythontk import SceneRecords

        path = self._shot_shape()
        MeshConvert.apply_glb_animations(path)
        with MeshConvert.open_glb(path) as edit:
            MeshConvert.embed_glb_handoff(
                edit, HandoffRecord.block("glb", [], {"application": "maya"})
            )
        handoff = MeshConvert.read_glb_handoff(path)
        block = f"extras.{SceneRecords.SHOTS.key}"
        (row,) = [r for r in handoff["assembly"] if r["data"] == [block]]
        self.assertEqual(row["part"], SceneRecords.SHOTS.key)
        self.assertEqual(row["spec"], SceneRecords.SHOTS.assembly)
        self.assertIn("one per shot", row["spec"])
        for retired in ("'zero_frame'", "'default_clip'", "whole-timeline clip"):
            self.assertNotIn(retired, row["spec"])


class TestGlbRenderEffects(unittest.TestCase):
    """The ``render_effects`` record (v2) over the file's own curves.

    The record names which nodes and materials carry which channels; each
    curve rides the transport (``SceneRecords.TRANSPORT``): a curve proxy per
    take in the FBX, read off the file (``MeshConvert.render_effect_curves``),
    and per clip in the GLB. Measured 2026-11-05 on real Maya 2025 and Blender
    5.1 exports: FBX2glTF 0.13.1 resamples every proxy on its 1/30 s grid
    (steps become 1/30 s ramps, a sub-frame key and a take's last key are
    lost) and drops a take's channel held at the node's static value, so the
    pass never reads the converter's proxy channels. Curves here are injected
    in that reader's shape: ``{clip: {(target, channel): [[t, value,
    interpolation], ...]}}``, seconds from the clip's start.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_effects_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _record(tracks=(), materials=()):
        record = {"version": SceneRecords.VISIBILITY.version, "tracks": list(tracks)}
        if materials:
            record["materials"] = list(materials)
        return record

    def _glb(self, record, clips=(("ShotA", 1.0), ("ShotB", 1.0)), name="fx.glb"):
        """SIGN wears an exclusive mapped emissive material; BOX and OTHER
        share one; SCREEN and PANEL wear two materials named ``screen``."""
        materials = [
            {
                "name": "sign_mat",
                "emissiveFactor": [1.0, 1.0, 1.0],
                "emissiveTexture": {"index": 0},
            },
            {"name": "box_mat", "pbrMetallicRoughness": {"baseColorFactor": [0.2, 0.4, 0.6, 1.0]}},
            {"name": "screen", "emissiveFactor": [0.1, 0.1, 0.1]},
            {"name": "ns:screen", "emissiveFactor": [0.1, 0.1, 0.1]},
        ]
        meshes = [
            {"primitives": [{"attributes": {}, "material": 0}]},
            {"primitives": [{"attributes": {}, "material": 1}]},
            {"primitives": [{"attributes": {}, "material": 1}]},
            {"primitives": [{"attributes": {}, "material": 2}]},
            {"primitives": [{"attributes": {}, "material": 3}]},
        ]
        nodes = [
            {"name": "SIGN", "mesh": 0},
            {"name": "BOX", "mesh": 1},
            {"name": "OTHER", "mesh": 2},
            {"name": "SCREEN", "mesh": 3},
            {"name": "PANEL", "mesh": 4},
        ]
        props = {}
        if record is not None:
            props[SceneRecords.VISIBILITY.key] = record
        nodes.append(
            {
                "name": "data_export",
                "extras": {
                    "fromFBX": {
                        "userProperties": {
                            k: {"type": "eFbxString", "value": json.dumps(v)}
                            for k, v in props.items()
                        }
                    }
                },
            }
        )
        accessors, animations = [], []
        for clip, length in clips:
            accessors.append({"type": "SCALAR", "min": [0.0], "max": [length]})
            animations.append(
                {
                    "name": clip,
                    "samplers": [{"input": len(accessors) - 1, "output": 99}],
                    "channels": [
                        {"sampler": 0, "target": {"node": 4, "path": "translation"}}
                    ],
                }
            )
        gltf = {
            "asset": {"version": "2.0"},
            "nodes": nodes,
            "meshes": meshes,
            "materials": materials,
            "textures": [{"source": 0}],
            "images": [{"uri": "glow.png"}],
            "accessors": accessors,
            "animations": animations,
            "buffers": [{"byteLength": 16}],
        }
        if not clips:
            del gltf["animations"]
        return _write_glb_file(
            os.path.join(self.tmp, name), gltf, bin_chunk=b"\x00" * 16
        )

    @staticmethod
    def _decoded(path, clip):
        """``[(target, path, times, values, interpolation, extras)]`` of one
        clip's channels (the fixture's stub left out); *target* is the node's
        name, or the pointer for a pointer channel."""
        with MeshConvert.open_glb(path) as edit:
            gltf, blob = edit.gltf, edit.bin_data
            views = gltf.get("bufferViews") or []

            def read(index):
                accessor = gltf["accessors"][index]
                per = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[accessor["type"]]
                view = views[accessor["bufferView"]]
                start = view.get("byteOffset", 0)
                raw = bytes(blob[start : start + view["byteLength"]])
                flat = struct.unpack(f"<{len(raw) // 4}f", raw)
                return [list(flat[i : i + per]) for i in range(0, len(flat), per)]

            out = []
            for animation in gltf.get("animations") or []:
                if animation.get("name") != clip:
                    continue
                for channel in animation.get("channels") or []:
                    sampler = animation["samplers"][channel["sampler"]]
                    if sampler["output"] == 99:
                        continue
                    target = channel["target"]
                    if target["path"] == "pointer":
                        label = target["extensions"]["KHR_animation_pointer"]["pointer"]
                    else:
                        label = gltf["nodes"][target["node"]]["name"]
                    out.append(
                        (
                            label,
                            target["path"],
                            [t[0] for t in read(sampler["input"])],
                            read(sampler["output"]),
                            sampler.get("interpolation"),
                            channel.get("extras") or {},
                        )
                    )
            return out

    @staticmethod
    def _gltf(path):
        with MeshConvert.open_glb(path) as edit:
            return copy.deepcopy(edit.gltf)

    # ------------------------------------------------------------------ record
    def test_a_version_1_record_is_refused_and_names_its_fix(self):
        """v1 (ramps in frames at its 'fps') is no shape this reader reads:
        refused whole, the warning naming the cure."""
        path = self._glb(
            {"version": 1, "fps": 30.0, "tracks": [{"node": "BOX", "visibility": [[0, 1], [9, 0]]}]}
        )
        curves = {"ShotA": {("BOX", "visibility"): [[0.0, 1.0, "step"], [0.3, 0.0, "step"]]}}
        with self.assertLogs("pythontk", level="WARNING") as logs:
            self.assertIsNone(MeshConvert.apply_glb_visibility(path, curves=curves))
            self.assertIsNone(MeshConvert.apply_glb_fades(path, curves=curves))
        self.assertTrue(any("re-export" in line for line in logs.output), logs.output)

    def test_the_record_rides_the_glb_as_it_stands(self):
        """One layout, two containers: the GLB's root extras carry the record
        the FBX's carrier holds, unchanged."""
        record = self._record([{"node": "BOX", "channels": ["visibility"]}])
        path = self._glb(record)
        with MeshConvert.open_glb(path) as edit:
            self.assertIn(SceneRecords.VISIBILITY.key, MeshConvert._carry_glb_records(edit.gltf))
            self.assertEqual(edit.gltf["extras"][SceneRecords.VISIBILITY.key], record)

    # -------------------------------------------------------------- visibility
    def test_visibility_steps_from_its_curve_in_each_clip(self):
        path = self._glb(self._record([{"node": "BOX", "channels": ["visibility"]}]))
        curves = {
            "ShotA": {("BOX", "visibility"): [[0.0, 1.0, "step"], [0.4, 0.0, "step"]]},
            "ShotB": {("BOX", "visibility"): [[0.0, 0.0, "step"], [0.25, 1.0, "step"]]},
        }
        self.assertIsNotNone(MeshConvert.apply_glb_visibility(path, curves=curves))
        ((label, kind, times, values, interp, _x),) = self._decoded(path, "ShotA")
        self.assertEqual((label, kind, interp), ("BOX", "scale", "STEP"))
        self.assertEqual([round(t, 5) for t in times], [0.0, 0.4, 1.0])
        self.assertEqual([v[0] for v in values], [1.0, 0.0, 0.0])
        ((_l, _k, times, values, _i, _x),) = self._decoded(path, "ShotB")
        self.assertEqual([round(t, 5) for t in times], [0.0, 0.25, 1.0])
        self.assertEqual([v[0] for v in values], [0.0, 1.0, 1.0])

    def test_a_clip_with_no_curve_for_a_node_costs_no_channel(self):
        path = self._glb(self._record([{"node": "BOX", "channels": ["visibility"]}]))
        curves = {"ShotA": {("BOX", "visibility"): [[0.0, 1.0, "step"], [0.4, 0.0, "step"]]}}
        MeshConvert.apply_glb_visibility(path, curves=curves)
        self.assertEqual(self._decoded(path, "ShotB"), [])

    def test_a_curve_held_visible_through_a_clip_costs_no_channel(self):
        path = self._glb(self._record([{"node": "BOX", "channels": ["visibility"]}]))
        curves = {"ShotA": {("BOX", "visibility"): [[0.0, 1.0, "step"], [1.0, 1.0, "step"]]}}
        self.assertIsNone(MeshConvert.apply_glb_visibility(path, curves=curves))

    def test_a_fade_hides_its_node_only_where_its_alpha_is_zero(self):
        """Opacity at 0 hides the renderer: the node is absent before a fade-in
        starts and after a fade-out ends, and present across both ramps."""
        path = self._glb(self._record([{"node": "BOX", "channels": ["opacity"]}]))
        curves = {
            "ShotA": {
                ("BOX", "opacity"): [
                    [0.2, 0.0, "linear"],
                    [0.5, 1.0, "linear"],
                    [0.7, 1.0, "linear"],
                    [0.9, 0.0, "linear"],
                ]
            }
        }
        MeshConvert.apply_glb_visibility(path, curves=curves)
        ((_l, kind, times, values, interp, _x),) = self._decoded(path, "ShotA")
        self.assertEqual((kind, interp), ("scale", "STEP"))
        self.assertEqual(
            [(round(t, 5), v[0]) for t, v in zip(times, values)],
            [(0.0, 0.0), (0.2, 1.0), (0.9, 0.0), (1.0, 0.0)],
        )

    # ---------------------------------------------------------------- opacity
    def test_a_fade_rides_the_alpha_lane_and_leaves_the_material_opaque(self):
        """Opaque at rest: the GLB keeps the material's authored alpha mode,
        and a reader blends it only while its alpha is below 1."""
        path = self._glb(self._record([{"node": "BOX", "channels": ["opacity"]}]))
        curves = {"ShotA": {("BOX", "opacity"): [[0.0, 1.0, "linear"], [0.5, 0.0, "linear"]]}}
        summary = MeshConvert.apply_glb_fades(path, curves=curves)
        self.assertEqual(summary["by_channel"], {"opacity": 1})
        gltf = self._gltf(path)
        for material in gltf["materials"]:
            self.assertNotEqual(material.get("alphaMode"), "BLEND", material["name"])
        ((pointer, kind, times, values, interp, _x),) = self._decoded(path, "ShotA")
        self.assertEqual(kind, "pointer")
        self.assertRegex(pointer, r"^/materials/\d+/pbrMetallicRoughness/baseColorFactor$")
        self.assertEqual(interp, "LINEAR")
        self.assertEqual([round(t, 5) for t in times], [0.0, 0.5])
        self.assertEqual([[round(c, 5) for c in v] for v in values], [[0.2, 0.4, 0.6, 1.0], [0.2, 0.4, 0.6, 0.0]])
        self.assertIn(EXTENSION_POINTER, gltf["extensionsUsed"])
        self.assertNotIn(EXTENSION_POINTER, gltf.get("extensionsRequired") or [])

    def test_a_shared_material_is_cloned_and_an_exclusive_one_animated_in_place(self):
        """BOX shares box_mat with OTHER, so BOX fades a copy; SIGN alone wears
        sign_mat, so its highlight animates it where it stands."""
        path = self._glb(
            self._record(
                [
                    {"node": "BOX", "channels": ["opacity"]},
                    {"node": "SIGN", "channels": ["highlight"]},
                ]
            )
        )
        curves = {
            "ShotA": {
                ("BOX", "opacity"): [[0.0, 1.0, "linear"], [0.5, 0.0, "linear"]],
                ("SIGN", "highlight"): [[0.0, 0.0, "linear"], [0.5, 1.0, "linear"]],
            }
        }
        MeshConvert.apply_glb_fades(path, curves=curves)
        gltf = self._gltf(path)
        self.assertEqual(len(gltf["materials"]), 5, "one copy, for BOX alone")
        self.assertEqual(gltf["meshes"][0]["primitives"][0]["material"], 0)
        self.assertEqual(gltf["meshes"][2]["primitives"][0]["material"], 1, "OTHER keeps the original")
        copied = gltf["meshes"][1]["primitives"][0]["material"]
        self.assertEqual(gltf["materials"][copied]["name"], "box_mat")
        pointers = {c[0] for c in self._decoded(path, "ShotA")}
        self.assertEqual(
            pointers,
            {
                f"/materials/{copied}/pbrMetallicRoughness/baseColorFactor",
                "/materials/0/emissiveFactor",
            },
        )

    def test_a_held_full_alpha_writes_nothing(self):
        path = self._glb(self._record([{"node": "BOX", "channels": ["opacity"]}]))
        curves = {"ShotA": {("BOX", "opacity"): [[0.0, 1.0, "linear"], [1.0, 1.0, "linear"]]}}
        self.assertIsNone(MeshConvert.apply_glb_fades(path, curves=curves))
        self.assertEqual(len(self._gltf(path)["materials"]), 4, "no copy for no channel")

    def test_a_stepped_segment_holds_until_its_next_key(self):
        """A pointer sampler is LINEAR, so a step is stated as a hold just
        before the key it cuts to -- the curve's own shape, not a ramp."""
        path = self._glb(self._record([{"node": "BOX", "channels": ["opacity"]}]))
        curves = {"ShotA": {("BOX", "opacity"): [[0.0, 0.5, "step"], [0.5, 1.0, "linear"], [0.8, 0.0, "linear"]]}}
        MeshConvert.apply_glb_fades(path, curves=curves)
        ((_p, _k, times, values, _i, _x),) = self._decoded(path, "ShotA")
        alpha = [(round(t, 4), round(v[3], 4)) for t, v in zip(times, values)]
        self.assertEqual(alpha[:2], [(0.0, 0.5), (round(0.5 - MeshConvert.EFFECT_STEP_SECONDS, 4), 0.5)])
        self.assertEqual(alpha[2:], [(0.5, 1.0), (0.8, 0.0)])

    # -------------------------------------------------------------- highlight
    def test_a_highlight_keeps_the_emissive_map_and_states_when_it_glows(self):
        """The map stays in the GLB; while the highlight is above 0 its glow
        is the highlight (the mapped emission set aside), at rest the factor
        is the material's own -- and the channel states the spans it glows
        over, which a reader sets the map aside for."""
        path = self._glb(
            self._record(
                [
                    {
                        "node": "SIGN",
                        "channels": ["highlight"],
                        "highlight_color": [0.0, 0.5, 1.0],
                    }
                ]
            )
        )
        curves = {
            "ShotA": {
                ("SIGN", "highlight"): [
                    [0.0, 0.0, "linear"],
                    [0.2, 0.0, "linear"],
                    [0.6, 1.0, "linear"],
                    [0.8, 0.0, "linear"],
                ]
            }
        }
        MeshConvert.apply_glb_fades(path, curves=curves)
        gltf = self._gltf(path)
        self.assertEqual(gltf["materials"][0]["emissiveTexture"], {"index": 0})
        self.assertEqual(gltf["materials"][0]["emissiveFactor"], [1.0, 1.0, 1.0])
        ((pointer, _k, times, values, _i, extras),) = self._decoded(path, "ShotA")
        self.assertEqual(pointer, "/materials/0/emissiveFactor")
        step = MeshConvert.EFFECT_STEP_SECONDS
        samples = {round(t, 4): [round(c, 4) for c in v] for t, v in zip(times, values)}
        self.assertEqual(samples[0.0], [1.0, 1.0, 1.0], "at rest: the material's own")
        self.assertEqual(samples[0.2], [1.0, 1.0, 1.0])
        self.assertEqual(samples[0.6], [0.0, 0.5, 1.0], "glowing: the highlight alone")
        self.assertEqual(samples[0.8], [1.0, 1.0, 1.0])
        edge = next(v for t, v in zip(times, values) if abs(t - (0.2 + step)) < 1e-6)
        for got, want in zip(edge, [0.0, 0.5 * step / 0.4, step / 0.4]):
            self.assertAlmostEqual(got, want, places=6)
        # The map is set aside from the edge key on, where the glow starts --
        # never at the ramp's first key, whose factor is still the material's.
        self.assertEqual(
            extras["render_effects"],
            {
                "channel": "highlight",
                "node": "SIGN",
                "active": [[0.0, 0], [round(0.2 + step, 6), 1], [0.8, 0]],
            },
        )
        self.assertAgrees(times, values, extras["render_effects"]["active"], [1.0, 1.0, 1.0])

    def assertAgrees(self, times, values, active, base):
        """At every key time the channel's two statements agree: the map is
        set aside (``active`` on) exactly where the factor is a glow, and back
        (off) exactly where the factor is the material's own -- what a page
        posing that instant (``selectClip``, ``seek``) draws."""
        for time, value in zip(times, values):
            on = 0
            for start, state in active:
                if start <= time + 1e-7:
                    on = state
            rest = [round(c, 6) for c in value] == [round(c, 6) for c in base]
            self.assertEqual(bool(on), not rest, f"t={time}: active {on}, factor {value}")

    def test_a_highlight_over_a_map_sets_it_aside_exactly_while_it_glows(self):
        """``extras.render_effects.active`` and the factor agree at every key,
        whatever the ramp: rising from a held 0, from the clip's t=0 (a
        shot-start pulse, posed by a clip switch), falling, stepped, and a
        rise shorter than the edge key's room. Measured before: the span
        started at a ramp's first key, where the factor is still white, so
        the page dropped the map and drew the bare factor for that instant."""
        step = MeshConvert.EFFECT_STEP_SECONDS
        ramps = {
            "rise_after_hold": [[0.0, 0.0, "linear"], [0.5, 0.0, "linear"], [1.0, 1.0, "linear"]],
            "rise_from_zero": [[0.0, 0.0, "linear"], [1.0, 1.0, "linear"]],
            "pulse": [[0.0, 0.0, "linear"], [0.4, 1.0, "linear"], [0.8, 0.0, "linear"]],
            "fall": [[0.0, 1.0, "linear"], [0.6, 0.0, "linear"]],
            "stepped": [[0.0, 0.0, "step"], [0.5, 1.0, "step"], [0.9, 0.0, "step"]],
            "short_rise": [[0.0, 0.0, "linear"], [step * 1.5, 1.0, "linear"]],
        }
        for label, ramp in ramps.items():
            with self.subTest(ramp=label):
                track = {"node": "SIGN", "channels": ["highlight"], "highlight_color": [0.1, 0.4, 1.0]}
                path = self._glb(self._record([track]), name=f"hl_{label}.glb")
                MeshConvert.apply_glb_fades(path, curves={"ShotA": {("SIGN", "highlight"): ramp}})
                ((_p, _k, times, values, _i, extras),) = self._decoded(path, "ShotA")
                active = extras["render_effects"]["active"]
                self.assertAgrees(times, values, active, [1.0, 1.0, 1.0])
                self.assertEqual(active[0][0], 0.0, "stated from the clip's t=0")

    def test_a_channels_values_compose_with_no_material_as_a_preview_calls_them(self):
        """A panel's preview swatch is handed a channel's own ``values`` (so it
        cannot drift from what ships) and calls it ``(base, sample, stops)``:
        no material, which reads as one with no map. Found 2026-11-07 in a
        launched Maya: the map rule made the material a required fourth
        argument, and every paint of the Render Effects panel's swatches
        raised ``missing 1 required positional argument``."""
        stops = ((0.2, 0.5, 1.0), (0.0, 0.0, 0.0))
        highlight = GlbFades.CHANNELS["highlight"].values
        self.assertEqual(
            [round(c, 4) for c in highlight([0.1, 0.1, 0.1], 0.5, stops)], [0.2, 0.35, 0.6]
        )
        opacity = GlbFades.CHANNELS["opacity"].values
        self.assertEqual(opacity([0.2, 0.4, 0.6, 1.0], 0.25, ()), [0.2, 0.4, 0.6, 0.25])
        emission = GlbFades.CHANNELS["emission"].values
        self.assertEqual(emission([0.0, 0.0, 0.0], 0.5, ((1.0, 0.5, 0.0),)), [0.5, 0.25, 0.0])

    def test_a_highlight_over_a_uniform_emission_adds_to_it(self):
        path = self._glb(self._record([{"node": "SCREEN", "channels": ["highlight"]}]))
        curves = {"ShotA": {("SCREEN", "highlight"): [[0.0, 0.0, "linear"], [0.5, 0.5, "linear"]]}}
        MeshConvert.apply_glb_fades(path, curves=curves)
        ((_p, _k, _t, values, _i, extras),) = self._decoded(path, "ShotA")
        self.assertEqual([[round(c, 4) for c in v] for v in values], [[0.1] * 3, [0.6] * 3])
        self.assertNotIn("active", extras["render_effects"], "no map to set aside")

    # --------------------------------------------------------------- emission
    def test_an_emission_track_animates_every_material_of_its_name(self):
        """A material target: the curve is the weight scaling the entry's
        colour times its strength, on every material the name joins (the
        namespace-stripped one too); a peak above 1 moves into the material's
        KHR_materials_emissive_strength, glTF capping the factor at 1."""
        path = self._glb(
            self._record(
                materials=[
                    {
                        "material": "screen",
                        "channels": ["emission"],
                        "emission_color": [1.0, 0.5, 0.0],
                        "emission_strength": 4.0,
                    }
                ]
            )
        )
        curves = {"ShotA": {("screen", "emission"): [[0.0, 0.0, "linear"], [0.5, 1.0, "linear"]]}}
        summary = MeshConvert.apply_glb_fades(path, curves=curves)
        self.assertEqual(summary["by_channel"], {"emission": 2})
        gltf = self._gltf(path)
        decoded = self._decoded(path, "ShotA")
        self.assertEqual(
            sorted(c[0] for c in decoded),
            ["/materials/2/emissiveFactor", "/materials/3/emissiveFactor"],
        )
        for index in (2, 3):
            material = gltf["materials"][index]
            self.assertEqual(
                material["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"], 4.0
            )
        (_p, _k, _t, values, _i, extras) = decoded[0]
        self.assertEqual([[round(c, 4) for c in v] for v in values], [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]])
        self.assertEqual(extras["render_effects"], {"channel": "emission", "material": "screen"})
        self.assertIn("KHR_materials_emissive_strength", gltf["extensionsUsed"])

    def test_a_namespaced_target_joins_the_curve_its_proxy_carries(self):
        """The record names a material as the delivery does, namespace
        included (measured 2026-11-07, Maya 2025: a copy imported under ``ns``
        reaches the FBX and the GLB as ``ns:panelMat_signal``), while its
        curve is keyed by its proxy, which the transport names by the leaf
        (``SceneRecords.TRANSPORT.target_name``). A node a host names with a
        namespace (Blender keeps a ``ns:`` in an object's name) joins the same
        way. Measured before: nothing was written, the curve keyed
        ``screen`` never found for ``ns:screen``."""
        transport = SceneRecords.TRANSPORT
        self.assertEqual(transport.target_name("ns:screen"), "screen")
        self.assertEqual(transport.target_name("|grp|ns:BOX"), "BOX")
        self.assertEqual(transport.parse("ns:screen__emission"), ("screen", "emission"))
        self.assertEqual(transport.proxy_name("ns:screen", "emission"), "screen__emission")
        entry = {
            "material": "ns:screen",
            "channels": ["emission"],
            "emission_color": [1.0, 0.5, 0.0],
            "emission_strength": 1.0,
        }
        track = {"node": "ns:BOX", "channels": ["opacity"]}
        path = self._glb(self._record([track], materials=[entry]))
        curves = {
            "ShotA": {
                (transport.target_name("ns:screen"), "emission"): [[0.0, 0.0, "linear"], [0.5, 1.0, "linear"]],
                (transport.target_name("ns:BOX"), "opacity"): [[0.0, 1.0, "linear"], [0.5, 0.0, "linear"]],
            }
        }
        summary = MeshConvert.apply_glb_fades(path, curves=curves)
        self.assertIsNotNone(summary, "the namespaced targets found their curves")
        self.assertEqual(summary["by_channel"], {"emission": 2, "opacity": 1})
        notes = [c[5]["render_effects"] for c in self._decoded(path, "ShotA")]
        self.assertIn({"channel": "emission", "material": "ns:screen"}, notes)
        self.assertIn({"channel": "opacity", "node": "ns:BOX"}, notes)
        gated = self._glb(self._record([{"node": "ns:BOX", "channels": ["visibility"]}]), name="gate.glb")
        self.assertIsNotNone(
            MeshConvert.apply_glb_visibility(
                gated, curves={"ShotA": {("BOX", "visibility"): [[0.0, 1.0, "step"], [0.4, 0.0, "step"]]}}
            )
        )

    # ---------------------------------------------- which node a curve plays on
    def _twins_glb(self, record, proxy=True, name="twins.glb"):
        """Two references of one box: ``a:box`` (whose child is the curve
        proxy ``box__opacity``, marked, when *proxy*) and ``b:box``, each
        wearing its own material; one clip."""
        marker = SceneRecords.TRANSPORT.marker
        nodes = [
            {"name": "a:box", "mesh": 0},
            {"name": "b:box", "mesh": 1},
        ]
        if proxy:
            nodes[0]["children"] = [2]
            nodes.append(
                {
                    "name": "box__opacity",
                    "extras": {"fromFBX": {"userProperties": {marker: {"type": "eFbxBool", "value": 1}}}},
                }
            )
        nodes.append(
            {
                "name": "data_export",
                "extras": {
                    "fromFBX": {
                        "userProperties": {
                            SceneRecords.VISIBILITY.key: {"type": "eFbxString", "value": json.dumps(record)}
                        }
                    }
                },
            }
        )
        gltf = {
            "asset": {"version": "2.0"},
            "nodes": nodes,
            "meshes": [
                {"primitives": [{"attributes": {}, "material": 0}]},
                {"primitives": [{"attributes": {}, "material": 1}]},
            ],
            "materials": [
                {"name": "a_mat", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0]}},
                {"name": "b_mat", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0]}},
            ],
            "accessors": [{"type": "SCALAR", "min": [0.0], "max": [1.0]}],
            "animations": [
                {
                    "name": "ShotA",
                    "samplers": [{"input": 0, "output": 99}],
                    "channels": [{"sampler": 0, "target": {"node": 1, "path": "translation"}}],
                }
            ],
            "buffers": [{"byteLength": 16}],
        }
        return _write_glb_file(os.path.join(self.tmp, name), gltf, bin_chunk=b"\x00" * 16)

    FADE_OUT = {"ShotA": {("box", "opacity"): [[0.0, 1.0, "linear"], [0.5, 0.0, "linear"]]}}

    def _animated(self, path):
        """``(faded material names, gated node names)`` in ShotA."""
        gltf = self._gltf(path)
        faded, gated = set(), set()
        for label, kind, *_rest in self._decoded(path, "ShotA"):
            if kind == "pointer":
                faded.add(gltf["materials"][int(label.split("/")[2])]["name"])
            elif kind == "scale":
                gated.add(label)
        return faded, gated

    def test_a_node_is_its_proxys_parent_never_a_leaf_twin(self):
        """Two references share the leaf ``box``; only ``a:box`` carries the
        curve's proxy, so only ``a:box`` fades and is gated. Measured before:
        the leaf join gave ``{'box': [0, 1]}`` and animated ``b:box`` with
        ``a:box``'s fade. Added: 2026-11-08"""
        path = self._twins_glb(self._record([{"node": "box", "channels": ["opacity"]}]))
        self.assertIsNotNone(MeshConvert.apply_glb_visibility(path, curves=self.FADE_OUT))
        self.assertIsNotNone(MeshConvert.apply_glb_fades(path, curves=self.FADE_OUT))
        self.assertEqual(self._animated(path), ({"a_mat"}, {"a:box"}))

    def test_the_proxys_parent_is_read_before_the_strip(self):
        """The conversion strips the proxies before these passes run; the
        parents read before it (``GlbFades.carriers``) still say which node
        plays the curve. Added: 2026-11-08"""
        path = self._twins_glb(self._record([{"node": "box", "channels": ["opacity"]}]))
        with MeshConvert.open_glb(path) as edit:
            carriers = GlbFades.carriers(edit.gltf)
            self.assertEqual(carriers, {("box", "opacity"): ["|a:box"]})
            MeshConvert.strip_glb_curve_proxies(edit)
            MeshConvert.apply_glb_visibility(edit, curves=self.FADE_OUT, carriers=carriers)
            MeshConvert.apply_glb_fades(edit, curves=self.FADE_OUT, carriers=carriers)
        self.assertEqual(self._animated(path), ({"a_mat"}, {"a:box"}))

    def test_with_no_proxy_a_shared_leaf_is_said_never_guessed(self):
        """No proxy anywhere: the name ``box`` states neither node exactly and
        its leaf names two, so nothing is written and the tie is said.
        Added: 2026-11-08"""
        path = self._twins_glb(
            self._record([{"node": "box", "channels": ["opacity"]}]), proxy=False
        )
        with self.assertLogs("pythontk", level="WARNING") as logs:
            self.assertIsNone(MeshConvert.apply_glb_visibility(path, curves=self.FADE_OUT))
            self.assertIsNone(MeshConvert.apply_glb_fades(path, curves=self.FADE_OUT))
        said = [line for line in logs.output if "never guessed" in line]
        self.assertEqual(len(said), 2, logs.output)
        self.assertEqual(self._animated(path), (set(), set()))

    def test_with_no_proxy_the_exact_name_wins_over_its_leaf_twin(self):
        """No proxy: a track naming ``b:box`` exactly plays on ``b:box``
        alone, though ``a:box`` shares its leaf. Added: 2026-11-08"""
        path = self._twins_glb(
            self._record([{"node": "b:box", "channels": ["opacity"]}]), proxy=False
        )
        MeshConvert.apply_glb_visibility(path, curves=self.FADE_OUT)
        MeshConvert.apply_glb_fades(path, curves=self.FADE_OUT)
        self.assertEqual(self._animated(path), ({"b_mat"}, {"b:box"}))

    def test_the_transport_joins_a_track_by_the_one_rule(self):
        """``SceneRecords.TRANSPORT.join``: the proxy's parent, else the exact
        name, else a unique leaf; a tie never guessed. Added: 2026-11-08"""
        join = SceneRecords.TRANSPORT.join
        paths = ["|a:box", "|a:box|box__opacity", "|b:box", "|grp|crate"]
        self.assertEqual(join("box", paths, [0]), ([0], "proxy"))
        self.assertEqual(join("b:box", paths, [0]), ([0], "proxy"))
        self.assertEqual(join("box", paths, [0, 2]), ([], "ambiguous"))
        self.assertEqual(join("b:box", paths, [0, 2]), ([2], "proxy"))
        self.assertEqual(join("box", paths), ([], "ambiguous"))
        self.assertEqual(join("a:box", paths), ([0], "exact"))
        self.assertEqual(join("|grp|ns:crate", paths), ([3], "leaf"))
        self.assertEqual(join("grp|crate", paths), ([3], "exact"))
        self.assertEqual(join("gone", paths), ([], "absent"))
        # Nodes sharing one name are one name: glTF names are not unique.
        twins = ["|g1|prop", "|g2|prop", "|g3|ns:prop"]
        self.assertEqual(join("prop", twins), ([0, 1], "exact"))
        self.assertEqual(join("g2|prop", twins), ([1], "exact"))
        self.assertEqual(join("x:prop", twins[:2]), ([0, 1], "leaf"))
        self.assertEqual(join("x:prop", twins), ([], "ambiguous"))
        # Carriers the name states none of exactly narrow by its leaf, as the
        # name alone would (measured before: ambiguous with the carriers,
        # ([0, 2], "leaf") without them).
        spaced = ["|g1|ns:prop", "|g1|ns:prop|prop__opacity", "|g2|ns:prop", "|g2|ns:prop|prop__opacity"]
        self.assertEqual(join("prop", spaced), ([0, 2], "leaf"))
        self.assertEqual(join("prop", spaced, [0, 2]), ([0, 2], "proxy"))
        self.assertEqual(join("prop", ["|a:prop", "|b:prop"], [0, 1]), ([], "ambiguous"))

    def test_a_preview_overlay_joins_by_the_transports_name(self):
        """The animation-off path (a push with no take: the host's curves,
        stated as the build's overlay) keys its curves the transport's way
        too, whatever spelling the host handed it."""
        record = self._record(
            materials=[{"material": "ns:screen", "channels": ["emission"], "emission_color": [1.0, 0.5, 0.0]}]
        )
        path = self._glb(record)
        overlay = MeshConvert.effect_preview_curves(
            [{"target": "ns:screen", "channel": "emission", "keys": [[0, 0.0], [15, 1.0]]}],
            start=0,
            fps=30.0,
        )
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["nodes"][-1]["extras"]["fromFBX"]["userProperties"][
                MeshConvert.EFFECT_PREVIEW_KEY
            ] = {"type": "eFbxString", "value": json.dumps(overlay[MeshConvert.EFFECT_PREVIEW_KEY])}
            edit.dirty = True
            curves = MeshConvert.take_preview_curves(edit.gltf)
        self.assertEqual({key for by_target in curves.values() for key in by_target}, {("screen", "emission")})
        self.assertIsNotNone(MeshConvert.apply_glb_fades(path, curves=curves))

    def test_a_second_pass_writes_nothing(self):
        path = self._glb(self._record([{"node": "BOX", "channels": ["opacity", "visibility"]}]))
        curves = {
            "ShotA": {
                ("BOX", "opacity"): [[0.0, 1.0, "linear"], [0.5, 0.0, "linear"]],
                ("BOX", "visibility"): [[0.0, 1.0, "step"], [0.8, 0.0, "step"]],
            }
        }
        MeshConvert.apply_glb_visibility(path, curves=curves)
        MeshConvert.apply_glb_fades(path, curves=curves)
        before = self._gltf(path)
        self.assertIsNone(MeshConvert.apply_glb_fades(path, curves=curves))
        self.assertEqual(self._gltf(path), before)

    def test_a_curve_the_record_does_not_name_is_not_written(self):
        """The record is what a file says it carries: a proxy curve beside it
        for a channel it does not name plays nowhere."""
        path = self._glb(self._record([{"node": "BOX", "channels": ["visibility"]}]))
        curves = {"ShotA": {("BOX", "opacity"): [[0.0, 1.0, "linear"], [0.5, 0.0, "linear"]]}}
        self.assertIsNone(MeshConvert.apply_glb_fades(path, curves=curves))

    # ------------------------------------------- ported from the v1 classes
    def _edited(self, path, change):
        with MeshConvert.open_glb(path) as edit:
            change(edit.gltf)
            edit.dirty = True
        return path

    GATE = {"ShotA": {("BOX", "visibility"): [[0.0, 1.0, "step"], [0.4, 0.0, "step"]]}}

    def test_a_keyed_node_missing_from_the_export_is_reported(self):
        path = self._glb(self._record([{"node": "GONE", "channels": ["visibility"]}]))
        curves = {"ShotA": {("GONE", "visibility"): self.GATE["ShotA"][("BOX", "visibility")]}}
        with self.assertLogs("pythontk", level="WARNING") as logs:
            self.assertIsNone(MeshConvert.apply_glb_visibility(path, curves=curves))
        self.assertTrue(any("GONE" in line and "not in this GLB" in line for line in logs.output))

    def test_a_node_the_clip_already_scales_keeps_its_scale(self):
        """A second scale channel on one node is undefined: the gate stands
        down, warned, rather than corrupt an authored scale animation."""
        def scaled(gltf):
            gltf["animations"][0]["channels"].append(
                {"sampler": 0, "target": {"node": 1, "path": "scale"}}
            )

        path = self._edited(self._glb(self._record([{"node": "BOX", "channels": ["visibility"]}])), scaled)
        with self.assertLogs("pythontk", level="WARNING") as logs:
            self.assertIsNone(MeshConvert.apply_glb_visibility(path, curves=self.GATE))
        self.assertTrue(any("already carry a scale animation" in line for line in logs.output))

    def test_a_node_with_a_baked_matrix_is_reported_as_its_own_problem(self):
        def baked(gltf):
            gltf["nodes"][1]["matrix"] = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]

        path = self._edited(self._glb(self._record([{"node": "BOX", "channels": ["visibility"]}])), baked)
        with self.assertLogs("pythontk", level="WARNING") as logs:
            self.assertIsNone(MeshConvert.apply_glb_visibility(path, curves=self.GATE))
        self.assertTrue(any("baked matrix" in line for line in logs.output))

    def test_the_gate_scales_the_authored_scale_not_one(self):
        def sized(gltf):
            gltf["nodes"][1]["scale"] = [0.5, 2.0, 0.5]

        path = self._edited(self._glb(self._record([{"node": "BOX", "channels": ["visibility"]}])), sized)
        MeshConvert.apply_glb_visibility(path, curves=self.GATE)
        ((_l, _k, _t, values, _i, _x),) = self._decoded(path, "ShotA")
        self.assertEqual(values[0], [0.5, 2.0, 0.5])
        self.assertEqual(values[1], [0.0, 0.0, 0.0])

    def test_keys_sharing_an_instant_resolve_to_the_later(self):
        """A cut on one instant: the later key is the state from it on, and
        the sampler's times still strictly increase."""
        path = self._glb(self._record([{"node": "BOX", "channels": ["visibility"]}]))
        curves = {
            "ShotA": {
                ("BOX", "visibility"): [
                    [0.0, 1.0, "step"],
                    [0.5, 1.0, "step"],
                    [0.5, 0.0, "step"],
                ]
            }
        }
        MeshConvert.apply_glb_visibility(path, curves=curves)
        ((_l, _k, times, values, _i, _x),) = self._decoded(path, "ShotA")
        self.assertEqual([round(t, 5) for t in times], [0.0, 0.5, 1.0])
        self.assertEqual([v[0] for v in values], [1.0, 0.0, 0.0])

    def test_a_mesh_shared_with_an_outsider_is_copied_first(self):
        """BOX and an outsider draw one mesh: BOX gets a copy of the mesh
        (sharing its accessors) wearing the copied material, the outsider the
        original."""
        def shared(gltf):
            gltf["nodes"][2]["mesh"] = 1

        path = self._edited(self._glb(self._record([{"node": "BOX", "channels": ["opacity"]}])), shared)
        curves = {"ShotA": {("BOX", "opacity"): [[0.0, 1.0, "linear"], [0.5, 0.0, "linear"]]}}
        MeshConvert.apply_glb_fades(path, curves=curves)
        gltf = self._gltf(path)
        self.assertNotEqual(gltf["nodes"][1]["mesh"], 1)
        self.assertEqual(gltf["nodes"][2]["mesh"], 1)
        mine = gltf["meshes"][gltf["nodes"][1]["mesh"]]["primitives"][0]["material"]
        self.assertNotEqual(mine, 1)
        self.assertEqual(gltf["meshes"][1]["primitives"][0]["material"], 1)

    def test_an_instanced_mesh_inside_the_subtree_is_isolated_once(self):
        """Two children of BOX draw one mesh: its material is copied once and
        animated once -- a second walk would read the copy as a source."""
        def instanced(gltf):
            gltf["meshes"].append({"primitives": [{"attributes": {}, "material": 1}]})
            gltf["nodes"][1]["children"] = [5, 6]
            gltf["nodes"].insert(5, {"name": "LEG_A", "mesh": 5})
            gltf["nodes"].insert(6, {"name": "LEG_B", "mesh": 5})

        path = self._edited(self._glb(self._record([{"node": "BOX", "channels": ["opacity"]}])), instanced)
        curves = {"ShotA": {("BOX", "opacity"): [[0.0, 1.0, "linear"], [0.5, 0.0, "linear"]]}}
        summary = MeshConvert.apply_glb_fades(path, curves=curves)
        self.assertEqual(summary["materials"], 1)
        self.assertEqual(summary["channels"], 1)

    def test_a_node_drawing_nothing_animates_nothing(self):
        """A locator drives the curve but carries no geometry: nothing under
        it renders, so nothing is animated."""
        def locator(gltf):
            gltf["nodes"].insert(5, {"name": "LOC"})

        path = self._edited(self._glb(self._record([{"node": "LOC", "channels": ["opacity"]}])), locator)
        curves = {"ShotA": {("LOC", "opacity"): [[0.0, 1.0, "linear"], [0.5, 0.0, "linear"]]}}
        self.assertIsNone(MeshConvert.apply_glb_fades(path, curves=curves))

    def test_a_partial_alpha_held_through_a_clip_is_written(self):
        path = self._glb(self._record([{"node": "BOX", "channels": ["opacity"]}]))
        curves = {"ShotA": {("BOX", "opacity"): [[0.0, 0.4, "linear"], [1.0, 0.4, "linear"]]}}
        MeshConvert.apply_glb_fades(path, curves=curves)
        ((_p, _k, _t, values, _i, _x),) = self._decoded(path, "ShotA")
        self.assertEqual({round(v[3], 5) for v in values}, {0.4})

    def test_a_held_zero_glows_only_over_a_coloured_low_stop(self):
        path = self._glb(
            self._record(
                [
                    {"node": "SCREEN", "channels": ["highlight"]},
                    {"node": "PANEL", "channels": ["highlight"], "highlight_color_dim": [0.2, 0.0, 0.0]},
                ]
            )
        )
        held = [[0.0, 0.0, "linear"], [1.0, 0.0, "linear"]]
        curves = {"ShotA": {("SCREEN", "highlight"): held, ("PANEL", "highlight"): held}}
        MeshConvert.apply_glb_fades(path, curves=curves)
        ((pointer, _k, _t, values, _i, _x),) = self._decoded(path, "ShotA")
        self.assertEqual(pointer, "/materials/3/emissiveFactor")
        self.assertEqual([round(c, 5) for c in values[0]], [0.3, 0.1, 0.1])

    def test_the_dim_stop_colours_the_low_half_of_the_glow(self):
        path = self._glb(
            self._record(
                [
                    {
                        "node": "SCREEN",
                        "channels": ["highlight"],
                        "highlight_color": [0.0, 0.0, 0.8],
                        "highlight_color_dim": [0.4, 0.0, 0.0],
                    }
                ]
            )
        )
        curves = {"ShotA": {("SCREEN", "highlight"): [[0.0, 0.0, "linear"], [0.5, 0.5, "linear"], [1.0, 1.0, "linear"]]}}
        MeshConvert.apply_glb_fades(path, curves=curves)
        ((_p, _k, _t, values, _i, _x),) = self._decoded(path, "ShotA")
        self.assertEqual(
            [[round(c, 5) for c in v] for v in values],
            [[0.5, 0.1, 0.1], [0.3, 0.1, 0.5], [0.1, 0.1, 0.9]],
        )

    def test_a_fade_and_a_highlight_on_one_node_share_one_copy(self):
        path = self._glb(self._record([{"node": "BOX", "channels": ["opacity", "highlight"]}]))
        curves = {
            "ShotA": {
                ("BOX", "opacity"): [[0.0, 1.0, "linear"], [0.5, 0.0, "linear"]],
                ("BOX", "highlight"): [[0.0, 0.0, "linear"], [0.5, 1.0, "linear"]],
            }
        }
        summary = MeshConvert.apply_glb_fades(path, curves=curves)
        self.assertEqual((summary["materials"], summary["channels"]), (1, 2))
        self.assertEqual(len(self._gltf(path)["materials"]), 5, "one copy for both")

    def test_a_channel_added_after_the_first_pass_is_still_written(self):
        """Guarded per pointer and clip, never per file: a highlight added to a
        node faded on an earlier pass is written on the same copy."""
        path = self._glb(self._record([{"node": "BOX", "channels": ["opacity", "highlight"]}]))
        fade = {"ShotA": {("BOX", "opacity"): [[0.0, 1.0, "linear"], [0.5, 0.0, "linear"]]}}
        MeshConvert.apply_glb_fades(path, curves=fade)
        both = {"ShotA": {**fade["ShotA"], ("BOX", "highlight"): [[0.0, 0.0, "linear"], [0.5, 1.0, "linear"]]}}
        summary = MeshConvert.apply_glb_fades(path, curves=both)
        self.assertEqual(summary["by_channel"], {"highlight": 1})
        self.assertEqual(len(self._gltf(path)["materials"]), 5)

    # ------------------------------------------------------------- the reader
    def test_the_curves_are_read_off_the_fbx_per_take(self):
        """Each proxy's scale.x, per take, from the clip's t=0: the take's
        first key floored to the converter's bake grid (measured on FBX2glTF
        0.13.1: ShotA keyed from 1/24 s opens on 1/30 s). A node proxy rides
        its node; a material proxy the carrier; a model without the marker,
        or a channel the transport does not carry, is no curve."""
        from test_fbx_media import build_curve_fbx, CONSTANT, LINEAR, USER

        def keys(pairs, flag):
            return {
                "keys": pairs,
                "flags": [flag | USER] * len(pairs),
                "data": [(0.0, 0.0, 0.0, 0.0)] * len(pairs),
            }

        scale = "Lcl Scaling"
        path = build_curve_fbx(
            os.path.join(self.tmp, "proxies.fbx"),
            {
                "ShotA": {
                    ("SIGN__opacity", scale, "d|X"): keys([(1.25, 1.0), (10, 0.0)], LINEAR),
                    ("SIGN__visibility", scale, "d|X"): keys([(1.25, 1.0), (6, 0.0)], CONSTANT),
                    ("screen__emission", scale, "d|X"): keys([(3, 0.0), (12, 1.0)], LINEAR),
                    ("SIGN", scale, "d|X"): keys([(1.25, 1.0), (12, 2.0)], LINEAR),
                    ("SIGN__wobble", scale, "d|X"): keys([(1.25, 1.0), (12, 2.0)], LINEAR),
                },
                "ShotB": {
                    ("SIGN__opacity", scale, "d|X"): keys([(30, 0.0), (45, 1.0)], LINEAR),
                },
            },
            user={
                SceneRecords.TRANSPORT.marker: {
                    "SIGN__opacity": 1,
                    "SIGN__visibility": 1,
                    "screen__emission": 1,
                    "SIGN__wobble": 1,
                }
            },
            parents={
                "SIGN__opacity": "SIGN",
                "SIGN__visibility": "SIGN",
                "SIGN__wobble": "SIGN",
            },
        )
        curves = MeshConvert.render_effect_curves(path, bake_fps=30.0)
        self.assertEqual(set(curves), {"ShotA", "ShotB"})
        self.assertEqual(
            set(curves["ShotA"]),
            {("SIGN", "opacity"), ("SIGN", "visibility"), ("screen", "emission")},
        )
        # ShotA's first key, frame 1.25 at 30 fps, floors to frame 1.
        opacity = [[round(t, 5), v, i] for t, v, i in curves["ShotA"][("SIGN", "opacity")]]
        self.assertEqual(opacity, [[round(0.25 / 30, 5), 1.0, "linear"], [round(9 / 30, 5), 0.0, "linear"]])
        self.assertEqual([k[2] for k in curves["ShotA"][("SIGN", "visibility")]], ["step", "step"])
        self.assertEqual([round(k[0], 5) for k in curves["ShotB"][("SIGN", "opacity")]], [0.0, 0.5])

    def test_a_smooth_segment_is_sampled_at_the_files_rate(self):
        """A cubic segment reaches the pass as the curve's own values at each
        of the file's frames inside it (``FbxFile.take_samplers``), linear
        between: the pass's samplers are LINEAR, and a chord is not the curve."""
        from test_fbx_media import CUBIC, FRAME, USER, build_curve_fbx, packed_weights

        path = build_curve_fbx(
            os.path.join(self.tmp, "smooth.fbx"),
            {
                "ShotA": {
                    ("BOX__opacity", "Lcl Scaling", "d|X"): {
                        "keys": [(0, 0.0), (4, 1.0)],
                        "flags": [CUBIC | USER] * 2,
                        "data": [(0.0, 0.0, packed_weights(), 0.0)] * 2,
                    }
                }
            },
            user={SceneRecords.TRANSPORT.marker: {"BOX__opacity": 1}},
        )
        keys = MeshConvert.render_effect_curves(path, bake_fps=30.0)["ShotA"][("BOX", "opacity")]
        self.assertEqual([round(k[0] * 30, 4) for k in keys], [0, 1, 2, 3, 4])
        self.assertEqual({k[2] for k in keys}, {"linear"})
        ((sample,),) = [
            list(c.values()) for c in FbxFile.load(path).take_samplers(("Lcl Scaling",)).values()
        ]
        self.assertEqual(
            [round(k[1], 5) for k in keys], [round(sample(f * FRAME), 5) for f in range(5)]
        )
        self.assertTrue(0.0 < keys[1][1] < 0.25, "eased, not the chord's 0.25")

    # ---------------------------------------------------------------- preview
    def test_a_preview_states_its_effect_in_seconds_beside_the_record(self):
        overlay = MeshConvert.effect_preview_channels(
            ["SIGN"], "highlight", [[0, 0.0], [15, 1.0]], colors=[[0, 1, 0], None], fps=30.0
        )
        record = overlay[SceneRecords.VISIBILITY.key]
        self.assertEqual(SceneRecords.problems(SceneRecords.VISIBILITY, record), [])
        self.assertEqual(
            record["tracks"], [{"node": "SIGN", "channels": ["highlight"], "highlight_color": [0.0, 1.0, 0.0]}]
        )
        self.assertIsNone(overlay[SceneRecords.SHOTS.key])
        preview = overlay[MeshConvert.EFFECT_PREVIEW_KEY]
        self.assertEqual(
            preview["curves"],
            [{"target": "SIGN", "channel": "highlight", "keys": [[0.0, 0.0], [0.5, 1.0]]}],
        )

    def test_a_preview_plays_in_a_file_with_no_clip(self):
        """The preview's curves are the build's when it states them: a file
        exported with no animation (the WebXR push's default) gets one clip
        for them, and the preview leaves nothing of itself on the carrier."""
        path = self._glb(None, clips=())
        overlay = MeshConvert.effect_preview_channels(["SIGN"], "highlight", [[0, 0.0], [15, 1.0]], fps=30.0)
        with MeshConvert.open_glb(path) as edit:
            MeshConvert.overlay_data_export(edit.gltf, overlay)
            curves = MeshConvert.take_preview_curves(edit.gltf)
            self.assertIsNone(
                MeshConvert.data_export_channel(edit.gltf, MeshConvert.EFFECT_PREVIEW_KEY)
            )
            edit.dirty = True
        self.assertEqual(set(curves), {MeshConvert.DEFAULT_CLIP_NAME})
        MeshConvert.apply_glb_fades(path, curves=curves)
        (channel,) = self._decoded(path, MeshConvert.DEFAULT_CLIP_NAME)
        self.assertEqual(channel[0], "/materials/0/emissiveFactor")


EXTENSION_POINTER = "KHR_animation_pointer"


class TestFadeClonesInTheConversion(unittest.TestCase):
    """The effect pass copies a shared material, and a copy is its source AS
    IT STANDS -- so every by-name repair has to land before it runs.

    The production shape: the preview used to apply the scene sidecar in a
    pass of its own AFTER ``fbx_to_glb``, whose chain had already copied the
    faded materials; the ``materials`` record the FBX carries is the by-name
    statement now. Measured: the metallic-roughness repair reached one
    one-primitive fade copy of ``SCREENS_TEST_CMPTS_MAT`` and the
    eleven-primitive original kept FBX2glTF's packing (roughness and metalness
    255 everywhere). Asserted through ``alphaMode``, the one value the record
    states that FBX2glTF does not: both the original and the fade's copy keep
    it -- opaque at rest, no channel writes BLEND (2026-11-05).
    """

    CURVES = {"Take 001": {("BODY", "opacity"): [[0.0, 1.0, "linear"], [0.5, 0.0, "linear"]]}}

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_fadeorder_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _converted(self, materials_record=None):
        """What FBX2glTF hands the chain: BODY and ARM share SKIN, one clip,
        and the carrier's records -- BODY fades."""
        props = {
            SceneRecords.VISIBILITY.key: {
                "version": SceneRecords.VISIBILITY.version,
                "tracks": [{"node": "BODY", "channels": ["opacity"]}],
            }
        }
        if materials_record is not None:
            props[MeshConvert.MATERIALS_KEY] = materials_record
        return _write_glb_file(
            os.path.join(self.tmp, "converted.glb"),
            {
                "asset": {"version": "2.0"},
                "nodes": [
                    {"name": "BODY", "mesh": 0},
                    {"name": "ARM", "mesh": 1},
                    {
                        "name": MeshConvert.DATA_EXPORT_NODE,
                        "extras": {k: json.dumps(v) for k, v in props.items()},
                    },
                ],
                "meshes": [
                    {"primitives": [{"attributes": {}, "material": 0}]},
                    {"primitives": [{"attributes": {}, "material": 0}]},
                ],
                "materials": [{"name": "SKIN"}],
                "accessors": [{"type": "SCALAR", "min": [0.0], "max": [1.0]}],
                "animations": [
                    {
                        "name": "Take 001",
                        "samplers": [{"input": 0, "output": 0}],
                        "channels": [{"sampler": 0, "target": {"node": 1, "path": "translation"}}],
                    }
                ],
                "buffers": [{"byteLength": 16}],
            },
            bin_chunk=b"\x00" * 16,
        )

    def test_the_original_takes_the_record_before_the_fade_copies_it(self):
        """The FBX's record states SKIN cut out; it lands on the original
        before the effect pass copies it for BODY, and the copy keeps it.
        (Ported 2026-10-31 from the retired sidecar's sections.)"""
        record = MaterialFamilies.payload(
            [
                {
                    "name": "SKIN",
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                    "emissiveFactor": [0.0, 0.0, 0.0],
                    "alphaMode": "MASK",
                    "alphaCutoff": 0.4,
                }
            ]
        )
        converted = self._converted(record)

        def _run(cmd, **kw):
            shutil.copyfile(converted, cmd[cmd.index("-o") + 1] + ".glb")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        src = build_stingray_fbx(os.path.join(self.tmp, "in.fbx"), {}, name="SKIN")
        report = {}
        with (
            unittest.mock.patch.object(MeshConvert, "resolve_binary", return_value="FBX2glTF"),
            unittest.mock.patch("subprocess.run", side_effect=_run),
            unittest.mock.patch.object(MeshConvert, "render_effect_curves", return_value=self.CURVES),
        ):
            out = MeshConvert.fbx_to_glb(
                src, overwrite=True, prompt=False, lightmaps=False, report=report
            )

        gltf = MeshConvert._read_glb(out).gltf
        skins = [m for m in gltf["materials"] if m.get("name") == "SKIN"]
        self.assertEqual(len(skins), 2, "the effect pass must have copied SKIN for BODY")
        for skin in skins:
            self.assertEqual(skin.get("alphaMode"), "MASK")
            self.assertEqual(skin.get("alphaCutoff"), 0.4)
        self.assertNotIn("sidecar", report)

    def test_a_handoff_that_blows_up_costs_the_handoff_not_the_conversion(self):
        """The preview routes its handoff through the conversion, and a push
        must never fail on it: a failure there is logged like every other
        pass in the chain, and the chain carries on."""
        converted = self._converted()

        def _run(cmd, **kw):
            shutil.copyfile(converted, cmd[cmd.index("-o") + 1] + ".glb")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        src = os.path.join(self.tmp, "in.fbx")
        open(src, "wb").write(b"fbx")
        with (
            unittest.mock.patch.object(MeshConvert, "resolve_binary", return_value="FBX2glTF"),
            unittest.mock.patch("subprocess.run", side_effect=_run),
            unittest.mock.patch.object(
                MeshConvert, "_write_glb_handoff", side_effect=TypeError("boom")
            ),
            unittest.mock.patch.object(MeshConvert, "render_effect_curves", return_value=self.CURVES),
            self.assertLogs("pythontk", level="WARNING") as caught,
        ):
            out = MeshConvert.fbx_to_glb(
                src,
                overwrite=True,
                prompt=False,
                sidecar=HandoffRecord.block("glb", []),
            )

        self.assertTrue(os.path.isfile(out))
        self.assertTrue(any("handoff skipped" in m for m in caught.output), caught.output)
        # The chain carried on past it: the effect pass still ran.
        gltf = MeshConvert._read_glb(out).gltf
        self.assertEqual(sum(1 for m in gltf["materials"] if m.get("name") == "SKIN"), 2)


class TestFbxToGlbReport(unittest.TestCase):
    """`fbx_to_glb(report=)`: the chain reports what it did, and sweeps for itself.

    The converter is stubbed (a pre-built GLB stands in for FBX2glTF's output)
    and everything after it is real: the one edit session the preview and the
    Scene Exporters both get through `GlbPipeline`, so what the report says is
    what the panel summarises.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_report_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _convert(self, converted, **kwargs):
        return _convert_prebuilt(self.tmp, converted, **kwargs)

    def _lit_glb(self, manifest, name="lit.glb"):
        """A one-mesh GLB carrying *manifest* the way FBX2glTF transcribes it."""
        carrier = {
            "fromFBX": {
                "userProperties": {
                    "lightmaps": {
                        "type": "eFbxString",
                        "value": json.dumps(manifest),
                    }
                }
            }
        }
        gltf = {
            "asset": {"version": "2.0"},
            "nodes": [
                {"name": "room", "mesh": 0},
                {"name": "data_export", "extras": carrier},
            ],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": 0,
                                "TEXCOORD_0": 1,
                                "TEXCOORD_1": 2,
                            },
                            "material": 0,
                        }
                    ]
                }
            ],
            "materials": [{"name": "roomMat"}],
        }
        return _write_glb_file(os.path.join(self.tmp, name), gltf)

    def _manifest(self, *names):
        return {
            "version": SceneRecords.LIGHTMAPS.version,
            "dir": self.tmp,  # a retired folder hint: exists, holds no map
            "objects": [
                _baked(
                    {
                        "name": name,
                        "map": "atlas_LightMap.exr",
                        "scaleOffset": [1, 1, 0, 0],
                    }
                )
                for name in names
            ],
        }

    def test_an_envelope_gives_the_file_its_handoff_alone(self):
        """An older host's envelope on a file with no materials record: the
        file carries the handoff alone, named against the blocks the build
        added, and no section is applied or reported (removed 2026-10-31)."""
        with self.assertWarns(DeprecationWarning):
            envelope = MeshConvert.build_scene_sidecar(
                {"emissive": {"roomMat": {"color": [1, 0, 0]}}},
                source={"application": "test", "version": "0"},
                asset="in.fbx",
            )
        report = {}
        out = self._convert(self._lit_glb({}), sidecar=envelope, report=report)
        self.assertNotIn("sidecar", report)
        extras = MeshConvert._read_glb(out).gltf["extras"]
        self.assertNotIn("scene_sidecar_applied", extras)
        self.assertNotIn("scene_sidecar", extras)
        # The envelope's handoff, named against the blocks the build added.
        held = {
            spec.key: extras[spec.key]
            for spec in SceneRecords.glb_records()
            if spec.key in extras
        }
        self.assertEqual(
            extras[HandoffRecord.KEY],
            HandoffRecord.block("glb", held, envelope["source"]),
        )

    def test_a_map_found_nowhere_is_reported_unbound(self):
        report = {}
        self._convert(self._lit_glb(self._manifest("room")), report=report)
        self.assertEqual(
            report["lightmaps"],
            {"expected": 1, "bound": 0, "unbound": ["room"], "out_of_scope": 0},
        )

    def test_the_report_counts_only_objects_this_glb_actually_carries(self):
        """The bake manifest is a SCENE record; an export is a subset. Counting
        the whole manifest against a selection-scoped GLB reported every object
        the artist did not select as UNLIT -- crying wolf exactly when the file
        is correct. An object with no node here is out of scope, not unbound."""
        report = {}
        self._convert(
            self._lit_glb(self._manifest("room", "hallway", "stairwell")), report=report
        )
        self.assertEqual(
            report["lightmaps"],
            {"expected": 1, "bound": 0, "unbound": ["room"], "out_of_scope": 2},
        )

    def test_no_manifest_reports_nothing_expected(self):
        report = {}
        self._convert(self._lit_glb({}), report=report)
        self.assertEqual(report["lightmaps"]["expected"], 0)
        self.assertNotIn("sidecar", report, "no section is applied, none reported")

    def test_the_report_keeps_the_manifest_the_file_no_longer_carries(self):
        """The GLB states its lightmaps once, the record as it stands in its
        root extras: the carrier's copy leaves with the FBX's other records,
        and the report keeps the record as the file ships it -- for the
        scene's own record of the export."""
        report = {}
        out = self._convert(self._lit_glb(self._manifest("room")), report=report)
        self.assertEqual(
            [entry["name"] for entry in report["lightmap_manifest"]["objects"]],
            ["room"],
        )
        with MeshConvert.open_glb(out) as edit:
            self.assertEqual(
                report["lightmap_manifest"],
                edit.gltf["extras"][SceneRecords.LIGHTMAPS.key],
            )
            self.assertIsNone(MeshConvert._lightmap_manifest(edit.gltf))
            self.assertEqual(
                [node.get("name") for node in edit.gltf["nodes"]], ["room"]
            )

    def test_lightmap_report_counts_only_what_this_file_carries(self):
        coverage = {
            "present": ["room", "hall"],
            "ambiguous": ["stair"],
            "absent": ["prop"],
        }
        self.assertEqual(
            MeshConvert.lightmap_report(coverage, [{"object": "room"}]),
            {
                "expected": 3,
                "bound": 1,
                "unbound": ["hall", "stair"],
                "out_of_scope": 1,
            },
        )

    def test_no_report_asked_for_means_no_coverage_walk(self):
        with unittest.mock.patch.object(
            MeshConvert,
            "lightmap_manifest_coverage",
            side_effect=AssertionError("walked"),
        ):
            self._convert(self._lit_glb(self._manifest("room")))

    def test_images_no_material_samples_are_swept_inside_the_chain(self):
        """Maya's IBL maps (`diffuse_cube` / `specular_cube` / `ibl_brdf_lut`)
        ride every embedded-media export and FBX2glTF re-embeds them; glTF has
        no slot for them. The retired sidecar applier swept them at its tail,
        so a conversion offered NO envelope shipped the dead payload and paid
        the texture pass to compress it. The sweep is the chain's own now."""
        pixel = "data:image/png;base64,iVBORw0KGgo="
        images = [{"name": "Albedo", "uri": pixel}] + [
            {"name": name, "uri": pixel}
            for name in ("diffuse_cube", "specular_cube", "ibl_brdf_lut")
        ]
        gltf = {
            "asset": {"version": "2.0"},
            "images": images,
            "textures": [{"source": i} for i in range(len(images))],
            "materials": [
                {
                    "name": "m",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                }
            ],
        }
        converted = _write_glb_file(os.path.join(self.tmp, "env.glb"), gltf)
        out = self._convert(converted)
        self.assertEqual(
            [i.get("name") for i in MeshConvert._read_glb(out).gltf["images"]],
            ["Albedo"],
        )

    def test_a_highlighted_material_keeps_its_emissive_map(self):
        """The highlight keeps the emissive map in the file (2026-11-05; it
        dropped it for good before): glTF cannot animate a texture's
        presence, so the channel states the spans the highlight is above 0,
        over which a reader sets the map aside -- and the image ships."""
        pixel = "data:image/png;base64,iVBORw0KGgo="
        props = {
            "animation": _shot_record(["Shot_1"]),
            SceneRecords.VISIBILITY.key: {
                "version": SceneRecords.VISIBILITY.version,
                "tracks": [
                    {
                        "node": "HL",
                        "channels": ["highlight"],
                        "highlight_color": [0.2, 0.5, 1.0],
                    }
                ],
            },
        }
        gltf = {
            "asset": {"version": "2.0"},
            "images": [
                {"name": "Albedo", "uri": pixel},
                {"name": "Glow", "uri": pixel},
            ],
            "textures": [{"source": 0}, {"source": 1}],
            "materials": [
                {
                    "name": "m",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                    "emissiveTexture": {"index": 1},
                    "emissiveFactor": [1.0, 1.0, 1.0],
                }
            ],
            "meshes": [{"primitives": [{"attributes": {}, "material": 0}]}],
            "nodes": [
                {"name": "HL", "mesh": 0},
                {
                    "name": "data_export",
                    "extras": {
                        "fromFBX": {
                            "userProperties": {
                                k: {"type": "eFbxString", "value": json.dumps(v)}
                                for k, v in props.items()
                            }
                        }
                    },
                },
            ],
            "accessors": [],
            "animations": [
                {"name": "Shot_1", "samplers": [], "channels": []}
            ],
            "buffers": [{"byteLength": 4}],
        }
        converted = _write_glb_file(
            os.path.join(self.tmp, "hl.glb"), gltf, bin_chunk=b"\x00" * 4
        )
        curves = {"Shot_1": {("HL", "highlight"): [[0.0, 0.0, "linear"], [0.5, 1.0, "linear"]]}}
        with patch.object(MeshConvert, "render_effect_curves", return_value=curves):
            out = self._convert(converted)
        read = MeshConvert._read_glb(out).gltf
        texture = read["materials"][0]["emissiveTexture"]["index"]
        self.assertIn(read["textures"][texture]["source"], range(len(read["images"])))
        (channel,) = read["animations"][0]["channels"]
        step = MeshConvert.EFFECT_STEP_SECONDS
        self.assertEqual(
            channel["extras"]["render_effects"]["active"],
            [[0.0, 0], [step, 1]],
            "the map at t=0, where the factor is the material's own; glowing from the edge key",
        )


class TestCurveProxyStrip(unittest.TestCase):
    """``strip_glb_curve_proxies``: a proxy leaves with everything only it
    read -- its channels, their samplers, their accessors and BIN bytes -- so
    no pass after it reads a span the file no longer plays. Measured before:
    the samplers stayed, so a clip whose surviving channel ends at 1 ms read
    the proxy's 4 s span (``_animation_time_span``) and was never held to its
    take. Added: 2026-11-07"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_strip_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _glb(self):
        floats = [
            [0.0, 0.001],  # 0: the surviving channel's times
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 1: its translations
            [0.0, 2.0, 4.0],  # 2: the proxy's times, the whole take
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],  # 3: its scale curve
        ]
        blob, views = b"", []
        for values in floats:
            views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": 4 * len(values)})
            blob += struct.pack(f"<{len(values)}f", *values)
        marker = {MeshConvert.CURVE_PROXY_MARKER: {"type": "eBool", "value": 1}}
        gltf = {
            "asset": {"version": "2.0"},
            "nodes": [
                {"name": "PANEL", "children": [1]},
                {"name": "PANEL__opacity", "extras": {"fromFBX": {"userProperties": marker}}},
            ],
            "scenes": [{"nodes": [0]}],
            "scene": 0,
            "bufferViews": views,
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [0.001]},
                {"bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC3"},
                {"bufferView": 2, "componentType": 5126, "count": 3, "type": "SCALAR", "min": [0.0], "max": [4.0]},
                {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC3"},
            ],
            "animations": [
                {
                    "name": "ShotA",
                    "samplers": [{"input": 2, "output": 3}, {"input": 0, "output": 1}],
                    "channels": [
                        {"sampler": 0, "target": {"node": 1, "path": "scale"}},
                        {"sampler": 1, "target": {"node": 0, "path": "translation"}},
                    ],
                }
            ],
            "buffers": [{"byteLength": len(blob)}],
        }
        return _write_glb_file(os.path.join(self.tmp, "strip.glb"), gltf, bin_chunk=blob)

    def test_a_stripped_proxy_takes_its_sampler_accessors_and_bytes(self):
        path = self._glb()
        before = os.path.getsize(path)
        self.assertEqual(MeshConvert.strip_glb_curve_proxies(path), ["PANEL__opacity"])
        self.assertLess(os.path.getsize(path), before - 48, "the proxy's 48 BIN bytes went")
        with MeshConvert.open_glb(path) as edit:
            (clip,) = edit.gltf["animations"]
            self.assertEqual(
                [(c["sampler"], c["target"]) for c in clip["channels"]],
                [(0, {"node": 0, "path": "translation"})],
            )
            self.assertEqual(clip["samplers"], [{"input": 0, "output": 1}], "the proxy's sampler went")
            self.assertEqual(len(edit.gltf["accessors"]), 2, "and the accessors only it read")
            self.assertEqual(edit.gltf["buffers"][0]["byteLength"], 4 * (2 + 6))
            self.assertEqual(MeshConvert._animation_time_span(edit, clip), (0.0, 0.001))
            self.assertEqual(MeshConvert._animation._hold_take_spans(edit, {"ShotA": 4.0}), ["ShotA"])

    def test_a_carrier_left_bare_takes_its_samplers_too(self):
        """The same rule wherever a node goes: the build's last strip removes a
        carrier left with no content (``strip_fbx_records``)."""
        gltf = {
            "nodes": [{"name": "data_export"}, {"name": "BOX"}],
            "accessors": [{"min": [0.0], "max": [9.0]}, {}, {"min": [0.0], "max": [1.0]}, {}],
            "animations": [
                {
                    "name": "A",
                    "samplers": [{"input": 0, "output": 1}, {"input": 2, "output": 3}],
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "scale"}},
                        {"sampler": 1, "target": {"node": 1, "path": "scale"}},
                    ],
                }
            ],
        }
        self.assertEqual(MeshConvert._remove_glb_nodes(gltf, {0}), ["data_export"])
        (clip,) = gltf["animations"]
        self.assertEqual(clip["channels"], [{"sampler": 0, "target": {"node": 0, "path": "scale"}}])
        self.assertEqual(clip["samplers"], [{"input": 0, "output": 1}])
        self.assertEqual(gltf["accessors"], [{"min": [0.0], "max": [1.0]}, {}])


class TestDataExportOverlay(unittest.TestCase):
    """A build can show an effect the scene does not carry.

    ``overlay_data_export`` replaces in-band channels ahead of every reader and
    ``effect_preview_channels`` states one render effect as those channels --
    together, what a panel's "preview at these settings" pushes. Nothing
    upstream of the GLB is involved, which is the point.
    """

    FPS = 30.0
    RED = [1.0, 0.0, 0.0]
    BLUE = [0.2, 0.5, 1.0]

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_overlay_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    @staticmethod
    def _carrier(**props):
        return {
            "name": "data_export",
            "extras": {
                "fromFBX": {
                    "userProperties": {
                        k: {"type": "eFbxString", "value": json.dumps(v)}
                        for k, v in props.items()
                    }
                }
            },
        }

    def test_the_shot_clips_are_read_from_the_record(self):
        """The record names one take per shot, in its order; a version-1
        record (each clip a window with a range) is refused, warned with its
        cure, and so is the take list an older file carried (``fbx_takes``),
        which is no record a reader reads -- such a file is re-exported."""
        shots = {"shots": [{"clip": "S"}, {"clip": "T", "description": "two"}]}
        gltf = {"nodes": [self._carrier(animation={"version": 3, "fps": 30.0, **shots})]}
        self.assertEqual(MeshConvert._animation._shot_clips(gltf), ["S", "T"])
        older = {
            "nodes": [
                self._carrier(
                    animation={
                        "version": 1,
                        "shots": [{"clip": "S", "start": 1, "end": 5}],
                    },
                    fbx_takes=[{"name": "S", "start": 1, "end": 5}],
                )
            ]
        }
        with self.assertLogs("pythontk", level="WARNING") as caught:
            self.assertEqual(MeshConvert._animation._shot_clips(older), [])
        self.assertTrue(any("re-export" in m for m in caught.output), caught.output)

    def test_a_hold_never_rides_a_carrier_however_numbered(self):
        """A take's hold rides a scene root, never a ``data_export`` carrier
        -- the scene's, a module's ``NS:data_export``, or a copy an import
        numbered (``data_export1``, ``data_export.001``:
        ``SceneStoreBase.is_carrier``) -- nor a node with a matrix. Added:
        2026-11-08"""
        names = ["data_export", "data_export1", "data_export.001", "NS:data_export", "box"]
        gltf = {
            "nodes": [{"name": name} for name in names] + [{"name": "m", "matrix": [1] * 16}],
            "scenes": [{"nodes": [0, 1, 2, 3, 4, 5]}],
        }
        self.assertEqual(MeshConvert._animation._hold_roots(gltf), [4])

    def test_a_channel_is_replaced_in_both_shapes_and_read_back(self):
        gltf = {
            "nodes": [
                self._carrier(render_effects={"version": 1, "tracks": ["scene"]}),
                {"name": "native", "extras": {"render_effects": '{"stale": 1}'}},
            ]
        }
        changed = MeshConvert.overlay_data_export(
            gltf, {"render_effects": {"tracks": ["preview"]}}
        )
        self.assertEqual(changed, ["render_effects"])
        self.assertEqual(
            MeshConvert.data_export_channel(gltf, "render_effects"),
            {"tracks": ["preview"]},
        )
        self.assertNotIn("render_effects", gltf["nodes"][1]["extras"])
        self.assertNotIn(
            "render_effects",
            gltf["nodes"][0]["extras"]["fromFBX"]["userProperties"],
        )

    def test_the_scenes_own_carrier_wins_over_a_referenced_modules(self):
        """An assembly export carries one ``NS:data_export`` per referenced
        module beside the scene's own carrier, in whatever order the writer
        put them. Every reader -- this one and unitytk's importers -- takes
        the scene's own (no namespace) first, so the two deliverables of one
        export read the same record; a module's is read only where the
        scene's carries no such channel."""
        module = self._carrier(lightmaps={"from": "module"})
        module["name"] = "LAMP:data_export"
        module_only = self._carrier(animation={"from": "module"})
        module_only["name"] = "LAMP:data_export"
        scene = self._carrier(lightmaps={"from": "scene"})
        gltf = {"nodes": [module, module_only, scene]}
        self.assertEqual(
            MeshConvert.data_export_channel(gltf, "lightmaps"),
            {"from": "scene"},
        )
        self.assertEqual(
            MeshConvert.data_export_channel(gltf, "animation"),
            {"from": "module"},
        )

    def test_a_carrier_under_a_group_is_its_modules_never_the_scenes(self):
        """The scene's own carrier is the root-level ``data_export`` -- a
        scene root, or a child of the converter's ``RootNode`` -- and one of
        the same name under a group is a grouped module's own, read only
        where the scene's carries no such channel, though it comes first;
        the overlay writes onto the scene's. Added: 2026-11-08"""
        nested = self._carrier(lightmaps={"from": "module"}, animation={"from": "module"})
        scene = self._carrier(lightmaps={"from": "scene"})
        gltf = {
            "nodes": [
                {"name": "RootNode", "children": [1, 3]},
                {"name": "Module", "children": [2]},
                nested,
                scene,
            ],
            "scenes": [{"nodes": [0]}],
            "scene": 0,
        }
        self.assertEqual(
            MeshConvert.data_export_channel(gltf, "lightmaps"), {"from": "scene"}
        )
        self.assertEqual(
            MeshConvert.data_export_channel(gltf, "animation"), {"from": "module"}
        )
        MeshConvert.overlay_data_export(gltf, {"lightmaps": {"from": "overlay"}})
        self.assertEqual(
            MeshConvert.data_export_channel(gltf, "lightmaps"), {"from": "overlay"}
        )
        self.assertIn("lightmaps", gltf["nodes"][3]["extras"])
        native = {  # a native glTF export: the scene's roots are the file's
            "nodes": [
                {"name": "Module", "children": [1]},
                self._carrier(lightmaps={"from": "module"}),
                self._carrier(lightmaps={"from": "scene"}),
            ],
            "scenes": [{"nodes": [0, 2]}],
        }
        self.assertEqual(
            MeshConvert.data_export_channel(native, "lightmaps"), {"from": "scene"}
        )
        # A file of modules' carriers alone: their records stand in, their
        # handoff never; the overlay makes the scene's own at the root.
        modules = {
            "nodes": [
                {"name": "Module", "children": [1]},
                self._carrier(handoff={"from": "module"}, lightmaps={"from": "module"}),
                dict(self._carrier(audio={"from": "module"}), name="MOD:data_export"),
            ],
            "scenes": [{"nodes": [0, 2]}],
            "scene": 0,
        }
        self.assertEqual(MeshConvert.data_export_channel(modules, "lightmaps"), {"from": "module"})
        self.assertIsNone(MeshConvert.data_export_channel(modules, "handoff"))
        MeshConvert.overlay_data_export(modules, {"audio": {"from": "overlay"}})
        self.assertEqual(modules["nodes"][3]["name"], "data_export")
        self.assertEqual(modules["scenes"][0]["nodes"], [0, 2, 3])
        self.assertEqual(MeshConvert.data_export_channel(modules, "audio"), {"from": "overlay"})

    def test_the_root_level_is_the_scenes_roots_or_the_converters_children(self):
        """``_Sidecar._root_level_nodes``: the scene roots, a converter's
        ``RootNode`` standing for its children; with no scene, every node no
        node parents; with several scenes, every scene's roots. Added:
        2026-11-08"""
        level = MeshConvert._sidecar._root_level_nodes
        unscened = {
            "nodes": [
                {"name": "Module", "children": [1]},
                {"name": "data_export"},
                {"name": "data_export"},
            ]
        }
        self.assertEqual(level(unscened), {0, 2})
        converted = {
            "nodes": [{"name": "RootNode", "children": [1, 2]}, {"name": "a"}, {"name": "b"}],
            "scenes": [{"nodes": [0]}],
        }
        self.assertEqual(level(converted), {1, 2})
        several = {
            "nodes": [{"name": "a"}, {"name": "b", "children": [2]}, {"name": "c"}],
            "scenes": [{"nodes": [0]}, {"nodes": [1]}],
        }
        self.assertEqual(level(several), {0, 1})
        self.assertEqual(level({}), set())

    def test_clearing_a_channel_clears_the_modules_copies_too(self):
        """An overlay states a channel for THIS build, ahead of every reader:
        ``None`` leaves no carrier holding it, a module's included, so no
        reader falls back to a module's stale copy (a preview clears the shot
        record: the effect is nobody's shot). A written value goes onto the
        scene's own carrier alone. Added: 2026-11-08"""
        shots = SceneRecords.SHOTS.key
        module = dict(self._carrier(**{shots: {"from": "module"}}), name="MOD:data_export")
        gltf = {"nodes": [module, self._carrier(**{shots: {"from": "scene"}})]}
        self.assertEqual(MeshConvert.overlay_data_export(gltf, {shots: None}), [shots])
        self.assertIsNone(MeshConvert.data_export_channel(gltf, shots))
        MeshConvert.overlay_data_export(gltf, {shots: {"from": "overlay"}})
        self.assertNotIn(shots, gltf["nodes"][0]["extras"])
        self.assertIn(shots, gltf["nodes"][1]["extras"])

    def test_none_clears_a_channel_and_leaves_the_rest(self):
        gltf = {
            "nodes": [self._carrier(animation={"version": 1}, lightmaps={"version": 2})]
        }
        shots = SceneRecords.SHOTS.key
        self.assertEqual(MeshConvert.overlay_data_export(gltf, {shots: None}), [shots])
        self.assertIsNone(MeshConvert.data_export_channel(gltf, shots))
        self.assertEqual(
            MeshConvert.data_export_channel(gltf, "lightmaps"), {"version": 2}
        )
        empty = {"nodes": []}
        self.assertEqual(MeshConvert.overlay_data_export(empty, {shots: None}), [])
        self.assertEqual(empty["nodes"], [], "clearing makes no carrier")

    def test_a_file_with_no_carrier_gets_one_in_its_scene(self):
        gltf = {
            "nodes": [{"name": "cube", "mesh": 0}],
            "scenes": [{"nodes": [0]}],
            "scene": 0,
        }
        MeshConvert.overlay_data_export(gltf, {"render_effects": {"tracks": []}})
        self.assertEqual(gltf["nodes"][1]["name"], MeshConvert.DATA_EXPORT_NODE)
        self.assertEqual(gltf["scenes"][0]["nodes"], [0, 1])
        self.assertEqual(
            MeshConvert.data_export_channel(gltf, "render_effects"), {"tracks": []}
        )

    def test_the_preview_channels_describe_one_effect_on_one_clip(self):
        keys = [(0, 0.0), (10, 1.0), (20, 0.0)]
        out = MeshConvert.effect_preview_channels(
            ["HL", "HL", "B"], "highlight", keys, colors=(self.RED, None), fps=24
        )
        self.assertEqual(
            set(out),
            {SceneRecords.VISIBILITY.key, SceneRecords.SHOTS.key, MeshConvert.EFFECT_PREVIEW_KEY},
        )
        self.assertIsNone(out[SceneRecords.SHOTS.key])
        record = out[SceneRecords.VISIBILITY.key]
        # The record names the channel and holds no curve or rate; the curve
        # rides the overlay for this build, in seconds.
        self.assertNotIn("fps", record)
        self.assertEqual([t["node"] for t in record["tracks"]], ["HL", "B"])
        track = record["tracks"][0]
        self.assertEqual(track["channels"], ["highlight"])
        self.assertEqual(track["highlight_color"], self.RED)
        self.assertNotIn("highlight_color_dim", track, "an unstated stop defaults")
        curves = out[MeshConvert.EFFECT_PREVIEW_KEY]["curves"]
        self.assertEqual([c["target"] for c in curves], ["HL", "B"])
        self.assertEqual(
            [[round(t, 6), v] for t, v in curves[0]["keys"]],
            [[0.0, 0.0], [round(10 / 24, 6), 1.0], [round(20 / 24, 6), 0.0]],
        )

    def test_an_opacity_preview_carries_only_its_ramp(self):
        """Opacity at 0 is presence (the record's "hidden at 0"); a visibility
        channel beside it would be a second copy of the same fact."""
        out = MeshConvert.effect_preview_channels(["F"], "opacity", [(0, 0), (15, 1)])
        track = out[SceneRecords.VISIBILITY.key]["tracks"][0]
        self.assertEqual(track["channels"], ["opacity"])
        (curve,) = out[MeshConvert.EFFECT_PREVIEW_KEY]["curves"]
        self.assertEqual(curve["keys"], [[0.0, 0.0], [0.5, 1.0]])

    def test_an_opacity_overlay_gates_the_node_it_fades(self):
        """End to end through `fbx_to_glb`: with no mirror in the overlay, the
        ramp alone still decides where HL is present, so the preview gates it."""
        overlay = MeshConvert.effect_preview_channels(
            ["HL"], "opacity", [(0, 0.0), (10, 1.0), (20, 0.0)], fps=self.FPS
        )
        out = self._convert(self._scene_glb(), data_export=overlay)
        with MeshConvert.open_glb(out) as edit:
            gltf = edit.gltf
        hl = next(i for i, n in enumerate(gltf["nodes"]) if n.get("name") == "HL")
        gates = [
            channel["target"]
            for animation in gltf.get("animations") or []
            for channel in animation.get("channels") or []
            if channel["target"] == {"node": hl, "path": "scale"}
        ]
        self.assertTrue(gates, "the faded node was not gated")

    def test_an_effect_preview_refuses_what_it_cannot_describe(self):
        ramp = [(0, 0.0), (1, 1.0)]
        with self.assertRaises(KeyError):
            MeshConvert.effect_preview_channels(["A"], "glow", ramp)
        with self.assertRaises(ValueError):
            MeshConvert.effect_preview_channels([], "opacity", ramp)
        with self.assertRaises(ValueError):
            MeshConvert.effect_preview_channels(["A"], "opacity", ramp[:1])

    def _convert(self, converted, **kwargs):
        return _convert_prebuilt(self.tmp, converted, **kwargs)

    def _scene_glb(self, shots=("Shot_1",)):
        """HL, highlighted BLUE over the declared *shots*, as the scene
        exported it with animation off: no clip, so the conversion gives it
        what its overlay states."""
        tracks = {
            "version": SceneRecords.VISIBILITY.version,
            "tracks": [{"node": "HL", "channels": ["highlight"], "highlight_color": self.BLUE}],
        }
        record = _shot_record(list(shots))
        record["shots"][0]["description"] = "the opening"
        carrier = self._carrier(animation=record, render_effects=tracks)
        return _write_glb_file(
            os.path.join(self.tmp, "scene.glb"),
            {
                "asset": {"version": "2.0"},
                "scenes": [{"nodes": [0, 1]}],
                "scene": 0,
                "nodes": [{"name": "HL", "mesh": 0}, carrier],
                "meshes": [{"primitives": [{"attributes": {}, "material": 0}]}],
                "materials": [{"name": "GLOW", "pbrMetallicRoughness": {}}],
                "accessors": [],
                "buffers": [{"byteLength": 4}],
            },
            bin_chunk=b"\x00" * 4,
        )

    @staticmethod
    def _emissive_channels(path):
        """``{clip: [(times, rgb values)]}`` for the clips animating an emissive."""
        out = {}
        for clip, channels in _glb_pointer_channels(path).items():
            emissive = [(t, v) for p, t, v in channels if p.endswith("/emissiveFactor")]
            if emissive:
                out[clip] = emissive
        return out

    def test_the_conversion_builds_the_overlay_rather_than_the_scene(self):
        """End to end through `fbx_to_glb`: the scene carries a take and a BLUE
        highlight on HL; the overlay's RED ramp is what animates, on one clip
        over the ramp's own extent from t=0, and the take's clip is never cut."""
        keys = [(0, 0.0), (10, 1.0), (20, 0.0)]
        overlay = MeshConvert.effect_preview_channels(
            ["HL"], "highlight", keys, colors=(self.RED, None), fps=self.FPS
        )
        report = {}
        out = self._convert(self._scene_glb(), data_export=overlay, report=report)

        # The report names the channels the overlay replaced -- how a caller
        # tells an applied overlay from one a stale bridge never forwarded.
        self.assertIn(SceneRecords.VISIBILITY.key, report["data_export"])
        channels = self._emissive_channels(out)
        self.assertEqual(list(channels), [MeshConvert.DEFAULT_CLIP_NAME])
        ((times, values),) = channels[MeshConvert.DEFAULT_CLIP_NAME]
        self.assertAlmostEqual(times[0], 0.0, places=5)
        self.assertAlmostEqual(times[-1], 20 / self.FPS, places=5)
        peak = max(values, key=lambda rgb: sum(rgb))
        self.assertEqual([round(c, 5) for c in peak], self.RED)
        self.assertTrue(all(rgb[2] == 0.0 for rgb in values), "no trace of BLUE")

    def test_an_animation_off_push_plays_the_scenes_own_curves(self):
        """Bug: a WebXR push with Include Animation off (the default) played
        the scene's render effects until render_effects v2 moved the curves
        into the takes -- such a file carries none, so HL's BLUE highlight
        played nowhere. Fixed: 2026-11-05 -- the push states the scene's own
        curves in the preview overlay, in seconds from the exported range's
        start (``effect_preview_curves``), and the build plays them on its one
        clip in the record's own colours; the record is the file's, untouched.
        """
        overlay = MeshConvert.effect_preview_curves(
            [
                {
                    "target": "HL",
                    "channel": "highlight",
                    "keys": [[10, 0.0, "linear"], [20, 1.0, "step"], [30, 0.0, "linear"]],
                }
            ],
            start=4,
            fps=self.FPS,
        )
        self.assertEqual(list(overlay), [MeshConvert.EFFECT_PREVIEW_KEY])
        report = {}
        out = self._convert(self._scene_glb(), data_export=overlay, report=report)
        self.assertEqual(report["data_export"], [MeshConvert.EFFECT_PREVIEW_KEY])
        channels = self._emissive_channels(out)
        self.assertEqual(list(channels), [MeshConvert.DEFAULT_CLIP_NAME])
        ((times, values),) = channels[MeshConvert.DEFAULT_CLIP_NAME]
        lit = [t for t, rgb in zip(times, values) if [round(c, 5) for c in rgb] == self.BLUE]
        self.assertTrue(lit, "the record's BLUE never plays")
        # Lit from frame 20 (16 frames into the range) and HELD -- a step --
        # until just before frame 30, where it cuts.
        self.assertAlmostEqual(lit[0], 16 / self.FPS, places=5)
        self.assertAlmostEqual(lit[-1], 26 / self.FPS - MeshConvert.EFFECT_STEP_SECONDS, places=5)
        self.assertAlmostEqual(times[-1], 26 / self.FPS, places=5)
        with MeshConvert.open_glb(out) as edit:
            extras = edit.gltf["extras"]
        self.assertNotIn(MeshConvert.EFFECT_PREVIEW_KEY, extras, "build-only, never shipped")
        self.assertEqual(extras[SceneRecords.VISIBILITY.key]["tracks"][0]["channels"], ["highlight"])

    def test_without_an_overlay_an_animation_off_file_animates_nothing(self):
        """The control: a file exported with animation off carries no take, so
        with no overlay stating the scene's curves nothing plays -- the record
        alone names channels and holds no curve."""
        report = {}
        out = self._convert(self._scene_glb(), report=report)
        self.assertNotIn("data_export", report, "no overlay, no claim of one")
        self.assertEqual(self._emissive_channels(out), {})

    def test_an_animation_off_push_of_shots_plays_each_shot_on_its_own_clip(self):
        """Bug: an animation-off push of a scene with declared shots gave the
        viewer one ``Take 001`` -- at HEAD it got one clip per declared take,
        and its picker switched between the shots and played their sequence.
        Fixed: 2026-11-07 -- the push states one clip per take an animated
        write would split, named as that take, each curve in seconds from its
        shot's start and pinned to both of its ends, so each clip spans its
        shot. The record's shots name them, describe them and order them."""
        fps = self.FPS
        overlay = MeshConvert.effect_preview_curves(
            [
                {
                    "target": "HL",
                    "channel": "highlight",
                    "keys": [[5, 0.0, "linear"], [15, 1.0, "step"], [35, 0.0, "linear"]],
                }
            ],
            start=1,
            fps=fps,
            clips=[
                {"name": "ShotA", "start": 1, "end": 24},
                {"name": "ShotB", "start": 25, "end": 48},
            ],
        )
        self.assertEqual(
            [c["name"] for c in overlay[MeshConvert.EFFECT_PREVIEW_KEY]["clips"]],
            ["ShotA", "ShotB"],
        )
        report = {}
        out = self._convert(self._scene_glb(("ShotA", "ShotB")), data_export=overlay, report=report)
        with MeshConvert.open_glb(out) as edit:
            gltf = edit.gltf
        self.assertEqual([a["name"] for a in gltf["animations"]], ["ShotA", "ShotB"])
        carried = gltf["extras"][SceneRecords.SHOTS.key]
        self.assertEqual([s["clip"] for s in carried["shots"]], ["ShotA", "ShotB"])
        self.assertEqual(carried["shots"][0]["description"], "the opening")
        channels = self._emissive_channels(out)
        blue = lambda rgb: [round(c, 5) for c in rgb] == self.BLUE  # noqa: E731
        ((a_times, a_values),) = channels["ShotA"]
        ((b_times, b_values),) = channels["ShotB"]
        # Each clip spans its shot: 23 frames from its first to its last.
        self.assertAlmostEqual(a_times[0], 0.0, places=5)
        self.assertAlmostEqual(a_times[-1], 23 / fps, places=5)
        self.assertAlmostEqual(b_times[0], 0.0, places=5)
        self.assertAlmostEqual(b_times[-1], 23 / fps, places=5)
        # Lit from frame 15 (14 frames into ShotA) to its end, held into ShotB
        # from its start until the step cuts at frame 35 (10 frames in).
        lit_a = [t for t, rgb in zip(a_times, a_values) if blue(rgb)]
        lit_b = [t for t, rgb in zip(b_times, b_values) if blue(rgb)]
        self.assertAlmostEqual(lit_a[0], 14 / fps, places=5)
        self.assertAlmostEqual(lit_a[-1], 23 / fps, places=5)
        self.assertAlmostEqual(lit_b[0], 0.0, places=5)
        self.assertAlmostEqual(lit_b[-1], 10 / fps - MeshConvert.EFFECT_STEP_SECONDS, places=5)

    def test_a_shot_whose_curves_rest_still_gets_its_clip(self):
        """A shot in which no effect moves -- HL's highlight at rest through
        S2 -- still has its clip, spanning the shot by a hold, so the page's
        picker and sequence keep it (the same rule as a still take)."""
        overlay = MeshConvert.effect_preview_curves(
            [{"target": "HL", "channel": "highlight", "keys": [[1, 0.0], [12, 1.0], [24, 0.0]]}],
            start=1,
            fps=self.FPS,
            clips=[{"name": "S1", "start": 1, "end": 24}, {"name": "S2", "start": 24, "end": 48}],
        )
        self.assertEqual(
            [c.get("duration") for c in overlay[MeshConvert.EFFECT_PREVIEW_KEY]["clips"]],
            [round(23 / self.FPS, 9), round(24 / self.FPS, 9)],
        )
        out = self._convert(self._scene_glb(("S1", "S2")), data_export=overlay)
        with MeshConvert.open_glb(out) as edit:
            gltf = edit.gltf
            spans = {a["name"]: MeshConvert._animation_time_span(edit, a) for a in gltf["animations"]}
            s2 = next(a for a in gltf["animations"] if a["name"] == "S2")
        self.assertEqual(list(spans), ["S1", "S2"])
        self.assertAlmostEqual(spans["S2"][1], 24 / self.FPS, places=5)
        ((channel),) = s2["channels"]
        self.assertEqual(s2["samplers"][channel["sampler"]]["interpolation"], "STEP")
        self.assertEqual(gltf["nodes"][channel["target"]["node"]]["name"], "HL")

    def test_a_shots_curves_are_cut_to_its_window(self):
        """Per clip, a curve is rebased to the clip's start and cut to its
        window, both ends pinned at the value it holds there (a step's held
        value, else interpolated); every curve reaches every clip."""
        out = MeshConvert.effect_preview_curves(
            [
                {"target": "A", "channel": "opacity", "keys": [[0, 0.0], [20, 1.0]]},
                {"target": "B", "channel": "visibility", "keys": [[5, 1.0, "step"], [12, 0.0, "step"]]},
            ],
            start=0,
            fps=10,
            clips=[{"name": "S1", "start": 0, "end": 10}, {"name": "S2", "start": 10, "end": 30}],
        )
        (s1, s2) = out[MeshConvert.EFFECT_PREVIEW_KEY]["clips"]
        curves = {c["name"]: {k["target"]: k["keys"] for k in c["curves"]} for c in (s1, s2)}
        self.assertEqual(curves["S1"]["A"], [[0.0, 0.0, "linear"], [1.0, 0.5, "linear"]])
        self.assertEqual(curves["S2"]["A"], [[0.0, 0.5, "linear"], [1.0, 1.0, "linear"], [2.0, 1.0, "linear"]])
        self.assertEqual(curves["S1"]["B"], [[0.0, 1.0, "linear"], [0.5, 1.0, "step"], [1.0, 1.0, "linear"]])
        self.assertEqual(curves["S2"]["B"], [[0.0, 1.0, "step"], [0.2, 0.0, "step"], [2.0, 0.0, "linear"]])
        self.assertNotIn("curves", out[MeshConvert.EFFECT_PREVIEW_KEY], "clips or curves, never both")

    def test_the_scenes_curves_are_stated_in_seconds_from_the_range_start(self):
        """Frames become seconds from *start*; each key keeps its segment's
        interpolation; a curve keyed before the range opens on the value it
        holds at the start (a step's, else the segment's own, interpolated) --
        glTF has no time before 0. A curve of no keys states nothing."""
        out = MeshConvert.effect_preview_curves(
            [
                {"target": "A", "channel": "opacity", "keys": [[0, 0.0], [20, 1.0]]},
                {"target": "B", "channel": "visibility", "keys": [[0, 1.0, "step"], [30, 0.0, "step"]]},
                {"target": "C", "channel": "highlight", "keys": []},
            ],
            start=10,
            fps=20,
        )
        curves = {c["target"]: c for c in out[MeshConvert.EFFECT_PREVIEW_KEY]["curves"]}
        self.assertEqual(set(curves), {"A", "B"})
        self.assertEqual(curves["A"]["channel"], "opacity")
        self.assertEqual(curves["A"]["keys"], [[0.0, 0.5, "linear"], [0.5, 1.0, "linear"]])
        self.assertEqual(curves["B"]["keys"], [[0.0, 1.0, "step"], [1.0, 0.0, "step"]])
        self.assertEqual(MeshConvert.effect_preview_curves([], start=1, fps=24), {})
        with self.assertRaises(ValueError):
            MeshConvert.effect_preview_curves(
                [{"target": "A", "channel": "opacity", "keys": [[0, 0.0]]}], start=0, fps=0
            )

    def test_without_an_overlay_the_files_own_curves_build(self):
        """The same scene exported with its take: the curve its proxy carries
        is what animates, BLUE, on the take's clip."""
        glb = self._scene_glb()
        with MeshConvert.open_glb(glb) as edit:
            edit.gltf["accessors"] = [{"type": "SCALAR", "min": [0.0], "max": [1.0]}]
            edit.gltf["animations"] = [
                {
                    "name": "Shot_1",
                    "samplers": [{"input": 0, "output": 0}],
                    "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
                }
            ]
            edit.dirty = True
        curves = {"Shot_1": {("HL", "highlight"): [[0.0, 0.0, "linear"], [0.5, 1.0, "linear"]]}}
        with patch.object(MeshConvert, "render_effect_curves", return_value=curves):
            channels = self._emissive_channels(self._convert(glb))
        self.assertEqual(list(channels), ["Shot_1"])
        ((_times, values),) = channels["Shot_1"]
        self.assertEqual([round(c, 5) for c in values[-1]], self.BLUE)


class TestOptimizeGlbWebpSemantics(unittest.TestCase):
    """WebP mode: the lossy encoder only where the channels ARE colour.

    Lossy WebP is YUV 4:2:0, so it resamples the chroma planes at half
    resolution. In a normal map those planes hold X and Z; in an ORM they hold
    occlusion and metalness. Measured on this pipeline's own 2K maps at the
    pass's own quality, base colour held 37.6 dB while normal X fell to 31.7 dB
    and ORM metalness to 30.8 dB -- smeared normals and flat roughness, which
    reads as a deliverable shipped WITHOUT those maps rather than with damaged
    ones. KTX2 mode has always split by semantic (`BASIS_BY_SEMANTIC`); this is
    the same rule reaching the default container.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_webp_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ helpers
    def _noise(self, seed, size=(96, 96)):
        """A deterministic noise image, and its raw pixels.

        Noise on purpose: it is the one content a lossy codec cannot reproduce
        and a lossless one cannot shrink much, so "did the pixels survive"
        answers cleanly either way. Deterministic so a failure is reproducible.
        """
        import base64 as b64

        from PIL import Image

        state = seed or 1
        data = bytearray()
        for _ in range(size[0] * size[1] * 3):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            data.append((state >> 16) & 0xFF)
        image = Image.frombytes("RGB", size, bytes(data))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        uri = "data:image/png;base64," + b64.b64encode(buffer.getvalue()).decode(
            "ascii"
        )
        return uri, image.tobytes()

    def _glb(self, gltf, name="scene.glb"):
        json_bytes = json.dumps(gltf).encode("utf-8")
        json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
        blob = (
            struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_bytes))
            + struct.pack("<I4s", len(json_bytes), b"JSON")
            + json_bytes
        )
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(blob)
        return path

    def _pixels(self, path):
        """Every image in the written GLB, decoded, by image index."""
        from PIL import Image

        edit = MeshConvert._read_glb(path)
        out = []
        for image in edit.gltf.get("images") or []:
            payload = edit._image_payload(image)
            out.append(Image.open(io.BytesIO(payload)).convert("RGB").tobytes())
        return out

    def _run(self, material, count=2):
        """One material wearing *count* distinct noise images, image 0 first."""
        images, pixels = [], []
        for index in range(count):
            uri, raw = self._noise(index + 1)
            images.append({"name": f"img{index}", "uri": uri})
            pixels.append(raw)
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": images,
                "textures": [{"source": i} for i in range(count)],
                "materials": [material],
            }
        )
        # max_size=0: no resample, so a pixel comparison is a statement about
        # the ENCODER and not about LANCZOS.
        MeshConvert.optimize_glb_textures(path, max_size=0, image_format="WEBP")
        return pixels, self._pixels(path)

    # -------------------------------------------------------------------- tests
    def test_a_normal_map_survives_the_pass_pixel_for_pixel(self):
        before, after = self._run(
            {
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                "normalTexture": {"index": 1},
            }
        )
        self.assertNotEqual(
            before[0], after[0], "base colour should still take the lossy encoder"
        )
        self.assertEqual(before[1], after[1], "the normal map was re-encoded lossily")

    def test_a_metallic_roughness_map_survives_the_pass_pixel_for_pixel(self):
        """Roughness sits in G and metalness in B -- both chroma planes."""
        before, after = self._run(
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                }
            }
        )
        self.assertNotEqual(before[0], after[0])
        self.assertEqual(before[1], after[1], "the ORM map was re-encoded lossily")

    def test_an_occlusion_map_survives_the_pass_pixel_for_pixel(self):
        before, after = self._run(
            {
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                "occlusionTexture": {"index": 1},
            }
        )
        self.assertNotEqual(before[0], after[0])
        self.assertEqual(before[1], after[1])

    def test_bytes_sampled_as_both_colour_and_normal_take_the_stricter_encode(self):
        """`_SEMANTIC_RANK` decides, and now it decides lossy vs lossless too.

        One image bound as base colour in one material and as a normal map in
        another is the case where a per-slot rule has to pick: encoding it as
        colour would damage the material reading it as a normal map, and no
        assertion about either material alone would catch it.
        """
        uri, raw = self._noise(7)
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "shared", "uri": uri}],
                "textures": [{"source": 0}],
                "materials": [
                    {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}},
                    {"normalTexture": {"index": 0}},
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path, max_size=0, image_format="WEBP")
        self.assertEqual(raw, self._pixels(path)[0])

    def test_a_lightmap_is_still_lossless_whatever_it_is_sampled_as(self):
        """The carve-out that predates the rule keeps working under it."""
        uri, raw = self._noise(3)
        path = self._glb(
            {
                "asset": {"version": "2.0"},
                "images": [{"name": "bake", "uri": uri}],
                "textures": [{"source": 0}],
                # texCoord 1 on the carrier slot: what makes it a lightmap,
                # whatever it is named.
                "materials": [
                    {
                        "name": "room",
                        "occlusionTexture": {"index": 0, "texCoord": 1},
                    }
                ],
            }
        )
        MeshConvert.optimize_glb_textures(path, max_size=0, image_format="WEBP")
        self.assertEqual(raw, self._pixels(path)[0])


class TestSetGlbNormalScale(unittest.TestCase):
    """`normalTexture.scale`, written where glTF already keeps it."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_normalscale_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _glb(self, manifest=True, version=None):
        """Three materials: baked+normal, baked without one, unbaked+normal,
        each worn by a node of its name -- the baked set the nodes the root's
        ``lightmaps`` record lights (of *version* when given), their maps
        riding the file.
        """
        gltf = {
            "asset": {"version": "2.0"},
            "nodes": [
                {"name": name, "mesh": index}
                for index, name in enumerate(("baked", "baked_flat", "prop"))
            ],
            "meshes": [
                {"primitives": [{"attributes": {"TEXCOORD_1": 0}, "material": i}]}
                for i in range(3)
            ],
            "materials": [
                {"name": "baked", "normalTexture": {"index": 0}},
                {"name": "baked_flat"},
                {"name": "prop", "normalTexture": {"index": 0}},
            ],
            "textures": [{"source": 0}],
            "images": [{"name": "n", "uri": "data:image/png;base64,"}],
        }
        if manifest:
            record = _lightmaps_record("baked", "baked_flat", version=version)
            gltf["extras"] = {SceneRecords.LIGHTMAPS.key: record}
        path = _write_glb_file(os.path.join(self.tmp, "scene.glb"), gltf)
        return _ride_media(path, ["baked_LM.exr", "baked_flat_LM.exr"])

    def _materials(self, path):
        return {m["name"]: m for m in MeshConvert._read_glb(path).gltf["materials"]}

    def test_it_writes_only_the_baked_materials_that_have_a_normal_map(self):
        path = self._glb()
        self.assertEqual(MeshConvert.set_glb_normal_scale(path, 1.8), 1)
        materials = self._materials(path)
        self.assertEqual(materials["baked"]["normalTexture"]["scale"], 1.8)
        # The prop is not baked, so the flattening this dial answers does not
        # apply to it -- turning it up would be re-authoring the asset.
        self.assertNotIn("scale", materials["prop"]["normalTexture"])
        self.assertNotIn("normalTexture", materials["baked_flat"])

    def test_off_the_manifest_it_writes_every_normal_mapped_material(self):
        path = self._glb()
        self.assertEqual(
            MeshConvert.set_glb_normal_scale(path, 0.5, lightmapped_only=False), 2
        )
        materials = self._materials(path)
        self.assertEqual(materials["baked"]["normalTexture"]["scale"], 0.5)
        self.assertEqual(materials["prop"]["normalTexture"]["scale"], 0.5)

    def test_one_removes_the_key_rather_than_writing_the_default(self):
        """A reset must leave the file as if the dial had never moved."""
        path = self._glb()
        MeshConvert.set_glb_normal_scale(path, 1.8)
        self.assertEqual(MeshConvert.set_glb_normal_scale(path, 1.0), 1)
        self.assertNotIn("scale", self._materials(path)["baked"]["normalTexture"])

    def test_writing_the_value_it_already_holds_changes_nothing(self):
        path = self._glb()
        MeshConvert.set_glb_normal_scale(path, 1.8)
        self.assertEqual(MeshConvert.set_glb_normal_scale(path, 1.8), 0)

    def test_a_reset_on_an_untouched_file_changes_nothing(self):
        self.assertEqual(MeshConvert.set_glb_normal_scale(self._glb(), 1.0), 0)

    def test_a_file_with_no_manifest_matches_nothing(self):
        """`lightmapped_only` means what it says: no record, no baked set."""
        path = self._glb(manifest=False)
        self.assertEqual(MeshConvert.set_glb_normal_scale(path, 1.8), 0)

    def test_a_block_of_another_version_names_no_baked_set(self):
        """The baked set is read off the record in its current schema alone
        (``SceneRecords.glb_block``): one of another version -- an older one
        too -- names none, and the file is re-exported. Added: 2026-10-20"""
        path = self._glb(version=SceneRecords.LIGHTMAPS.version - 1)
        self.assertEqual(MeshConvert.set_glb_normal_scale(path, 1.8), 0)

    def test_a_node_whose_map_does_not_ride_is_not_baked_here(self):
        """The baked set is what the record lights IN THIS FILE: a node its
        record names whose map the file does not carry renders unlit, and the
        flattening this dial answers is not there. Added: 2026-11-07"""
        path = self._glb()
        with MeshConvert.open_glb(path) as edit:
            for view in edit.gltf["bufferViews"]:
                view.pop(SceneRecords.GLB_MEDIA.name, None)
            edit.dirty = True
        self.assertEqual(MeshConvert.set_glb_normal_scale(path, 1.8), 0)


class TestPruneGlbAnimations(unittest.TestCase):
    """Animations with no channels or samplers are dropped before the manifest.

    Measured on two static production modules (2026-09-02): the Scene Exporter
    had animation armed, the scenes carried no keys, Maya wrote its whole-
    timeline ``Take 001`` AnimStack anyway (a stack and a layer, no curves), and
    FBX2glTF transcribed it as ``animations[0]`` with neither channels nor
    samplers. glTF requires at least one of each, so the file did not validate
    -- and the manifest pass, which saw the entry, listed it as the default clip.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_prune_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _glb(self, animations, name="prune.glb"):
        accessors = [{"type": "SCALAR", "min": [0.0], "max": [1.0]}]
        return _write_glb_file(
            os.path.join(self.tmp, name),
            {
                "asset": {"version": "2.0"},
                "nodes": [{"name": "cube", "mesh": 0}],
                "accessors": accessors,
                "animations": animations,
            },
        )

    @staticmethod
    def _populated(name):
        return {
            "name": name,
            "samplers": [{"input": 0, "output": 0}],
            "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
        }

    def _names(self, path):
        with MeshConvert.open_glb(path) as edit:
            return [a.get("name") for a in edit.gltf.get("animations") or []]

    def test_a_hollow_whole_timeline_stack_is_dropped(self):
        """The static-scene case: the one entry goes, and the key with it."""
        path = self._glb([{"name": "Take 001"}])
        self.assertEqual(MeshConvert.prune_glb_animations(path), ["Take 001"])
        with MeshConvert.open_glb(path) as edit:
            # ``animations`` has ``minItems: 1``; an empty list is as invalid
            # as the hollow entry was.
            self.assertNotIn("animations", edit.gltf)
        report = MeshConvert.verify_glb(path)
        # Only the animation verdict is under test here: ``verify_glb`` also
        # wants an ``extras.handoff``, which this fixture never had.
        self.assertFalse(
            [p for p in report["problems"] if "channels" in p], report["problems"]
        )

    def test_populated_animations_survive_in_order(self):
        path = self._glb(
            [
                {"name": "Take 001"},
                self._populated("SHOT_A"),
                {"name": "SHOT_B", "samplers": [{"input": 0, "output": 0}]},
                self._populated("SHOT_C"),
            ]
        )
        self.assertEqual(MeshConvert.prune_glb_animations(path), ["Take 001", "SHOT_B"])
        self.assertEqual(self._names(path), ["SHOT_A", "SHOT_C"])

    def test_a_sound_file_is_not_rewritten(self):
        path = self._glb([self._populated("SHOT_A")])
        with open(path, "rb") as fh:
            before = fh.read()
        self.assertEqual(MeshConvert.prune_glb_animations(path), [])
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(), before)

    def test_a_file_with_no_animations_is_a_no_op(self):
        path = _write_glb_file(
            os.path.join(self.tmp, "static.glb"),
            {"asset": {"version": "2.0"}, "nodes": [{"name": "cube"}]},
        )
        self.assertEqual(MeshConvert.prune_glb_animations(path), [])

    def test_the_conversion_ships_no_hollow_clip_and_no_manifest_for_it(self):
        """End to end: the converter's hollow ``Take 001`` never reaches the file.

        The manifest pass runs after the prune, so a static scene gets neither
        an invalid animation nor an ``animation`` block naming it as the
        default clip.
        """
        src = os.path.join(self.tmp, "static.fbx")
        with open(src, "wb") as fh:
            fh.write(b"Kaydara FBX Binary  \x00")
        fake_bin = os.path.join(self.tmp, "FBX2glTF.exe")
        with open(fake_bin, "wb") as fh:
            fh.write(b"MZ")
        dst = os.path.join(self.tmp, "static.glb")

        def _run(cmd, *args, **kwargs):
            _write_glb_file(
                cmd[cmd.index("-o") + 1] + ".glb",
                {
                    "asset": {"version": "2.0"},
                    "nodes": [{"name": "cube", "mesh": 0}],
                    "animations": [{"name": "Take 001"}],
                },
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with (
            patch.object(MeshConvert, "resolve_binary", return_value=fake_bin),
            patch("subprocess.run", side_effect=_run),
        ):
            MeshConvert.fbx_to_glb(src, dst, auto_install=False)

        with MeshConvert.open_glb(dst) as edit:
            self.assertNotIn("animations", edit.gltf)
            self.assertNotIn(SceneRecords.SHOTS.key, edit.gltf.get("extras") or {})
        report = MeshConvert.verify_glb(dst)
        # Only the animation verdict is under test here: ``verify_glb`` also
        # wants an ``extras.handoff``, which this fixture never had.
        self.assertFalse(
            [p for p in report["problems"] if "channels" in p], report["problems"]
        )


class TestGlbSkins(unittest.TestCase):
    """A GLB's skinning data: what glTF requires of it, and what anything binds."""

    #: A rigid mesh's positions, a kept skin's bind matrix and a clip's output:
    #: distinct values, so a renumber that re-points one at another accessor's
    #: bytes reads back wrong.
    RIGID = [5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 5.0, 1.0, 0.0]
    IBM = [float(i) for i in range(16)]
    OUTPUT = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0]

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="glb_skins_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _flattened_glb(self):
        """A rig whose chain was re-anchored: jnt_2 beside jnt_1 under RIG_GRP."""
        return _write_glb_file(
            os.path.join(self.tmp, "skeleton.glb"),
            {
                "asset": {"version": "2.0"},
                "scenes": [{"nodes": [0, 5, 7]}],
                "nodes": [
                    {"name": "LOOMS", "children": [1, 4]},
                    {"name": "RIG_GRP", "children": [2, 3]},
                    {"name": "jnt_1"},
                    {"name": "jnt_2"},
                    {"name": "LOOM", "mesh": 0, "skin": 0},
                    {"name": "OTHER", "children": [6]},
                    {"name": "jnt_a"},
                    {"name": "SOLO"},
                ],
                "meshes": [{"primitives": [{"attributes": {}}]}],
                "skins": [
                    {"name": "loom", "joints": [2, 3], "skeleton": 2},
                    {"name": "above", "joints": [2, 3], "skeleton": 0},
                    {"name": "rootless", "joints": [6, 7], "skeleton": 6},
                    {"name": "bare", "joints": [3]},
                ],
            },
        )

    def _bound_glb(self, skeleton=None, name="bound.glb"):
        """One skinned and one rigid mesh, both carrying JOINTS_0/WEIGHTS_0 the
        way FBX2glTF writes them, and two skins no node binds beside the real one.
        """
        parts, views, accessors = [], [], []

        def pack(values, fmt, type_, component=5126):
            payload = struct.pack(f"<{len(values)}{fmt}", *values)
            payload += b"\x00" * ((4 - len(payload) % 4) % 4)
            offset = sum(len(part) for part in parts)
            views.append(
                {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
            )
            parts.append(payload)
            width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[type_]
            accessors.append(
                {
                    "bufferView": len(views) - 1,
                    "componentType": component,
                    "count": len(values) // width,
                    "type": type_,
                }
            )
            return len(accessors) - 1

        joints, weights = [0] * 12, [1.0, 0.0, 0.0, 0.0] * 3
        skinned = {
            "POSITION": pack(
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], "f", "VEC3"
            ),
            "JOINTS_0": pack(joints, "B", "VEC4", 5121),
            "WEIGHTS_0": pack(weights, "f", "VEC4"),
        }
        rigid = {
            "POSITION": pack(self.RIGID, "f", "VEC3"),
            "JOINTS_0": pack(joints, "B", "VEC4", 5121),
            "WEIGHTS_0": pack(weights, "f", "VEC4"),
        }
        skin = {"joints": [1], "inverseBindMatrices": pack(self.IBM, "f", "MAT4")}
        if skeleton is not None:
            skin["skeleton"] = skeleton
        stray = {"joints": [1], "inverseBindMatrices": pack([1.0] * 16, "f", "MAT4")}
        sampler = {
            "input": pack([0.0, 1.0], "f", "SCALAR"),
            "output": pack(self.OUTPUT, "f", "VEC3"),
        }
        bin_chunk = b"".join(parts)
        return _write_glb_file(
            os.path.join(self.tmp, name),
            {
                "asset": {"version": "2.0"},
                "scenes": [{"nodes": [0]}],
                "nodes": [
                    {"name": "rig", "children": [1, 2, 3]},
                    {"name": "jnt"},
                    {"name": "skinned", "mesh": 0, "skin": 1},
                    {"name": "rigid", "mesh": 1},
                ],
                "meshes": [
                    {"primitives": [{"attributes": skinned}]},
                    {"primitives": [{"attributes": rigid}]},
                ],
                "skins": [{"joints": [0]}, skin, stray],
                "animations": [
                    {
                        "name": "Take 001",
                        "samplers": [sampler],
                        "channels": [
                            {"sampler": 0, "target": {"node": 1, "path": "translation"}}
                        ],
                    }
                ],
                "accessors": accessors,
                "bufferViews": views,
                "buffers": [{"byteLength": len(bin_chunk)}],
            },
            bin_chunk=bin_chunk,
        )

    @staticmethod
    def _floats(edit, accessor):
        view = edit.gltf["bufferViews"][edit.gltf["accessors"][accessor]["bufferView"]]
        start = view.get("byteOffset", 0)
        data = bytes(edit.bin_data[start : start + view["byteLength"]])
        return list(struct.unpack(f"<{len(data) // 4}f", data))

    def test_a_skeleton_that_is_not_its_joints_root_is_repointed(self):
        """glTF requires ``skin.skeleton`` to be a common root of the joints.

        FBX2glTF names a skin's first joint, which holds only while the joints
        hang in one chain under it. Measured on a production assembly
        (2026-09-14): the export's sheared-chain flatten re-anchors a rig's
        joints under its group, all 7 skins went invalid, and the Khronos
        validator rejected the file (SKIN_SKELETON_INVALID x7). three.js never
        reads the field, so no viewer showed it.
        """
        path = self._flattened_glb()
        self.assertEqual(MeshConvert.fix_glb_skin_skeletons(path), ["loom", "rootless"])
        with MeshConvert.open_glb(path) as edit:
            skins = edit.gltf["skins"]
        self.assertEqual(skins[0]["skeleton"], 1, "the joints' closest common root")
        self.assertEqual(skins[1]["skeleton"], 0, "an ancestor of that root is valid")
        self.assertNotIn("skeleton", skins[2], "no shared root: the field is optional")
        self.assertNotIn("skeleton", skins[3])
        self.assertEqual(MeshConvert.fix_glb_skin_skeletons(path), [])

    def test_skinning_data_no_node_binds_is_dropped(self):
        """Skins no node references, and JOINTS/WEIGHTS on a mesh no skinned
        node instantiates, mean nothing in glTF.

        Measured on a production assembly (2026-09-14): FBX2glTF wrote 255 of
        its 262 skins for no node, and JOINTS_0/WEIGHTS_0 on 836 meshes no
        skinned node instantiates -- 0.49 MB of vertex data, and a Khronos
        validator warning per node (NODE_SKINNED_MESH_WITHOUT_SKIN x1493).
        """
        path = self._bound_glb()
        with MeshConvert.open_glb(path) as edit:
            before = edit.gltf["buffers"][0]["byteLength"]
        self.assertEqual(
            MeshConvert.prune_glb_unused_skins(path),
            {"skins": 2, "attributes": 2, "bytes": 124},
        )
        with MeshConvert.open_glb(path) as edit:
            gltf = edit.gltf
            self.assertEqual(len(gltf["skins"]), 1)
            self.assertEqual(gltf["nodes"][2]["skin"], 0, "renumbered to the survivor")
            skinned, rigid = (m["primitives"][0]["attributes"] for m in gltf["meshes"])
            self.assertEqual(sorted(skinned), ["JOINTS_0", "POSITION", "WEIGHTS_0"])
            self.assertEqual(sorted(rigid), ["POSITION"])
            self.assertEqual(len(gltf["accessors"]), 7)
            self.assertEqual(gltf["buffers"][0]["byteLength"], before - 124)
            # What survived still reads its own bytes after the renumber + repack.
            self.assertEqual(self._floats(edit, rigid["POSITION"]), self.RIGID)
            ibm = gltf["skins"][0]["inverseBindMatrices"]
            self.assertEqual(self._floats(edit, ibm), self.IBM)
            output = gltf["animations"][0]["samplers"][0]["output"]
            self.assertEqual(self._floats(edit, output), self.OUTPUT)
        self.assertEqual(
            MeshConvert.prune_glb_unused_skins(path),
            {"skins": 0, "attributes": 0, "bytes": 0},
        )

    def test_a_draco_primitive_keeps_its_skin_attributes(self):
        """``KHR_draco_mesh_compression`` names a primitive's attributes a second
        time, inside the extension: dropping them from ``attributes`` alone
        would leave the extension naming attributes the primitive no longer has
        (``fbx_to_glb`` takes ``--draco`` through *extra_args*)."""
        path = self._bound_glb()
        with MeshConvert.open_glb(path) as edit:
            edit.gltf["meshes"][1]["primitives"][0]["extensions"] = {
                "KHR_draco_mesh_compression": {
                    "bufferView": 0,
                    "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                }
            }
            edit.dirty = True
        self.assertEqual(MeshConvert.prune_glb_unused_skins(path)["attributes"], 0)
        with MeshConvert.open_glb(path) as edit:
            rigid = edit.gltf["meshes"][1]["primitives"][0]["attributes"]
        self.assertEqual(sorted(rigid), ["JOINTS_0", "POSITION", "WEIGHTS_0"])

    def test_the_conversion_repairs_and_prunes_skins(self):
        converted = self._bound_glb(skeleton=2, name="converted.glb")
        dst = _convert_prebuilt(self.tmp, converted)
        with MeshConvert.open_glb(dst) as edit:
            skins = edit.gltf["skins"]
            rigid = edit.gltf["meshes"][1]["primitives"][0]["attributes"]
        self.assertEqual(len(skins), 1)
        self.assertEqual(skins[0]["skeleton"], 1, "a lone joint is its own root")
        self.assertEqual(sorted(rigid), ["POSITION"])


class TestGlbTangentRepair(unittest.TestCase):
    """The conversion session repairs every shipped tangent."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="glb_tangents_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_the_conversion_points_a_mirrored_shells_tangents_image_up(self):
        """REGRESSION (2026-09-23): FBX2glTF hands over ``w = +1`` on every
        vertex, and the preview and both Scene Exporters ship what this session
        leaves -- so the repair has to run inside it (``GlbTangents`` carries
        the rules and their own tests)."""
        converted = tangent_quads_glb(
            os.path.join(self.tmp, "converted.glb"),
            [("plain", 1.0), ("mirrored", 1.0)],
        )
        dst = _convert_prebuilt(self.tmp, converted)
        self.assertEqual([round(t[3]) for t in read_tangents(dst)], [1] * 4 + [-1] * 4)


class TestBinView(unittest.TestCase):
    """Which accessors' bytes a GLB's BIN chunk actually holds."""

    def _gltf(self):
        return {
            "buffers": [{"byteLength": 64}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 64}],
            "accessors": [{"bufferView": 0}],
        }

    def test_only_a_plain_view_on_the_embedded_buffer_is_in_the_bin(self):
        """Read at their offsets in the BIN, the others decode to plausible
        garbage rather than failing: a view on another buffer, an embedded
        buffer that names an external file instead, a view a compression
        extension decodes, and an accessor with no view at all."""
        gltf = self._gltf()
        self.assertIs(
            MeshConvert._bin_view(gltf, gltf["accessors"][0]), gltf["bufferViews"][0]
        )
        cases = {
            "another buffer": lambda g: g["bufferViews"][0].update(buffer=1),
            "external": lambda g: g["buffers"][0].update(uri="scene.bin"),
            "compressed": lambda g: g["bufferViews"][0].update(
                extensions={"EXT_meshopt_compression": {}}
            ),
            "no view": lambda g: g["accessors"][0].pop("bufferView"),
            "view out of range": lambda g: g["accessors"][0].update(bufferView=3),
        }
        for name, edit in cases.items():
            with self.subTest(name):
                gltf = self._gltf()
                edit(gltf)
                self.assertIsNone(MeshConvert._bin_view(gltf, gltf["accessors"][0]))


class TestCompactGlbAnimations(unittest.TestCase):
    """A clip pins the pose of what it does not animate; two keys say it."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="compact_anim_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ fixture
    def _scene(self, *, share_output=False, image=True, own_input=False):
        """One clip: a moving VEC3, a constant VEC3, and (optionally) an image.

        The two channels SHARE their time input, as a converter writes them,
        so collapsing one must not disturb the other. *own_input* gives the
        constant scale channel a time accessor of its own over the SAME view,
        the shape a clip rebuild writes (one input per channel).
        """
        times = struct.pack("<4f", 0.0, 1.0, 2.0, 3.0)
        moving = struct.pack("<12f", *[float(i) for i in range(12)])
        constant = struct.pack("<12f", *([1.0, 2.0, 3.0] * 4))
        blob = b"fakepng-payload!"
        chunks, views, offset = [], [], 0
        for payload in (times, moving, constant, blob):
            padded = payload + b"\x00" * ((4 - (len(payload) % 4)) % 4)
            views.append(
                {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
            )
            chunks.append(padded)
            offset += len(padded)
        gltf = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": offset}],
            "bufferViews": views,
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": 4,
                    "type": "SCALAR",
                    "min": [0.0],
                    "max": [3.0],
                },
                {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
                {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC3"},
            ],
            "animations": [
                {
                    "name": "Clip",
                    "samplers": [
                        {"input": 0, "output": 1, "interpolation": "LINEAR"},
                        {
                            "input": 0,
                            "output": 2 if not share_output else 1,
                            "interpolation": "LINEAR",
                        },
                        {
                            "input": 3 if own_input else 0,
                            "output": 2,
                            "interpolation": "LINEAR",
                        },
                    ],
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "translation"}},
                        {"sampler": 1, "target": {"node": 1, "path": "translation"}},
                        {"sampler": 2, "target": {"node": 2, "path": "scale"}},
                    ],
                }
            ],
            "nodes": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
        }
        if own_input:
            gltf["accessors"].append(dict(gltf["accessors"][0]))
        if image:
            gltf["images"] = [{"mimeType": "image/png", "bufferView": 3}]
            gltf["textures"] = [{"source": 0}]
        path = os.path.join(self.tmp, "scene.glb")
        return _write_glb_file(path, gltf, b"".join(chunks)), blob

    @staticmethod
    def _open(path):
        return MeshConvert._read_glb(path)

    # ------------------------------------------------------------------- tests
    def test_a_channel_that_never_moves_collapses_to_two_keys(self):
        path, _ = self._scene()

        result = MeshConvert.compact_glb_animations(path)

        self.assertEqual(result["channels"], 2)  # the constant one, in 2 channels
        edit = self._open(path)
        sampler = edit.gltf["animations"][0]["samplers"][1]
        values = MeshConvert._accessor_elements(edit, sampler["output"])
        self.assertEqual(len(values), 2, "a constant channel needs two keys")
        self.assertEqual(values[0], values[1])
        self.assertEqual(
            struct.unpack("<3f", values[0]),
            (1.0, 2.0, 3.0),
            "the value the original held on every frame must survive",
        )
        # Payload reclaimed, not file size: at fixture scale the JSON chunk
        # grows by more than a 48-byte curve is worth. The saving is a BIN
        # property, and that is what the pass reports.
        self.assertGreater(result["bytes"], 0)

    def test_a_collapsed_channel_leaves_no_accessor_nothing_reads(self):
        """The time input a collapsed channel stops reading goes with it.

        A clip rebuild gives every channel a time accessor of its own over one
        shared view, so rewiring a collapsed channel to the clip's two-key input
        strands the old accessor. Its view stays alive through the siblings, so
        no BIN byte leaks -- the JSON does: 15,843 dead accessors (1.59 MB) on
        the production 4K assembly, each one a validator UNUSED_OBJECT.
        """
        path, _ = self._scene(own_input=True)

        MeshConvert.compact_glb_animations(path)

        edit = self._open(path)
        gltf = edit.gltf
        self.assertEqual(
            MeshConvert._referenced_accessors(gltf),
            set(range(len(gltf["accessors"]))),
            "every accessor left in the file is read by something",
        )
        # Renumbered, not re-pointed: the collapsed channel still reads its two
        # keys and the value it held on every frame.
        sampler = gltf["animations"][0]["samplers"][2]
        self.assertEqual(len(MeshConvert._accessor_elements(edit, sampler["input"])), 2)
        values = MeshConvert._accessor_elements(edit, sampler["output"])
        self.assertEqual(struct.unpack("<3f", values[0]), (1.0, 2.0, 3.0))

    def test_a_moving_channel_is_left_byte_identical(self):
        path, _ = self._scene()
        original = MeshConvert._accessor_elements(self._open(path), 1)

        MeshConvert.compact_glb_animations(path)

        edit = self._open(path)
        sampler = edit.gltf["animations"][0]["samplers"][0]
        self.assertEqual(
            MeshConvert._accessor_elements(edit, sampler["output"]),
            original,
            "a channel that moves must not be touched at all",
        )

    def test_the_clip_keeps_its_duration(self):
        """Two keys at the clip's OWN ends, or the clip gets shorter."""
        path, _ = self._scene()

        MeshConvert.compact_glb_animations(path)

        edit = self._open(path)
        low, high = MeshConvert._animation_time_span(edit, edit.gltf["animations"][0])
        self.assertAlmostEqual(low, 0.0)
        self.assertAlmostEqual(high, 3.0)

    def test_an_output_a_moving_channel_shares_is_never_collapsed(self):
        """Samplers are shared; shrinking one would freeze the other's motion."""
        path, _ = self._scene(share_output=True)
        original = MeshConvert._accessor_elements(self._open(path), 1)

        MeshConvert.compact_glb_animations(path)

        edit = self._open(path)
        for index in (0, 1):
            sampler = edit.gltf["animations"][0]["samplers"][index]
            self.assertEqual(
                MeshConvert._accessor_elements(edit, sampler["output"]),
                original,
                "an output any moving channel reads must survive whole",
            )

    def test_image_payload_survives_the_repack(self):
        """Images name a bufferView DIRECTLY, not through an accessor.

        Sweeping only accessor references reads every image in the file as
        garbage -- 75.78 MB of it on the assembly this pass was sized against.
        """
        path, blob = self._scene()

        MeshConvert.compact_glb_animations(path)

        edit = self._open(path)
        view = edit.gltf["bufferViews"][edit.gltf["images"][0]["bufferView"]]
        start = view["byteOffset"]
        self.assertEqual(
            bytes(edit.bin_data[start : start + view["byteLength"]]),
            blob,
            "the image payload must survive, and at its new offset",
        )

    def test_a_collapsed_accessor_with_a_byte_offset_reads_its_new_view(self):
        """REGRESSION: an output accessor that starts INSIDE its view (a
        non-zero accessor byteOffset -- legal glTF, what a packer sharing one
        view between accessors writes) was re-pointed to its fresh two-key
        view with the offset left in place, so the collapsed channel read past
        the payload it had just been given. Added: 2026-09-13"""
        path, _blob = self._scene(image=False)
        with MeshConvert.open_glb(path) as edit:
            view = edit.gltf["bufferViews"][2]
            start = view["byteOffset"]
            payload = bytes(edit.bin_data[start : start + view["byteLength"]])
            new_view = MeshConvert._append_bin_views(edit, [bytes(8) + payload])[0]
            accessor = edit.gltf["accessors"][2]
            accessor["bufferView"] = new_view
            accessor["byteOffset"] = 8
            edit.dirty = True
        summary = MeshConvert.compact_glb_animations(path)
        self.assertGreaterEqual(summary["channels"], 1)
        elements = MeshConvert._accessor_elements(self._open(path), 2)
        self.assertEqual(elements, [struct.pack("<3f", 1.0, 2.0, 3.0)] * 2)

    def test_a_file_with_nothing_to_collapse_is_left_alone(self):
        path, _ = self._scene()
        MeshConvert.compact_glb_animations(path)
        settled = os.path.getsize(path)

        result = MeshConvert.compact_glb_animations(path)

        self.assertEqual(result, {"channels": 0, "accessors": 0, "bytes": 0})
        self.assertEqual(os.path.getsize(path), settled, "idempotent")

    def test_a_cubicspline_channel_is_refused(self):
        """CUBICSPLINE stores in-tangent, value and out-tangent per key.

        Two keys is SIX elements there, not two, so collapsing one writes a
        sampler whose output no longer matches its input -- a malformed file
        rather than a smaller one. This fixture's tangents and values are all
        identical, so the constancy test alone WOULD accept it.
        """
        times = struct.pack("<4f", 0.0, 1.0, 2.0, 3.0)
        # 4 keys x 3 elements (in-tangent, value, out-tangent) x VEC3
        spline = struct.pack("<36f", *([1.0, 2.0, 3.0] * 12))
        chunks, views, offset = [], [], 0
        for payload in (times, spline):
            padded = payload + b"\x00" * ((4 - (len(payload) % 4)) % 4)
            views.append(
                {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
            )
            chunks.append(padded)
            offset += len(padded)
        gltf = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": offset}],
            "bufferViews": views,
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": 4,
                    "type": "SCALAR",
                    "min": [0.0],
                    "max": [3.0],
                },
                {"bufferView": 1, "componentType": 5126, "count": 12, "type": "VEC3"},
            ],
            "animations": [
                {
                    "name": "Clip",
                    "samplers": [
                        {"input": 0, "output": 1, "interpolation": "CUBICSPLINE"}
                    ],
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "translation"}}
                    ],
                }
            ],
            "nodes": [{"name": "a"}],
        }
        path = _write_glb_file(
            os.path.join(self.tmp, "spline.glb"), gltf, b"".join(chunks)
        )

        result = MeshConvert.compact_glb_animations(path)

        self.assertEqual(result["channels"], 0)
        edit = self._open(path)
        self.assertEqual(
            edit.gltf["accessors"][1]["count"], 12, "the spline must be left whole"
        )

    def test_it_runs_inside_an_ALREADY_OPEN_session(self):
        """The form the conversion uses, and the one nothing else covered.

        `fbx_to_glb` calls this with its open `GlbEdit`, not a path -- so the
        pass appends views and REPACKS the BIN while other passes still hold
        that session. Two things have to survive it: the session stays usable
        afterwards, and payload no accessor names (an image reads its
        bufferView DIRECTLY) is still there -- an accessor-only liveness sweep
        would collect every image in the file, which is the trap `_compact_bin`
        was written around.
        """
        path, blob = self._scene(image=True)

        with MeshConvert.open_glb(path) as edit:
            result = MeshConvert.compact_glb_animations(edit)
            self.assertEqual(result["channels"], 2)
            # Usable after the repack: the session must not be left holding a
            # stale buffer or a view table the JSON no longer agrees with.
            self.assertEqual(len(edit.gltf["accessors"]), 4)

        edit = self._open(path)
        sampler = edit.gltf["animations"][0]["samplers"][1]
        self.assertEqual(
            len(MeshConvert._accessor_elements(edit, sampler["output"])), 2
        )
        image = edit.gltf["images"][0]
        view = edit.gltf["bufferViews"][image["bufferView"]]
        start = view["byteOffset"]
        self.assertEqual(
            bytes(edit.bin_data[start : start + view["byteLength"]]),
            b"fakepng-payload!",
            "the image payload must survive a repack no accessor references it through",
        )

    def test_a_view_reaching_past_the_bin_declines_the_whole_repack(self):
        """Slicing past the end of the BIN yields a short string in silence.

        The kept view would carry the `byteLength` it declared while holding
        fewer bytes, so a repack turns a file that arrived truncated into one
        that merely lies about its offsets -- no longer detectable as damaged.
        The fixture is that exact shape: an ORPHANED view (so the pass has
        real work and gets past its no-op check) beside a LIVE one whose
        declared extent runs past the buffer.
        """
        payload = struct.pack("<3f", 1.0, 2.0, 3.0)
        gltf = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": len(payload)}],
            "bufferViews": [
                # Live, but declares 64 bytes over a 12-byte buffer.
                {"buffer": 0, "byteOffset": 0, "byteLength": 64},
                # Orphaned: nothing names it, so the collector wants to run.
                {"buffer": 0, "byteOffset": 0, "byteLength": 4},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3"}
            ],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "nodes": [{"name": "a", "mesh": 0}],
        }
        path = _write_glb_file(os.path.join(self.tmp, "short.glb"), gltf, payload)
        before = os.path.getsize(path)

        edit = self._open(path)
        self.assertEqual(
            MeshConvert._compact_bin(edit), 0, "a short buffer must reclaim nothing"
        )

        self.assertEqual(
            len(edit.gltf["bufferViews"]), 2, "the view table must be left as found"
        )
        self.assertEqual(os.path.getsize(path), before, "and nothing may be rewritten")

    def test_a_file_with_no_animations_is_a_no_op(self):
        path = _write_glb_file(
            os.path.join(self.tmp, "static.glb"),
            {"asset": {"version": "2.0"}, "nodes": [{"name": "a"}]},
        )
        self.assertEqual(
            MeshConvert.compact_glb_animations(path),
            {"channels": 0, "accessors": 0, "bytes": 0},
        )

    def test_a_clip_left_with_nothing_to_collapse_mints_no_time_input(self):
        """A refusal is a property of the SAMPLER, so it can arrive late.

        Clip A reads a constant output through a LINEAR sampler, so the
        classifier accepts it; clip B then reads the SAME output through a
        CUBICSPLINE one and refuses it for both. Clip C keeps a collapsible
        output of its own, so the pass still runs. Clip A must not mint a
        two-key time input no sampler will read: an accessor keeps its
        bufferView alive, so the collector cannot reclaim it and every re-run
        adds another -- the leak `_release_animation_payload` exists to close,
        reopened one accessor at a time.
        """
        times = struct.pack("<4f", 0.0, 1.0, 2.0, 3.0)
        shared = struct.pack("<12f", *([1.0, 2.0, 3.0] * 4))  # constant, shared
        own = struct.pack("<12f", *([9.0, 9.0, 9.0] * 4))  # constant, clip C only
        chunks, views, offset = [], [], 0
        for payload in (times, shared, own):
            padded = payload + b"\x00" * ((4 - (len(payload) % 4)) % 4)
            views.append(
                {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
            )
            chunks.append(padded)
            offset += len(padded)
        accessors = [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 4,
                "type": "SCALAR",
                "min": [0.0],
                "max": [3.0],
            },
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC3"},
        ]

        def clip(name, output, interpolation):
            return {
                "name": name,
                "samplers": [
                    {"input": 0, "output": output, "interpolation": interpolation}
                ],
                "channels": [
                    {"sampler": 0, "target": {"node": 0, "path": "translation"}}
                ],
            }

        gltf = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": offset}],
            "bufferViews": views,
            "accessors": accessors,
            "animations": [
                clip("A", 1, "LINEAR"),
                clip("B", 1, "CUBICSPLINE"),
                clip("C", 2, "LINEAR"),
            ],
            "nodes": [{"name": "a"}],
        }
        path = _write_glb_file(
            os.path.join(self.tmp, "late_refusal.glb"), gltf, b"".join(chunks)
        )
        before = len(accessors)

        result = MeshConvert.compact_glb_animations(path)

        self.assertEqual(result["channels"], 1, "only clip C's channel collapses")
        edit = self._open(path)
        # One new accessor: clip C's time input. Clip A gets none -- its only
        # output was refused, so it has nothing to rewire.
        self.assertEqual(
            len(edit.gltf["accessors"]),
            before + 1,
            "a clip with nothing to collapse must mint no accessor",
        )
        self.assertEqual(
            edit.gltf["animations"][0]["samplers"][0]["input"],
            0,
            "clip A keeps the input it came in with",
        )
        self.assertEqual(
            edit.gltf["accessors"][1]["count"], 4, "the refused output stays whole"
        )
        # And every accessor still has a reader, which is what makes the pass
        # idempotent: a second run finds nothing to add.
        live = MeshConvert._referenced_accessors(edit.gltf)
        self.assertEqual(
            sorted(live),
            list(range(len(edit.gltf["accessors"]))),
            "no accessor may be left unreferenced",
        )


class TestDropGlbTextureFallbacks(unittest.TestCase):
    """KTX2 beside PNG doubles the texture budget the KTX2 pass exists to cut."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="drop_fallback_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _scene(self):
        ktx, png, lone = b"ktx2-payload!!!!", b"png-payload!!!!!", b"lone-png-bytes!!"
        chunks, views, offset = [], [], 0
        for payload in (ktx, png, lone):
            padded = payload + b"\x00" * ((4 - (len(payload) % 4)) % 4)
            views.append(
                {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
            )
            chunks.append(padded)
            offset += len(padded)
        gltf = {
            "asset": {"version": "2.0"},
            "extensionsUsed": ["KHR_texture_basisu"],
            "buffers": [{"byteLength": offset}],
            "bufferViews": views,
            "images": [
                {"mimeType": "image/ktx2", "bufferView": 0},
                {"mimeType": "image/png", "bufferView": 1},
                {"mimeType": "image/png", "bufferView": 2},
            ],
            "textures": [
                {
                    "source": 1,
                    "extensions": {"KHR_texture_basisu": {"source": 0}},
                },
                {"source": 2},
            ],
            "materials": [
                {
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                    "emissiveTexture": {"index": 1},
                },
            ],
        }
        return _write_glb_file(os.path.join(self.tmp, "t.glb"), gltf, b"".join(chunks))

    def test_the_fallback_beside_a_ktx2_twin_is_dropped(self):
        path = self._scene()

        result = MeshConvert.drop_glb_texture_fallbacks(path)

        self.assertEqual(result["textures"], 1)
        edit = MeshConvert._read_glb(path)
        self.assertNotIn("source", edit.gltf["textures"][0])
        self.assertEqual(
            edit.gltf["textures"][0]["extensions"]["KHR_texture_basisu"]["source"], 0
        )

    def test_a_texture_with_no_ktx2_twin_keeps_its_only_image(self):
        path = self._scene()

        MeshConvert.drop_glb_texture_fallbacks(path)

        edit = MeshConvert._read_glb(path)
        self.assertIn(
            "source",
            edit.gltf["textures"][1],
            "a lone PNG is the picture, not a fallback",
        )

    def test_basisu_becomes_required(self):
        """Without a fallback the file cannot be read without the extension."""
        path = self._scene()

        MeshConvert.drop_glb_texture_fallbacks(path)

        edit = MeshConvert._read_glb(path)
        self.assertIn("KHR_texture_basisu", edit.gltf.get("extensionsRequired") or [])

    def test_a_file_with_no_ktx2_is_left_alone(self):
        path = _write_glb_file(
            os.path.join(self.tmp, "plain.glb"),
            {
                "asset": {"version": "2.0"},
                "images": [{"uri": "a.png"}],
                "textures": [{"source": 0}],
            },
        )
        self.assertEqual(
            MeshConvert.drop_glb_texture_fallbacks(path),
            {"textures": 0, "images": 0, "bytes": 0},
        )


_FBX2GLTF = MeshConvert.resolve_binary(required=False, auto_install=False)


@unittest.skipUnless(
    _FBX2GLTF, "FBX2glTF is not installed; this test downloads nothing"
)
class TestGrayscaleMapsThroughTheConverter(unittest.TestCase):
    """A grayscale roughness or metallic map reaches the GLB with its values.

    FBX2glTF 0.13.1 packs a material's occlusion-roughness-metallic texture
    from each map's red, green and blue, and reads a channel the decoded image
    lacks as white. Measured on this Stingray PBS quad (roughness a 60..220
    ramp, metallic 8): stored RGB the packing carries both; stored grayscale --
    ``L``, ``LA``, 16-bit, a grayscale JPEG -- green and blue read 255 on every
    texel, roughness 1 and metallic 1, which renders black under a lightmap.
    Maya writes exactly that input: ``GameShader`` splits an ORM into ``L`` maps.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_gray_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _fbx(self, mode):
        """The same roughness and metallic values, stored as *mode*."""
        from PIL import Image

        rough = Image.new("L", (32, 32))
        rough.putdata([60 + (160 * x) // 31 for _y in range(32) for x in range(32)])
        metal = Image.new("L", (32, 32), 8)
        color = Image.new("RGB", (32, 32), (200, 150, 100))
        return build_stingray_fbx(
            os.path.join(self.tmp, f"orm_{mode}.fbx"),
            {
                "color": ("t_Base_Color.png", encoded(color)),
                "roughness": ("t_Roughness.png", encoded(rough.convert(mode))),
                "metallic": ("t_Metallic.png", encoded(metal.convert(mode))),
            },
        )

    @staticmethod
    def _orm(glb):
        """``{channel: (min, max)}`` of green and blue in the packed texture."""
        with MeshConvert.open_glb(glb) as edit:
            (material,) = edit.materials
            pbr = material["pbrMetallicRoughness"]
            image = edit.image_for_texture(pbr["metallicRoughnessTexture"]["index"])
            return {channel: edit.channel_extrema(image, channel) for channel in "GB"}

    def test_a_grayscale_map_packs_like_its_rgb_twin(self):
        options = dict(auto_install=False, prompt=False, lightmaps=False)
        reference = self._orm(MeshConvert.fbx_to_glb(self._fbx("RGB"), **options))
        self.assertEqual(reference, {"G": (60, 220), "B": (8, 8)}, "the RGB twin")
        self.assertEqual(
            self._orm(MeshConvert.fbx_to_glb(self._fbx("L"), **options)), reference
        )

    def test_the_converter_alone_still_packs_grayscale_as_white(self):
        """The premise, pinned against the binary with none of the passes.

        Red after a converter upgrade means the expansion ``fbx_to_glb`` runs
        first may be retired.
        """
        base = os.path.join(self.tmp, "direct")
        media = os.path.join(self.tmp, "media")
        os.makedirs(media)
        subprocess.run(
            [_FBX2GLTF, "-i", self._fbx("L"), "-o", base, "--binary"]
            + ["--fbx-temp-dir", media],
            check=True,
            capture_output=True,
            timeout=120,
        )
        self.assertEqual(self._orm(base + ".glb"), {"G": (255, 255), "B": (255, 255)})


class TestFacadeLayout(unittest.TestCase):
    """The 2026-09-26 split kept every ``MeshConvert`` name reaching the SAME
    object: the container moved to ``glb/edit.py`` and the passes into one
    private mixin per job. Pinned so a later move cannot turn a binding into a
    copy (a copied table drifts; a re-wrapped helper escapes a patch).
    Added: 2026-09-26"""

    def test_container_names_are_the_containers_own(self):
        from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit

        self.assertIs(MeshConvert.GlbEdit, GlbEdit)
        for name in (
            "ACCESSOR_COMPONENT_TYPES",
            "ACCESSOR_TYPE_COUNT",
            "TEXTURE_CONTAINER_EXTENSIONS",
        ):
            self.assertIs(getattr(MeshConvert, name), getattr(GlbEdit, name), name)
        for private, public in (
            ("_read_glb", "read"),
            ("_write_glb", "write"),
            ("_append_bin_views", "append_bin_views"),
            ("_compact_bin", "compact_bin"),
            ("_relocate_embedded_images", "relocate_embedded_images"),
            ("_bin_view", "bin_view"),
            ("_accessor_elements", "accessor_elements"),
            ("_drop_orphaned_accessors", "drop_orphaned_accessors"),
            ("_release_animation_payload", "release_animation_payload"),
        ):
            self.assertEqual(
                getattr(MeshConvert, private), getattr(GlbEdit, public), private
            )

    def test_passes_resolve_through_the_facade(self):
        """A pass reaches the others through ``cls``, so patching the facade
        reaches a call made from another job's module."""

        class Reached(Exception):
            pass

        seen = []

        def stop(glb):
            seen.append(glb)
            raise Reached

        with patch.object(MeshConvert, "open_glb", side_effect=stop):
            with self.assertRaises(Reached):
                MeshConvert.check_glb_materials("x.glb")
        self.assertEqual(seen, ["x.glb"])


class TestDecodeGlbTextures(unittest.TestCase):
    """A copy of a GLB with its KTX2 images decoded to PNG.

    REGRESSION (2026-10-09): Blender 5.1's glTF importer refuses a file that
    REQUIRES ``KHR_texture_basisu`` ("Extension KHR_texture_basisu is not
    available on this addon version") -- every KTX2 deliverable -- so tentacle's
    Import glTF failed in Blender, and the Maya pull through headless Blender
    with it. The decoded copy is what both now hand to Blender.
    """

    def setUp(self):
        from pythontk.file_utils.temp_artifacts import TempArtifacts
        from test_gltf2fbx import scene_builder

        self.temp = TempArtifacts("test_decode_glb_textures", policy="scoped")
        self.addCleanup(self.temp.cleanup)
        self.builder = scene_builder(animated=False)
        self.src = self.temp.path(extension=".glb")
        self.dst = self.temp.path(extension=".glb")

    def _ktx2_only(self):
        """Texture 0 reaches its image through the extension alone."""
        from PIL import Image

        ktx2 = self.temp.path(extension=".ktx2")
        Ktx2Encoder().encode(
            Image.new("RGB", (8, 8), (200, 10, 10)), ktx2, codec="ETC1S"
        )
        with open(ktx2, "rb") as fh:
            index = self.builder.image(fh.read(), "albedo.ktx2")
        self.builder.gltf["images"][index]["mimeType"] = "image/ktx2"
        self.builder.gltf["textures"][0] = {
            "extensions": {"KHR_texture_basisu": {"source": index}}
        }
        self.builder.gltf["extensionsUsed"] = ["KHR_texture_basisu"]
        self.builder.gltf["extensionsRequired"] = ["KHR_texture_basisu"]
        return self.builder.write_glb(self.src)

    def test_a_file_every_importer_reads_needs_no_copy(self):
        self.builder.write_glb(self.src)
        self.assertIsNone(MeshConvert.decode_glb_textures(self.src, self.dst))
        self.assertFalse(os.path.exists(self.dst))

    def test_a_core_fallback_beside_the_extension_needs_no_copy(self):
        self.builder.gltf["textures"][0]["extensions"] = {
            "KHR_texture_basisu": {"source": 1}
        }
        self.builder.write_glb(self.src)
        self.assertIsNone(MeshConvert.decode_glb_textures(self.src, self.dst))

    @unittest.skipUnless(
        Ktx2Encoder.available() and Ktx2Encoder.resolve_ktx(), "KTX-Software absent"
    )
    def test_ktx2_only_textures_are_decoded_in_place(self):
        from pythontk.file_utils.mesh_convert.glb.reader import GlbReader

        src = self._ktx2_only()
        before = open(src, "rb").read()
        out = MeshConvert.decode_glb_textures(src, self.dst)
        self.assertEqual(out, self.dst)
        reader = GlbReader.load(out)
        self.assertEqual(reader.extensions(), ([], []))
        texture = reader.gltf["textures"][0]
        self.assertNotIn("extensions", texture)
        image = reader.gltf["images"][texture["source"]]
        self.assertEqual(image["mimeType"], "image/png")
        data = MeshConvert.GlbEdit.read(out).image_bytes(image)
        self.assertEqual(bytes(data[:8]), b"\x89PNG\r\n\x1a\n")
        # Geometry untouched; the source never is.
        self.assertEqual(reader.counts()["accessors"], len(self.builder.accessors))
        self.assertEqual(open(src, "rb").read(), before)

    def test_without_the_tool_nothing_is_left_behind(self):
        self.builder.gltf["textures"][0] = {
            "extensions": {"KHR_texture_basisu": {"source": 0}}
        }
        self.builder.write_glb(self.src)
        with patch.object(Ktx2Encoder, "resolve_ktx", return_value=None):
            with self.assertRaises(FileNotFoundError):
                MeshConvert.decode_glb_textures(self.src, self.dst)
        self.assertFalse(os.path.exists(self.dst))


class TestGlbHighlightProbes(unittest.TestCase):
    """The bake's lights drawn into per-receiver copies of its probe (``_highlights.py``).

    A room box 8 x 4 x 8 m with its probe at head height: a baked floor on the
    box's bottom face (the shell), a baked table top inside, and an unbaked
    asset of two materials whose bounds overlap -- plus a third unbaked part
    across the room.
    """

    BOX = {"min": [-4.0, 0.0, -4.0], "max": [4.0, 4.0, 4.0]}
    PROBE = {"position": [0.0, 1.8, 2.0], "box": BOX}

    def setUp(self):
        from pythontk import TempArtifacts

        artifacts = TempArtifacts("mesh_convert_highlights", policy="scoped")
        self.tmp = artifacts.dir_path()
        self.addCleanup(artifacts.cleanup)

    @staticmethod
    def _quad(center, size, normal):
        """Corners of an axis-aligned quad facing *normal* (+Y or +Z)."""
        cx, cy, cz = center
        sx, sy = size[0] / 2, size[1] / 2
        if normal == (0, 1, 0):
            return [(cx - sx, cy, cz + sy), (cx + sx, cy, cz + sy), (cx + sx, cy, cz - sy), (cx - sx, cy, cz - sy)]
        return [(cx - sx, cy - sy, cz), (cx + sx, cy - sy, cz), (cx + sx, cy + sy, cz), (cx - sx, cy + sy, cz)]

    def _room(self, quads, extra_nodes=(), uvs=False, moves=None):
        """A GLB of *quads* ``(node, material, center, size, normal)``, each its own mesh.

        *uvs* gives each quad two UV sets (the second the bake's); *moves*
        ``{node: [(seconds, translation), ...]}`` is one take keying them."""
        materials = sorted({q[1] for q in quads})
        blob, views, accessors, nodes, meshes = bytearray(), [], [], [], []

        def add(values, kind, fmt, component):
            data = struct.pack(f"<{len(values)}{fmt}", *values)
            data += b"\x00" * (-len(data) % 4)
            views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
            blob.extend(data)
            accessor = {"bufferView": len(views) - 1, "componentType": component, "count": len(values) // {"VEC3": 3, "VEC2": 2, "SCALAR": 1}[kind], "type": kind}
            accessors.append(accessor)
            return len(accessors) - 1

        for name, material, center, size, normal in quads:
            corners = self._quad(center, size, normal)
            positions = [c for v in corners for c in v]
            pos = add(positions, "VEC3", "f", 5126)
            accessors[pos]["min"] = [min(positions[i::3]) for i in range(3)]
            accessors[pos]["max"] = [max(positions[i::3]) for i in range(3)]
            nrm = add([c for _ in corners for c in normal], "VEC3", "f", 5126)
            idx = add([0, 1, 2, 0, 2, 3], "SCALAR", "H", 5123)
            attributes = {"POSITION": pos, "NORMAL": nrm}
            if uvs:
                attributes["TEXCOORD_0"] = add([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], "VEC2", "f", 5126)
                attributes["TEXCOORD_1"] = add([0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9], "VEC2", "f", 5126)
            meshes.append({"primitives": [{"attributes": attributes, "indices": idx, "material": materials.index(material)}]})
            nodes.append({"name": name, "mesh": len(meshes) - 1})
        nodes.extend(extra_nodes)
        gltf = {
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": list(range(len(nodes)))}],
            "scene": 0,
            "nodes": nodes,
            "meshes": meshes,
            "materials": [{"name": m} for m in materials],
            "accessors": accessors,
            "bufferViews": views,
        }
        if moves:
            samplers, channels = [], []
            for node, keys in moves.items():
                times = [float(t) for t, _ in keys]
                time = add(times, "SCALAR", "f", 5126)
                accessors[time]["min"], accessors[time]["max"] = [min(times)], [max(times)]
                value = add([float(c) for _, v in keys for c in v], "VEC3", "f", 5126)
                samplers.append({"input": time, "output": value, "interpolation": "LINEAR"})
                target = {"node": [n["name"] for n in nodes].index(node), "path": "translation"}
                channels.append({"sampler": len(samplers) - 1, "target": target})
            gltf["animations"] = [{"name": "Take 001", "samplers": samplers, "channels": channels}]
        gltf["buffers"] = [{"byteLength": len(blob)}]
        return _write_glb_file(os.path.join(self.tmp, "room.glb"), gltf, bytes(blob))

    def _scene(self, extra_nodes=(), **room):
        return self._room(
            [
                ("FLOOR", "floorMat", (0.0, 0.0, 0.0), (8.0, 8.0), (0, 1, 0)),
                ("TABLE", "tableMat", (0.0, 0.9, 0.0), (1.6, 0.8), (0, 1, 0)),
                ("LAMP_BODY", "lampMat", (0.3, 1.2, 0.0), (0.3, 0.3), (0, 0, 1)),
                ("LAMP_GLASS", "glassMat", (0.35, 1.25, 0.01), (0.2, 0.2), (0, 0, 1)),
                ("STOOL", "stoolMat", (-3.0, 0.5, 3.0), (0.4, 0.4), (0, 1, 0)),
            ],
            extra_nodes,
            **room,
        )

    BAKED = ["floorMat", "tableMat"]
    # Over the table, and one on the ceiling (the box's top face).
    HANGING = {"name": "OVER", "type": "point", "position": [0.0, 1.6, 0.3], "color": [1.0, 1.0, 1.0], "intensity": 2.0}
    CEILING = {"name": "PANEL", "type": "rect", "position": [0.0, 4.0, 0.0], "direction": [0.0, -1.0, 0.0],
               "right": [1.0, 0.0, 0.0], "color": [1.0, 1.0, 1.0], "intensity": 3.0, "width": 1.0, "height": 0.5}

    @staticmethod
    def _as_recorded(light):
        """A host's light as the record states it: its name is the scene's
        bookkeeping, which no reader of the record uses (the hosts' producers
        leave it out)."""
        return {k: v for k, v in light.items() if k != "name"}

    def _layers(self, lights, glb=None, **kw):
        return MeshConvert.highlight_probe_layers(
            glb or self._scene(), lights, probe=dict(self.PROBE, **kw.pop("probe", {})),
            width=128, baked=self.BAKED, **kw
        )

    def test_the_shell_reads_the_probe_as_captured(self):
        """The floor lies on the box's bottom face: lights placed for the bench
        and read from it glowed where no light shines, so it gets no copy."""
        layers = self._layers([self.HANGING])
        materials = [m for layer in layers for m in layer["materials"]]
        self.assertNotIn("floorMat", materials)
        self.assertIn("tableMat", materials)

    def test_an_unbaked_assets_parts_share_one_copy_and_a_far_part_gets_its_own(self):
        """Everything an unbaked object shows comes from the probe, so it gets a
        copy placed for it: one per asset -- parts whose bounds meet -- not one
        per material, and a part across the room one of its own."""
        layers = self._layers([self.HANGING])
        unbaked = sorted(sorted(layer["materials"]) for layer in layers if layer["receivers"] == "unbaked")
        self.assertEqual(unbaked, [["glassMat", "lampMat"], ["stoolMat"]])
        (interior,) = [layer for layer in layers if layer["receivers"] == "interior"]
        self.assertEqual(interior["materials"], ["tableMat"])
        self.assertEqual(interior["image"].shape, (64, 128, 3))
        self.assertTrue(all(layer["image"].shape == (32, 64, 3) for layer in layers if layer["receivers"] == "unbaked"))

    def test_a_light_on_the_box_is_already_true_and_keeps_its_own_level(self):
        """A light lying on the box's face is where box projection reads it from
        any point, so drawn there it gives every surface its own light: the
        scale that matches the two is exactly 1."""
        meshes = MeshConvert._highlight_meshes(MeshConvert._read_glb(self._scene()))
        table = [m for m in meshes if m["material"] == "tableMat"]
        pts, nrm = MeshConvert._highlight_samples(table)
        lo, hi = (np.asarray(self.BOX[k], float) for k in ("min", "max"))
        eye = np.asarray(self.PROBE["position"], float)
        for irradiance in (True, False):
            _, scale = MeshConvert._highlight_placement(
                dict(self.CEILING, type="point", radius=0.0), pts, nrm, eye, lo, hi, irradiance=irradiance
            )
            self.assertAlmostEqual(scale, 1.0, places=6)

    @staticmethod
    def _sheet(x0, cuts, size=1.0):
        """A flat +Y sheet *size* metres square from x = *x0*, cut into
        ``cuts x cuts`` cells of two triangles each, as ``_highlight_meshes``
        returns a mesh."""
        step = size / cuts
        corners = []
        for i in range(cuts):
            for j in range(cuts):
                a = (x0 + i * step, 0.0, j * step)
                b = (x0 + (i + 1) * step, 0.0, j * step)
                c = (x0 + (i + 1) * step, 0.0, (j + 1) * step)
                d = (x0 + i * step, 0.0, (j + 1) * step)
                corners += [(a, c, b), (a, d, c)]
        triangles = np.asarray(corners, float)
        normals = np.zeros_like(triangles)
        normals[..., 1] = 1.0
        return {"material": "sheetMat", "triangles": triangles, "normals": normals}

    def test_samples_follow_area_not_tessellation_and_stay_bounded(self):
        """REGRESSION (2026-10-19): every triangle drew one sample on top of its
        area's share, so a dense mesh drew at least one per triangle -- a
        square rect light's field over a million-triangle interior cost 30 s
        per group -- and its samples crowded where it was finely cut rather
        than where its area is, pulling an irradiance centroid toward the
        tessellation. Two square metres, one cut into 2 triangles and one into
        20,000, share the samples evenly; a 1000 m^2 floor stops at the cap."""
        coarse, fine = self._sheet(0.0, 1), self._sheet(2.0, 100)
        points, _ = MeshConvert._highlight_samples([coarse, fine])
        self.assertLess(len(points), 2 * MeshConvert.HIGHLIGHT_MIN_SAMPLES)
        self.assertAlmostEqual(float((points[:, 0] >= 2.0).mean()), 0.5, delta=0.05)
        floor, _ = MeshConvert._highlight_samples([self._sheet(0.0, 1, size=31.7)])
        self.assertLess(len(floor), 1.05 * MeshConvert._HIGHLIGHT_MAX_SAMPLES)
        self.assertGreater(len(floor), 0.95 * MeshConvert._HIGHLIGHT_MAX_SAMPLES)

    def test_samples_are_the_geometrys_not_its_storage_order(self):
        """REGRESSION (2026-10-19): samples were drawn in the order the triangles
        were stored, so the same surfaces stored another way drew another set --
        an FBX-only export's copies (a build of the FBX) differed from the GLB
        export's (whose pipeline strips takes and downsizes first, which moves
        FBX2glTF's triangle order) by 30% of a copy's texels, the lights drawn
        a little elsewhere. Shuffled, corners rotated, or split across meshes,
        one surface draws one set of samples."""
        sheet = self._sheet(0.0, 20)
        points, normals = MeshConvert._highlight_samples([sheet])
        rng = np.random.default_rng(3)
        order = rng.permutation(len(sheet["triangles"]))
        rotate = rng.integers(0, 3, len(order))
        idx = (rotate[:, None] + np.arange(3)[None]) % 3
        moved = {
            "material": "sheetMat",
            "triangles": np.take_along_axis(sheet["triangles"][order], idx[..., None], 1),
            "normals": np.take_along_axis(sheet["normals"][order], idx[..., None], 1),
        }
        half = len(order) // 2
        split = [
            {k: (v[:half] if k != "material" else v) for k, v in moved.items()},
            {k: (v[half:] if k != "material" else v) for k, v in moved.items()},
        ]
        for meshes in ([moved], split[::-1]):
            got_points, got_normals = MeshConvert._highlight_samples(meshes)
            np.testing.assert_allclose(got_points, points, atol=1e-9)
            np.testing.assert_allclose(got_normals, normals, atol=1e-9)

    def _focal(self, light, box=BOX):
        """The interior's focal point for *light*, as the pass places it (with
        no box there is no shell: the floor joins the interior)."""
        meshes = MeshConvert._highlight_meshes(MeshConvert._read_glb(self._scene()))
        lo, hi = (np.asarray(box[k], float) for k in ("min", "max")) if box else (None, None)
        groups = MeshConvert._highlight_groups(meshes, set(self.BAKED), lo, hi)
        (interior,) = [g for g in groups if g["receivers"] == "interior"]
        F, _ = MeshConvert._highlight_placement(
            light, interior["points"], interior["normals"],
            np.asarray(self.PROBE["position"], float), lo, hi, irradiance=False,
        )
        return F, lo, hi

    def _assert_peak_toward(self, image, direction, texels=2):
        img = image.mean(2)
        row, col = np.unravel_index(int(np.argmax(img)), img.shape)
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        u = 0.5 + math.atan2(d[2], d[0]) / (2 * math.pi)
        v = 0.5 + math.asin(d[1]) / math.pi
        h, w = img.shape
        du = abs(col + 0.5 - u * w)
        self.assertLessEqual(abs(row + 0.5 - (1.0 - v) * h), texels, (row, col))
        self.assertLessEqual(min(du, w - du), texels, (row, col))

    def test_a_hanging_light_is_drawn_where_box_projection_reads_it_from_its_focal_point(self):
        """The copy shows the light along the ray from its focal point F through
        the light to the box: the probe reads that box point from where it was
        captured, so a reflection from F sees the light true."""
        (layer,) = [entry for entry in self._layers([self.HANGING]) if entry["receivers"] == "interior"]
        F, lo, hi = self._focal(self.HANGING)
        toward = np.asarray(self.HANGING["position"]) - F
        B = MeshConvert._box_exit(F, (toward / np.linalg.norm(toward))[None], lo, hi)[0]
        self.assertLess(F[1], 1.0, "F is on the table")
        self._assert_peak_toward(layer["image"], B - np.asarray(self.PROBE["position"]))

    def test_a_distant_probe_draws_the_light_along_its_own_direction(self):
        """A probe with no box is read as distant: its copy shows the light in
        the direction F sees it."""
        layers = self._layers([self.HANGING], probe={"box": None})
        (layer,) = [entry for entry in layers if entry["receivers"] == "interior"]
        F, _, _ = self._focal(self.HANGING, box=None)
        self._assert_peak_toward(layer["image"], np.asarray(self.HANGING["position"]) - F)

    def test_a_malformed_light_is_left_out_of_the_copies(self):
        """REGRESSION (2026-10-19): a light a caller wrote wrong (an intensity
        that is no number, a position of two) raised out of the public pass --
        a look capture's ``lights`` request failed whole -- where the lightmap
        pass leaves such a light out (``lights_in_metres``). It is left out
        here too, and the rest are drawn as they would be alone."""
        bad = [
            dict(self.HANGING, name="WORD", intensity="bright"),
            dict(self.HANGING, name="FLAT", position=[0.0, 1.6]),
        ]
        alone = self._layers([self.HANGING])
        layers = self._layers(bad + [self.HANGING])
        self.assertEqual([entry["materials"] for entry in layers], [entry["materials"] for entry in alone])
        for layer, own in zip(layers, alone):
            np.testing.assert_array_equal(layer["image"], own["image"])

    def test_a_light_just_off_a_wall_is_drawn(self):
        """REGRESSION (2026-10-19): a light between its receivers' focal point
        and a nearby wall reads, from the probe's point, many texels wide -- a
        texel scaled past 90 degrees, whose tangent is negative, sized its
        sphere 0, and the zero solid angle raised ZeroDivisionError out of the
        drawing: every copy of the file was dropped ("Probe highlights not
        drawn"). Such a light is drawn at the largest size a light is."""
        lo, hi = (np.asarray(self.BOX[k], float) for k in ("min", "max"))
        C = np.asarray(self.PROBE["position"], float)
        light = {"name": "WALL", "type": "point", "position": [3.95, 1.0, 0.0],
                 "color": [1.0, 1.0, 1.0], "intensity": 2.0}
        F = np.array([3.917, 1.0, 0.0])
        dist = float(np.linalg.norm(np.asarray(light["position"]) - F))
        self.assertAlmostEqual(MeshConvert._proxy_radius(light, F, C, lo, hi, 128), 0.5 * dist)
        layer = MeshConvert._highlight_layer([light], [F], [1.0], C, lo, hi, 128)
        self.assertGreater(float(layer.max()), 0.0)

    def test_a_spot_whose_cone_misses_the_receivers_draws_nothing_for_them(self):
        spot = dict(self.HANGING, type="spot", direction=[0.0, 1.0, 0.0], angle=0.3, penumbra=0.2)
        layers = self._layers([spot])
        self.assertEqual([entry for entry in layers if entry["receivers"] == "interior"], [])

    def test_the_start_camera_is_the_eye(self):
        glb = self._scene(extra_nodes=[{"name": "set:user_pos", "translation": [0.5, 1.7, 3.0]}])
        with MeshConvert.open_glb(glb) as edit:
            self.assertEqual(MeshConvert._highlight_eye(edit), [0.5, 1.7, 3.0])
        with MeshConvert.open_glb(self._scene()) as edit:
            self.assertIsNone(MeshConvert._highlight_eye(edit))

    def test_lights_in_metres_carries_metres_and_candela(self):
        """The record's lights arrive in scene units with ``unit_scale``: a point
        or spot light's intensity is candela over metres (its irradiance falls
        over the unit squared), a rect's radiance is as it is."""
        lights = MeshConvert.lights_in_metres([
            {"name": "KEY", "type": "spot", "position": [100.0, 300.0, 0.0], "direction": [0.0, -1.0, 0.0],
             "color": [1.0, 0.8, 0.6], "intensity": 40000.0, "angle": 0.5, "penumbra": 0.2, "radius": 5.0, "unit_scale": 0.01},
            {"name": "BAR", "type": "rect", "position": [0.0, 120.0, 250.0], "direction": [0.0, 0.0, -1.0],
             "right": [1.0, 0.0, 0.0], "intensity": 2.5, "width": 250.0, "height": 16.0, "unit_scale": 0.01},
            {"name": "BROKEN", "type": "spot", "intensity": 1.0},
        ])
        self.assertEqual([light["name"] for light in lights], ["KEY", "BAR"])
        key, bar = lights
        self.assertEqual(key["position"], [1.0, 3.0, 0.0])
        self.assertAlmostEqual(key["intensity"], 4.0)
        self.assertAlmostEqual(key["radius"], 0.05)
        self.assertEqual((bar["width"], bar["height"], bar["intensity"]), (2.5, 0.16, 2.5))

    def test_lights_in_metres_takes_a_lights_own_unit_then_the_records(self):
        """A record states its unit once (``LightmapRecord``, version 2) while
        a host's ``specular_lights`` carry one each: a light's own wins, then
        the record's, then metres -- and a unit that is no length leaves the
        light out. Added: 2026-10-20"""
        bulb = {"name": "BULB", "type": "point", "position": [0.0, 200.0, 0.0], "intensity": 100.0}
        (record,) = MeshConvert.lights_in_metres([bulb], unit_scale=0.1)
        self.assertEqual(record["position"], [0.0, 20.0, 0.0])
        self.assertAlmostEqual(record["intensity"], 1.0)
        (own,) = MeshConvert.lights_in_metres([dict(bulb, unit_scale=0.01)], unit_scale=0.1)
        self.assertEqual(own["position"], [0.0, 2.0, 0.0])
        self.assertAlmostEqual(own["intensity"], 0.01)
        (metres,) = MeshConvert.lights_in_metres([bulb])
        self.assertEqual((metres["position"], metres["intensity"]), ([0.0, 200.0, 0.0], 100.0))
        self.assertIsNone(MeshConvert.lights_in_metres([bulb], unit_scale=0.0))
        self.assertIsNone(MeshConvert.lights_in_metres([dict(bulb, unit_scale=-1.0)], unit_scale=0.1))

    def _lit_scene(self):
        """The scene GLB through ``apply_glb_lightmaps``, and the record it carried
        (the floor and table baked, the probe and the hanging light)."""
        cv2 = _require_cv2()

        cv2.imwrite(os.path.join(self.tmp, "room_Lightmap.exr"), np.full((8, 8, 3), 0.5, np.float32))
        probe = np.full((32, 64, 3), 0.2, np.float32)
        cv2.imwrite(os.path.join(self.tmp, "room_Probe.exr"), probe)
        manifest = LightmapRecord.payload(
            [
                _baked(
                    {
                        "name": n,
                        "map": "room_Lightmap.exr",
                        "scaleOffset": [1.0, 1.0, 0.0, 0.0],
                        "hierarchy": [n],
                    }
                )
                for n in ("FLOOR", "TABLE")
            ],
            probe={
                "map": "room_Probe.exr",
                "position": self.PROBE["position"],
                "box": [self.BOX["min"], self.BOX["max"]],
            },
            lights=[self._as_recorded(self.HANGING)],
            unit_scale=1.0,
        )
        carrier = {"name": "data_export", "extras": {"fromFBX": {"userProperties": {
            "lightmaps": {"type": "eFbxString", "value": json.dumps(manifest)}}}}}
        glb = self._scene(extra_nodes=[carrier])
        with MeshConvert.open_glb(glb) as edit:  # the bake's UV set, as a conversion leaves it
            for mesh in edit.gltf["meshes"]:
                for prim in mesh["primitives"]:
                    prim["attributes"]["TEXCOORD_1"] = prim["attributes"]["POSITION"]
            edit.dirty = True
        self.assertTrue(MeshConvert.apply_glb_lightmaps(glb, search_dirs=[self.tmp]))
        return glb, manifest

    def _deliverable(self, manifest):
        """The FBX twin of the scene: a carrier holding *manifest*."""
        from test_fbx_media import build_carrier_fbx

        return build_carrier_fbx(os.path.join(self.tmp, "room.fbx"), {"lightmaps": manifest})

    @staticmethod
    def _textured_probe(h=64, w=128):
        """A probe that is not flat: a sky gradient and a bright window (RGB, rows top-down)."""
        v = (np.arange(h)[:, None] + 0.5) / h
        u = (np.arange(w)[None, :] + 0.5) / w
        base = 0.05 + 0.6 * (1.0 - v) * (0.5 + 0.5 * np.cos(2 * np.pi * u))
        image = np.stack([base, base * 0.9, base * 0.8], -1).astype(np.float32)
        image[h // 5 : h // 3, w // 3 : w // 2] = (6.0, 5.5, 5.0)
        return image

    def _delivery(self, lights=None, pixels=None, baked=("FLOOR", "TABLE"), **room):
        """``(glb, record)``: the scene's geometry as an export's GLB holds it,
        and the lightmaps record its FBX carries -- *baked* lightmapped, the
        probe (its EXR written beside the scene: *pixels*, else a flat 0.2)
        and *lights* (default the hanging one)."""
        cv2 = _require_cv2()

        pixels = np.full((32, 64, 3), 0.2, np.float32) if pixels is None else pixels
        cv2.imwrite(os.path.join(self.tmp, "room_Probe.exr"), np.ascontiguousarray(pixels[:, :, ::-1]))
        record = LightmapRecord.payload(
            [
                {"name": n, "map": "room_Lightmap.exr", "scaleOffset": [1.0, 1.0, 0.0, 0.0], "hierarchy": [n]}
                for n in baked
            ],
            probe={
                "map": "room_Probe.exr",
                "position": self.PROBE["position"],
                "box": [self.BOX["min"], self.BOX["max"]],
            },
            lights=[self._as_recorded(light) for light in (lights or [self.HANGING])],
            unit_scale=1.0,
        )
        quads = room.pop("quads", None)
        return (self._room(quads, **room) if quads else self._scene(**room)), record

    @staticmethod
    def _radiance(raw):
        """Radiance bytes decoded as RGB floats, rows top-down -- what three.js' RGBELoader reads."""
        import cv2

        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        return np.asarray(image, np.float32)[:, :, 2::-1]

    def _shipped(self, fbx):
        """The FBX's record and each file it embeds now, ``{name: decoded RGB}``."""
        out = os.path.join(self.tmp, "extracted")
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        files = {}
        for name, path in FbxMedia.extract_dependencies(fbx, out).items():
            with open(path, "rb") as fh:
                files[name] = fh.read()
        return FbxMedia.record(fbx, "lightmaps"), files

    @staticmethod
    def _viewer_copy(probe, layer):
        """A copy as the viewer built one from the probe and a layer
        (``kernel/probe.js`` ``addProbe``): the probe box-filtered down to the
        layer's size by a whole factor, plus the layer."""
        f = probe.shape[1] // layer.shape[1]
        h, w = layer.shape[:2]
        return probe.reshape(h, f, w, f, 3).mean((1, 3)) + layer

    #: The most a texel of a complete copy may differ from the viewer's sum of
    #: the probe and its layer, as a share of its brightest channel: one step
    #: of an RGBE channel's 8-bit mantissa, at most 2^-7 of the texel's peak
    #: (measured 2026-11-07 against the copies the GLB build drew before
    #: they were complete: 0.52% at worst).
    RGBE_TOLERANCE = 2.0**-7

    def test_an_fbx_deliverable_ships_its_probe_lit_and_its_copies_complete(self):
        """Lightmaps v3 (2026-11-07): the delivered probe is the environment
        with the bake's lights drawn in as its capture point sees them -- a
        reader ignoring the copies still sees highlights -- and each copy is a
        COMPLETE probe for its group of receivers, the probe plus that group's
        lights, used in the probe's place. Each rides inside the FBX under its
        file name; the record names each copy with the objects it shades (an
        engine binds a renderer, not a material), the rest as it was."""
        glb, record = self._delivery()
        fbx = self._deliverable(record)
        names = MeshConvert.embed_fbx_highlight_probes(fbx, glb)
        shipped, files = self._shipped(fbx)
        probe = shipped["probe"]
        highlights = probe["highlights"]
        self.assertEqual(probe["map"], "room_Probe_Lights.hdr")
        self.assertEqual([h["map"] for h in highlights], [f"room_Probe_Highlight{i}.hdr" for i in (1, 2, 3)])
        self.assertEqual(names, [probe["map"]] + [h["map"] for h in highlights], "every file it shipped")
        shaded = {tuple(o["name"] for o in h["objects"]): h for h in highlights}
        self.assertEqual(set(shaded), {("TABLE",), ("LAMP_BODY", "LAMP_GLASS"), ("STOOL",)})
        table = shaded[("TABLE",)]
        self.assertEqual(set(table), {"map", "objects"}, "the map and its objects")
        self.assertEqual(table["objects"], [{"name": "TABLE", "hierarchy": ["TABLE"]}])
        self.assertEqual(shipped["objects"], record["objects"], "the rest as it was")
        self.assertEqual(shipped["lights"], record["lights"])
        self.assertEqual(
            {k: v for k, v in probe.items() if k not in ("map", "highlights")},
            {k: v for k, v in record["probe"].items() if k != "map"},
        )
        for name in names:
            with self.subTest(file=name):
                self.assertEqual(self._radiance(files[name]).shape[2], 3, "Radiance bytes")
        self.assertNotIn("room_Probe.exr", files, "the probe as captured no longer ships")
        self.assertEqual(SceneRecords.problems(SceneRecords.LIGHTMAPS, shipped), [])

    def test_each_copy_is_the_probe_and_its_layer_as_the_viewer_composed_them(self):
        """A copy is what the WebXR viewer composed from the probe and a
        layer before the copies were complete (``addProbe``: the probe
        box-filtered to the layer's size, plus the layer), within RGBE
        precision -- on a textured probe twice an unbaked copy's width, so
        the box filter is a real one."""
        pixels = self._textured_probe()
        glb, record = self._delivery(pixels=pixels)
        fbx = self._deliverable(record)
        MeshConvert.embed_fbx_highlight_probes(fbx, glb)
        shipped, files = self._shipped(fbx)
        layers = MeshConvert.highlight_probe_layers(
            glb,
            record["lights"],
            probe={"position": self.PROBE["position"], "box": self.BOX},
            width=pixels.shape[1],
            baked=self.BAKED,
        )
        by_receivers = {
            "TABLE": "tableMat",
            "LAMP_BODY,LAMP_GLASS": "glassMat,lampMat",
            "STOOL": "stoolMat",
        }
        layer_of = {",".join(layer["materials"]): layer["image"] for layer in layers}
        # The viewer decoded both from Radiance bytes.
        probe = self._radiance(ImgUtils.encode_hdr_radiance_pixels(pixels))
        widths = set()
        for entry in shipped["probe"]["highlights"]:
            key = ",".join(o["name"] for o in entry["objects"])
            with self.subTest(copy=entry["map"], objects=key):
                layer = self._radiance(ImgUtils.encode_hdr_radiance_pixels(layer_of[by_receivers[key]]))
                want = self._viewer_copy(probe, layer)
                copy = self._radiance(files[entry["map"]])
                self.assertEqual(copy.shape, want.shape)
                err = np.abs(copy - want) / np.maximum(want.max(-1, keepdims=True), 1e-6)
                self.assertLessEqual(float(err.max()), self.RGBE_TOLERANCE)
                self.assertGreater(float(layer.max()), 1.0, "the copy holds a light")
                widths.add(copy.shape[1])
        self.assertEqual(widths, {64, 128}, "the interior at the probe's size, an unbaked group's at half")

    def test_the_delivered_probe_holds_the_lights_as_the_capture_point_sees_them(self):
        """The probe the record names is the environment plus each light
        drawn from the capture point at its own level, onto the box as the
        probe reads it -- the light along its own direction from that point.
        Receivers in no group reflect these highlights now; nothing else of
        the environment changes."""
        pixels = self._textured_probe()
        glb, record = self._delivery(pixels=pixels)
        fbx = self._deliverable(record)
        MeshConvert.embed_fbx_highlight_probes(fbx, glb)
        shipped, files = self._shipped(fbx)
        lit = self._radiance(files[shipped["probe"]["map"]])
        lights = MeshConvert.lights_in_metres(record["lights"], record["unit_scale"])
        C = np.asarray(self.PROBE["position"], float)
        lo, hi = (np.asarray(self.BOX[k], float) for k in ("min", "max"))
        drawn = MeshConvert._highlight_layer(lights, [C] * len(lights), [1.0] * len(lights), C, lo, hi, pixels.shape[1])
        want = pixels + drawn
        self.assertEqual(lit.shape, pixels.shape, "the probe's own size")
        err = np.abs(lit - want) / np.maximum(want.max(-1, keepdims=True), 1e-6)
        self.assertLessEqual(float(err.max()), 2 * 2.0**-7)
        self.assertGreater(float(drawn.max()), float(pixels.max()), "the light outshines the window")
        self._assert_peak_toward(drawn, np.asarray(self.HANGING["position"]) - C)
        untouched = drawn.max(-1) == 0.0
        self.assertTrue(untouched.mean() > 0.9, "the environment is otherwise the probe as captured")

    def test_a_node_is_grouped_by_its_own_bake_not_by_its_material(self):
        """A GLB carries no material copy per lightmap: a material the baked
        table and the unbaked stool share is worn by both, and each node is
        under the copy its own bake calls for -- the table the interior's,
        the stool one of its own -- never both."""
        glb, record = self._delivery(
            quads=[
                ("FLOOR", "floorMat", (0.0, 0.0, 0.0), (8.0, 8.0), (0, 1, 0)),
                ("TABLE", "woodMat", (0.0, 0.9, 0.0), (1.6, 0.8), (0, 1, 0)),
                ("STOOL", "woodMat", (-3.0, 0.5, 3.0), (0.4, 0.4), (0, 1, 0)),
            ],
        )
        fbx = self._deliverable(record)
        MeshConvert.embed_fbx_highlight_probes(fbx, glb)
        shipped, _files = self._shipped(fbx)
        groups = [[o["name"] for o in h["objects"]] for h in shipped["probe"]["highlights"]]
        self.assertEqual(sorted(groups), [["STOOL"], ["TABLE"]])

    def test_drawing_again_changes_nothing(self):
        """A probe whose lights are drawn names its own file: a second pass
        over the delivered file finds nothing to draw and writes nothing."""
        glb, record = self._delivery()
        fbx = self._deliverable(record)
        self.assertTrue(MeshConvert.embed_fbx_highlight_probes(fbx, glb))
        with open(fbx, "rb") as fh:
            before = fh.read()
        with patch.object(MeshConvert, "fbx_to_glb") as convert:
            self.assertEqual(MeshConvert.embed_fbx_highlight_probes(fbx, glb), [])
        convert.assert_not_called()
        with open(fbx, "rb") as fh:
            self.assertEqual(fh.read(), before)

    def test_a_probe_not_found_leaves_the_deliverable_as_written(self):
        """Without the bake's probe there is nothing to draw the lights into:
        the record and the file stay as written, and the host's logger says so."""
        glb, record = self._delivery()
        os.remove(os.path.join(self.tmp, "room_Probe.exr"))
        fbx = self._deliverable(record)
        with open(fbx, "rb") as fh:
            before = fh.read()
        host = logging.getLogger("test_mesh_convert.highlight_probe_missing")
        with self.assertLogs(host, level="WARNING") as logs:
            self.assertEqual(MeshConvert.embed_fbx_highlight_probes(fbx, glb, logger=host), [])
        self.assertTrue(any("room_Probe.exr" in line for line in logs.output), logs.output)
        with open(fbx, "rb") as fh:
            self.assertEqual(fh.read(), before)

    def test_without_a_glb_the_copies_come_from_a_build_of_the_fbx(self):
        """An FBX-only export builds the GLB its copies read the geometry of
        the way a GLB export does -- against the host's folders -- and
        downloads no converter to do it."""
        glb, record = self._delivery()
        fbx = self._deliverable(record)
        seen = {}

        def convert(src, dst, **kwargs):
            seen.update(kwargs, src=src)
            shutil.copyfile(glb, dst)
            return dst

        with patch.object(MeshConvert, "fbx_to_glb", side_effect=convert):
            names = MeshConvert.embed_fbx_highlight_probes(fbx, search_dirs=[self.tmp])
        self.assertEqual(len(names), 4, "the lit probe and three copies")
        self.assertEqual(seen["src"], fbx, "a file without takes is read as written")
        self.assertIsNone(seen["data_export"])
        # Geometry alone: the build runs no lightmap pass, so it encodes no
        # map and needs no host codec (2026-11-07).
        self.assertIs(seen["lightmaps"], False)
        self.assertNotIn("hdr_codec", seen)
        self.assertNotIn("lightmap_dirs", seen)
        self.assertFalse(seen["auto_install"])

    def test_a_record_without_lights_ships_no_copies(self):
        glb, record = self._delivery()
        fbx = self._deliverable({k: v for k, v in record.items() if k != "lights"})
        with open(fbx, "rb") as fh:
            before = fh.read()
        self.assertEqual(MeshConvert.embed_fbx_highlight_probes(fbx, glb), [])
        with open(fbx, "rb") as fh:
            self.assertEqual(fh.read(), before)

    def _animated_deliverable(self, manifest):
        """The FBX twin as a Maya take split writes it: three takes beside *manifest*."""
        from test_fbx_media import build_animated_fbx

        return build_animated_fbx(
            os.path.join(self.tmp, "room_takes.fbx"),
            ["Take 001", "Shot_A", "Shot_B"],
            channels={"lightmaps": manifest},
        )

    def test_the_copies_build_converts_the_scene_without_its_takes(self):
        """REGRESSION (2026-10-19): an FBX-only export, and every Unity-bridge
        send, converted the whole deliverable to draw its copies -- and
        FBX2glTF bakes every node at every frame of every take (a production
        assembly's 13 takes: 1013 s of conversion, 108 s without them) for
        animation the copies never read. The build reads a copy with no take
        in a store of its own, removed after; the deliverable keeps its takes."""
        glb, record = self._delivery()
        fbx = self._animated_deliverable(record)
        seen = {}

        def convert(src, dst, **kwargs):
            read = FbxFile.load(src)
            seen.update(kwargs, src=src, takes=read.take_names(), census=read.objects_census())
            seen["record"] = FbxMedia.record(src, "lightmaps")
            shutil.copyfile(glb, dst)
            return dst

        with patch.object(MeshConvert, "fbx_to_glb", side_effect=convert):
            names = MeshConvert.embed_fbx_highlight_probes(fbx, search_dirs=[self.tmp])
        self.assertEqual(len(names), 4)
        self.assertNotEqual(os.path.abspath(seen["src"]), os.path.abspath(fbx))
        self.assertEqual(os.path.basename(seen["src"]), os.path.basename(fbx))
        self.assertEqual(seen["takes"], [])
        for kind in FbxMedia.TAKE_CHAIN:
            self.assertNotIn(kind.decode(), seen["census"])
        self.assertEqual(seen["record"], record, "the record the build reads")
        # No shot to cut in it: the clip pass is not told the scene declares some.
        self.assertEqual(seen["data_export"], {SceneRecords.SHOTS.key: None})
        self.assertFalse(os.path.exists(seen["src"]), "the copy is removed")
        self.assertEqual(FbxFile.load(fbx).take_names(), ["Take 001", "Shot_A", "Shot_B"])

    def test_a_glb_the_export_built_is_read_and_nothing_converted(self):
        """An FBX+GLB export's copies read the GLB it built: no second
        conversion, no copy of the FBX."""
        glb, record = self._delivery()
        fbx = self._animated_deliverable(record)
        with (
            patch.object(MeshConvert, "fbx_to_glb") as convert,
            patch.object(FbxMedia, "drop_takes") as drop,
        ):
            names = MeshConvert.embed_fbx_highlight_probes(fbx, glb)
        self.assertEqual(len(names), 4)
        convert.assert_not_called()
        drop.assert_not_called()

    def test_a_copy_that_cannot_be_written_leaves_the_deliverable_as_written(self):
        """The copy the build reads is written first; when it cannot be (a
        full temp disk), nothing is converted, the deliverable keeps every
        byte, and the host's logger says the copies did not ship."""
        glb, record = self._delivery()
        fbx = self._animated_deliverable(record)
        with open(fbx, "rb") as fh:
            before = fh.read()
        host = logging.getLogger("test_mesh_convert.highlight_host")
        with (
            patch.object(FbxMedia, "drop_takes", side_effect=OSError(errno.ENOSPC, "No space left on device")),
            patch.object(MeshConvert, "fbx_to_glb") as convert,
            self.assertLogs(host, level="WARNING") as logs,
        ):
            names = MeshConvert.embed_fbx_highlight_probes(fbx, search_dirs=[self.tmp], logger=host)
        self.assertEqual(names, [])
        convert.assert_not_called()
        self.assertTrue(any("highlight copies not shipped" in line for line in logs.output), logs.output)
        with open(fbx, "rb") as fh:
            self.assertEqual(fh.read(), before)

    #: Two lights: one over the table, one by the stool across the room.
    _NEAR_STOOL = {"name": "STOOL_KEY", "type": "point", "position": [-2.5, 1.5, 2.5],
                   "color": [1.0, 0.9, 0.8], "intensity": 1.5}

    def _room_fbx(self, name, moves):
        """The room as an FBX deliverable: ``gltf_to_fbx``'s build of it -- its
        bake's UV set, and a take keying *moves* -- with the record a Scene
        Exporter carries on its ``data_export`` node: the floor and table
        baked into a lightmap, the probe and two lights."""
        from pythontk.file_utils.mesh_convert._fbx_writer import FbxNode, FbxWriter
        from pythontk.file_utils.mesh_convert._gltf2fbx import _GltfDocument, _GltfToFbx

        _glb, manifest = self._delivery(lights=[self.HANGING, self._NEAR_STOOL])
        build = _GltfToFbx(
            _GltfDocument(self._scene(uvs=True, moves=moves)),
            texture_dir=os.path.join(self.tmp, "textures"),
            include_animation=True,
            fps=24.0,
        )
        build.build()
        uid = build._id()
        carrier = build._object(FbxNode("Model", FbxWriter.i64(uid), "data_export\x00\x01Model", "Null"), "Model")
        carrier.add("Properties70").p("lightmaps", "KString", "", "U", json.dumps(manifest))
        build._link(uid, 0)
        return FbxWriter.write(os.path.join(self.tmp, name), build.records("test_mesh_convert"))

    def test_the_copies_without_the_takes_are_the_copies_of_the_file_as_written(self):
        """The copies read the scene at rest: FBX2glTF writes each node's
        transform as its properties hold it, which no take moves -- so a build
        of the deliverable without its takes draws the very copies a build of
        it as written does, to the byte, on a still scene (a take holding the
        rest pose, as a still export writes one) and on one whose take carries
        the stool and the lamp a metre and more from where they rest (placed
        there, they draw other copies: measured 2026-10-19)."""
        if not MeshConvert.resolve_binary(required=False, auto_install=False):
            self.skipTest("FBX2glTF absent: the identity needs the real converter")
        convert = MeshConvert.fbx_to_glb
        cases = {
            "still": {"STOOL": [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.0, 0.0, 0.0))]},
            "animated": {
                "STOOL": [(0.0, (1.0, 0.0, -1.0)), (1.0, (2.0, 0.5, -2.0))],
                "LAMP_BODY": [(0.0, (0.0, 0.6, 0.0)), (2.0, (-1.0, 1.2, 0.0))],
            },
        }
        for label, moves in cases.items():
            with self.subTest(label):
                source = self._room_fbx(f"{label}.fbx", moves)
                shipped = {}
                for build in ("as_written", "without_takes"):
                    fbx = os.path.join(self.tmp, build, f"{label}.fbx")
                    os.makedirs(os.path.dirname(fbx), exist_ok=True)
                    shutil.copyfile(source, fbx)
                    takes = []

                    def spy(src, dst, **kwargs):
                        takes.append(FbxFile.load(src).take_names())
                        return convert(src, dst, **kwargs)

                    with patch.object(MeshConvert, "fbx_to_glb", side_effect=spy):
                        if build == "as_written":
                            with patch.object(MeshConvert, "_fbx_static_scene", side_effect=lambda f, _store: f):
                                names = MeshConvert.embed_fbx_highlight_probes(fbx, search_dirs=[self.tmp])
                        else:
                            names = MeshConvert.embed_fbx_highlight_probes(fbx, search_dirs=[self.tmp])
                    self.assertEqual([len(t) for t in takes], [1] if build == "as_written" else [0])
                    record = FbxMedia.record(fbx, "lightmaps")["probe"]
                    out = os.path.join(self.tmp, build, f"{label}_maps")
                    os.makedirs(out)
                    extracted = FbxMedia.extract_dependencies(fbx, out)
                    maps = {}
                    for name in names:
                        with open(extracted[name], "rb") as fh:
                            maps[name] = fh.read()
                    shipped[build] = (names, record, maps)
                self.assertEqual(len(shipped["as_written"][0]), 4)
                self.assertEqual(shipped["without_takes"], shipped["as_written"])

    def test_a_node_goes_under_the_copy_holding_most_of_its_triangles(self):
        """Named as the lightmap record names an object: its hierarchy root
        first, without the root FBX2glTF puts every scene under."""
        gltf = {
            "nodes": [
                {"name": "RootNode", "children": [1]},
                {"name": "GRP", "children": [2]},
                {"name": "MIXED", "mesh": 0},
            ],
            "meshes": [{"primitives": [{"material": 0, "indices": 0}, {"material": 1, "indices": 1}]}],
            "materials": [{"name": "a"}, {"name": "b"}],
            "accessors": [{"count": 6}, {"count": 30}],
        }
        self.assertEqual(
            MeshConvert._highlight_receivers(gltf, [{"a"}, {"b"}]),
            [[], [{"name": "MIXED", "hierarchy": ["GRP", "MIXED"]}]],
        )

    def test_the_pass_writes_only_what_its_block_declares(self):
        """The record the lightmaps pass carries -- its drawn probe and each
        complete copy included -- is one its shape declares, so nothing is
        cut, nor warned, on the way into the file."""
        _require_cv2()

        with self.assertNoLogs("pythontk.core_utils.engines.scene_export", "WARNING"):
            glb, _manifest = self._lit_scene()
        with MeshConvert.open_glb(glb) as edit:
            record = edit.gltf["extras"][SceneRecords.LIGHTMAPS.key]
        self.assertTrue(record["probe"]["highlights"], "the copies are written")
        self.assertEqual(SceneRecords.problems(SceneRecords.LIGHTMAPS, record), [])

    def test_the_lightmap_pass_rides_the_drawn_probe_and_its_copies(self):
        """Through ``apply_glb_lightmaps`` -- the GLB's half of the
        deliverable-time pass: the record it carries names the probe with the
        bake's lights drawn in and the complete copies, each riding the file
        as Radiance bytes under its name (``SceneRecords.GLB_MEDIA``) --
        exactly the files and the record the FBX deliverable's pass draws
        from the same geometry -- and no light rides the GLB as a light.
        Each copy names the nodes that read it, the table's and not the
        floor's (the shell reads the probe). Added: 2026-11-07"""
        cv2 = _require_cv2()

        glb, manifest = self._lit_scene()
        media = SceneRecords.GLB_MEDIA
        drawn = MeshConvert._highlights._deliverable_probes(glb, manifest, [self.tmp])
        with MeshConvert.open_glb(glb) as edit:
            record = edit.gltf["extras"][SceneRecords.LIGHTMAPS.key]
            self.assertEqual(record, drawn["record"])
            probe = record["probe"]
            self.assertEqual(
                probe["map"], MeshConvert.HIGHLIGHT_PROBE_NAME.format(stem="room_Probe")
            )
            self.assertEqual(record["lights"], manifest["lights"])
            readers = {o["name"]: c["map"] for c in probe["highlights"] for o in c["objects"]}
            self.assertIn("TABLE", readers)
            self.assertNotIn("FLOOR", readers)
            self.assertNotIn("KHR_lights_punctual", edit.gltf.get("extensions") or {})
            for name, data in drawn["files"]:
                index = media.view(edit.gltf, name)
                self.assertIsNotNone(index, name)
                view = edit.gltf["bufferViews"][index]
                self.assertEqual(view["extras"], media.extras(media.radiance))
                start = int(view.get("byteOffset", 0))
                self.assertEqual(
                    bytes(edit.bin_data[start : start + int(view["byteLength"])]), data
                )
            # The bake's own probe went with the drawing: the record names it no more.
            self.assertIsNone(media.view(edit.gltf, "room_Probe.exr"))
        lit = cv2.imdecode(np.frombuffer(drawn["files"][0][1], np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(lit.shape, (32, 64, 3), "the probe's own size")
        self.assertGreater(float(lit.max()), 0.2, "the light drawn in")

    def test_the_glb_of_a_finished_fbx_rides_the_files_its_record_names(self):
        """A GLB built from a delivered FBX -- whose record names the drawn
        probe and copies already -- draws nothing again: each file rides as
        the FBX embedded it, read out of the FBX itself (the drawn files are
        nowhere on disk: they exist only inside the delivery).
        Added: 2026-11-07"""
        from test_fbx_media import build_carrier_fbx

        cv2 = _require_cv2()
        glb, manifest = self._lit_scene()
        drawn = MeshConvert._highlights._deliverable_probes(glb, manifest, [self.tmp])
        self.assertGreaterEqual(len(drawn["files"]), 2, "the fixture draws a probe and a copy")
        for name, _data in drawn["files"]:
            self.assertFalse(os.path.exists(os.path.join(self.tmp, name)), name)
        fbx = build_carrier_fbx(
            os.path.join(self.tmp, "delivered.fbx"),
            {"lightmaps": drawn["record"]},
            media=dict(drawn["files"]),
        )
        carrier = {"name": "data_export", "extras": {"fromFBX": {"userProperties": {
            "lightmaps": {"type": "eFbxString", "value": json.dumps(drawn["record"])}}}}}
        again = self._scene(extra_nodes=[carrier])
        with MeshConvert.open_glb(again) as edit:
            for mesh in edit.gltf["meshes"]:
                for prim in mesh["primitives"]:
                    prim["attributes"]["TEXCOORD_1"] = prim["attributes"]["POSITION"]
            edit.dirty = True
        converted = os.path.join(self.tmp, "converted.glb")
        shutil.copyfile(again, converted)
        with patch.object(
            MeshConvert._highlights, "_deliverable_probes", side_effect=AssertionError("drawn")
        ):
            out = _convert_prebuilt(self.tmp, converted, src=fbx, lightmap_dirs=[self.tmp])
        with MeshConvert.open_glb(out) as edit:
            self.assertEqual(edit.gltf["extras"][SceneRecords.LIGHTMAPS.key], drawn["record"])
            for name, data in drawn["files"]:
                index = SceneRecords.GLB_MEDIA.view(edit.gltf, name)
                self.assertIsNotNone(index, name)
                view = edit.gltf["bufferViews"][index]
                start = int(view.get("byteOffset", 0))
                held = bytes(edit.bin_data[start : start + view["byteLength"]])
                decode = lambda raw: cv2.imdecode(  # noqa: E731
                    np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH
                )
                np.testing.assert_allclose(decode(held), decode(data), rtol=2**-7, err_msg=name)
            self.assertEqual(MeshConvert._lightmaps.media_findings(edit.gltf), ([], []))
        # The store the FBX's files went to is scoped: nothing is left beside it.
        for name, _data in drawn["files"]:
            self.assertFalse(os.path.exists(os.path.join(self.tmp, name)), name)


class TestFinishFbxDeliverable(unittest.TestCase):
    """``finish_fbx_deliverable``: the one sequence every FBX hand-off runs."""

    def setUp(self):
        from pythontk import TempArtifacts
        from test_fbx_media import build_carrier_fbx

        artifacts = TempArtifacts("mesh_convert_finish_fbx", policy="scoped")
        self.addCleanup(artifacts.cleanup)
        self.fbx = build_carrier_fbx(os.path.join(artifacts.dir_path(), "room.fbx"), {})
        self.text = os.path.join(artifacts.dir_path(), "room.txt")
        with open(self.text, "w", encoding="utf-8") as fh:
            fh.write("not an fbx")

    def test_the_passes_run_in_order_and_a_failure_costs_only_its_own(self):
        ran = []

        def step(name, result=None, error=None):
            def run(*_args, **_kwargs):
                ran.append(name)
                if error:
                    raise error
                return result
            return run

        with (
            patch.object(FbxMedia, "split_pivot_instances", side_effect=step("instances", {"models": []})),
            patch.object(FbxMedia, "drop_host_lighting_media", side_effect=step("host_lighting", error=ValueError("boom"))),
            patch.object(MeshConvert, "embed_fbx_highlight_probes", side_effect=step("highlights", ["a.hdr"])),
            patch.object(FbxMedia, "embed_dependencies", side_effect=step("dependencies", {"embedded": []})),
            self.assertLogs("pythontk", level="WARNING") as logs,
        ):
            report = MeshConvert.finish_fbx_deliverable(self.fbx, search_dirs=["D:/maps"])
        # The copies first: a build then reads the file a GLB export reads.
        self.assertEqual(ran, ["highlights", "instances", "host_lighting", "dependencies"])
        self.assertIsNone(report["host_lighting"])
        self.assertEqual(report["highlights"], ["a.hdr"])
        self.assertTrue(any("host_lighting pass failed" in line for line in logs.output))

    def test_a_hosts_inputs_answer_where_the_scene_keeps_its_files(self):
        """``inputs`` is the host's answer to where the scene keeps the files
        its records name -- each ``None`` asked of the scene -- and what it
        answers reaches the passes; one that fails is warned and the passes
        run on what was given. The guarded body every host's
        ``FbxUtils.finish_deliverable`` once wrote, written once.
        Added: 2026-11-07"""
        seen = {}

        def embed(fbx, search_dirs=(), walk_root="", files=(), logger=None):
            seen.update(search_dirs=search_dirs, walk_root=walk_root, files=list(files))
            return {"embedded": []}

        def asked(fbx, search_dirs, walk_root, files, log):
            seen["asked"] = (search_dirs, walk_root, files)
            return ["D:/scene"], "D:/root", ["a.wav"]

        def broken(*_args):
            raise RuntimeError("no scene")

        quiet = (
            patch.object(MeshConvert, "embed_fbx_highlight_probes", return_value=[]),
            patch.object(FbxMedia, "split_pivot_instances", return_value={}),
            patch.object(FbxMedia, "drop_host_lighting_media", return_value={}),
            patch.object(FbxMedia, "embed_dependencies", side_effect=embed),
        )
        with quiet[0], quiet[1], quiet[2], quiet[3]:
            MeshConvert.finish_fbx_deliverable(
                self.fbx, search_dirs=None, walk_root=None, files=None, inputs=asked
            )
        self.assertEqual(seen["asked"], (None, None, None))
        self.assertEqual(
            (seen["search_dirs"], seen["walk_root"], seen["files"]),
            (["D:/scene"], "D:/root", ["a.wav"]),
        )
        with quiet[0], quiet[1], quiet[2], quiet[3], self.assertLogs(
            "pythontk", level="WARNING"
        ) as logs:
            MeshConvert.finish_fbx_deliverable(
                self.fbx, search_dirs=["D:/given"], inputs=broken
            )
        self.assertEqual((seen["search_dirs"], seen["walk_root"]), (["D:/given"], ""))
        self.assertTrue(any("files not resolved (no scene)" in line for line in logs.output))

    def test_the_handoff_states_what_the_records_hold_after_the_passes(self):
        """A pass after the write can change a record -- the highlight copies
        add the probe's ``highlights`` -- so the handoff written at the commit
        no longer says what the record holds; the last pass states it again
        from the file's records, its source and the export's lighting choices
        kept. Measured on the soldering export: the FBX's lightmaps row lacked
        the highlight copies its record listed. Added: 2026-10-20"""
        from test_fbx_media import build_carrier_fbx

        lightmaps = {
            "version": SceneRecords.LIGHTMAPS.version,
            "unit_scale": 0.01,
            "objects": [
                {
                    "name": "Crate",
                    "map": "Room_Lightmap.exr",
                    "scaleOffset": [1, 1, 0, 0],
                    "hierarchy": ["Crate"],
                }
            ],
            "probe": {"map": "Room_Probe.exr", "position": [0, 0, 0], "box": None},
        }
        source = {"application": "maya", "version": "2025"}
        choices = {"lightmappedMaterials": {"envMapIntensity": 0.5}}
        handoff = HandoffRecord.block("fbx", {"lightmaps": lightmaps}, source, choices)
        fbx = build_carrier_fbx(
            os.path.join(os.path.dirname(self.fbx), "lit.fbx"),
            {"lightmaps": lightmaps, HandoffRecord.KEY: handoff},
        )
        copies = [{"map": "Room_Probe_Highlight1.hdr", "objects": [{"name": "Crate", "hierarchy": ["Crate"]}]}]
        lit = dict(lightmaps, probe=dict(lightmaps["probe"], highlights=copies))
        FbxMedia.set_record(fbx, "lightmaps", lit)

        def finish():
            with (
                patch.object(MeshConvert, "embed_fbx_highlight_probes", return_value=[]),
                patch.object(FbxMedia, "embed_dependencies", return_value={}),
            ):
                return MeshConvert.finish_fbx_deliverable(fbx)

        self.assertTrue(finish()["handoff"])
        stated = FbxMedia.record(fbx, HandoffRecord.KEY)
        rows = {r["part"]: r["spec"] for r in stated["assembly"]}
        self.assertEqual(rows["lightmaps"], SceneRecords.LIGHTMAPS.assembly_for(lit))
        self.assertEqual(stated["extras"], {"source": source})
        self.assertEqual(
            stated["rendering"]["lightmappedMaterials"]["envMapIntensity"], 0.5
        )
        self.assertFalse(finish()["handoff"], "stated once: nothing left to change")

    def test_a_file_that_is_not_a_binary_fbx_is_left_alone(self):
        with patch.object(FbxMedia, "split_pivot_instances") as split:
            self.assertEqual(MeshConvert.finish_fbx_deliverable(self.text), {})
        split.assert_not_called()

    def test_the_highlight_pass_says_its_outcome_where_the_host_logs(self):
        """Every pass but the copies' said its outcome on the host's logger:
        a copy build that could not run (an FBX-only export with no FBX2glTF
        installed, which this pass never downloads) warned on pythontk's
        module logger, which no host's run log shows, and the deliverable
        shipped without its highlights in silence. Added: 2026-10-19"""
        host = logging.getLogger("test_mesh_convert.finish_host")
        with (
            patch.object(FbxMedia, "record", side_effect=OSError("locked")),
            patch.object(FbxMedia, "split_pivot_instances", return_value={}),
            patch.object(FbxMedia, "drop_host_lighting_media", return_value={}),
            patch.object(FbxMedia, "embed_dependencies", return_value={}),
            self.assertLogs(host, level="WARNING") as logs,
        ):
            MeshConvert.finish_fbx_deliverable(self.fbx, logger=host)
        self.assertTrue(
            any("highlight copies" in line for line in logs.output), logs.output
        )

    def test_the_deliverable_names_no_folder_of_the_machine_that_wrote_it(self):
        """The last pass strips the producer's paths (``strip_machine_paths``),
        after every pass that adds media, so what they embed is covered too;
        an ASCII write -- which no other pass reads -- takes it alone.
        Added: 2026-11-08"""
        from test_fbx_media import ASCII_FBX, build_paths_fbx

        order = []
        real = FbxMedia.strip_machine_paths

        def strip(path, *args, **kwargs):
            order.append("paths")
            return real(path, *args, **kwargs)

        fbx = build_paths_fbx(os.path.join(os.path.dirname(self.fbx), "paths.fbx"))
        with (
            patch.object(MeshConvert, "embed_fbx_highlight_probes", return_value=[]),
            patch.object(FbxMedia, "embed_dependencies", side_effect=lambda *a, **k: order.append("dependencies") or {}),
            patch.object(FbxMedia, "strip_machine_paths", side_effect=strip),
        ):
            report = MeshConvert.finish_fbx_deliverable(fbx)
        self.assertEqual(order, ["dependencies", "paths"])
        self.assertEqual(report["paths"]["media"], ["Wood_BaseColor.png", "Wood_Normal.png"])
        with open(fbx, "rb") as fh:
            self.assertNotIn(b"producer\\AppData", fh.read())

        ascii_fbx = os.path.join(os.path.dirname(self.fbx), "ascii.fbx")
        with open(ascii_fbx, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(ASCII_FBX)
        with patch.object(FbxMedia, "split_pivot_instances") as split:
            report = MeshConvert.finish_fbx_deliverable(ascii_fbx)
        split.assert_not_called()
        self.assertEqual(list(report), ["paths"])
        self.assertEqual(report["paths"]["media"], ["Wood_BaseColor.png"])
        with open(ascii_fbx, encoding="utf-8") as fh:
            self.assertNotIn("producer", fh.read())


def _png(width: int = 4, height: int = 4, rgb=(200, 120, 40)) -> bytes:
    """A small RGB PNG, written without an imaging library."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    )


@unittest.skipUnless(_FBX2GLTF, "FBX2glTF is not installed; this test downloads nothing")
class TestStrippedPathsReachTheGlb(unittest.TestCase):
    """An FBX whose embedded maps name no folder still converts with every
    map bound: FBX2glTF reads the embedded ``Content``, never the source
    path (``FbxMedia.strip_machine_paths``). Added: 2026-11-08"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_paths_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_the_images_survive_the_strip(self):
        fbx = build_stingray_fbx(
            os.path.join(self.tmp, "orm.fbx"),
            {
                "color": ("t_Base_Color.png", _png(rgb=(200, 120, 40))),
                "roughness": ("t_Roughness.png", _png(rgb=(90, 90, 90))),
            },
            scene_info={"DocumentUrl": "C:\\Users\\producer\\AppData\\Local\\Temp\\orm.fbx"},
        )
        report = FbxMedia.strip_machine_paths(fbx)
        self.assertEqual(report["media"], ["t_Base_Color.png", "t_Roughness.png"])
        self.assertEqual(report["document"], ["DocumentUrl"])
        with open(fbx, "rb") as fh:
            data = fh.read()
        self.assertNotIn(b"C:/tex/", data)
        self.assertNotIn(b"producer", data)
        glb = MeshConvert.fbx_to_glb(fbx, auto_install=False, prompt=False, lightmaps=False)
        with MeshConvert.open_glb(glb) as edit:
            (material,) = edit.materials
            pbr = material["pbrMetallicRoughness"]
            self.assertIn("baseColorTexture", pbr)
            self.assertIn("metallicRoughnessTexture", pbr)
            images = edit.gltf.get("images") or []
            self.assertGreaterEqual(len(images), 2, images)
            self.assertTrue(all("bufferView" in image for image in images), images)


class TestBakedLightsLeaveTheGlb(unittest.TestCase):
    """A lightmapped GLB ships no ``KHR_lights_punctual`` lights.

    Every scene light is already in the lightmaps (the Arnold bake renders all
    of them), so a light that also ships lands on top of its own bake, with no
    shadow: an export run under a saved FBX preset (lights on) brought twelve
    bake spots into a moody room's GLB and the floor under the table read 3x
    its baked value in WebXR (2026-10-16).
    """

    def setUp(self):
        import pythontk as ptk

        artifacts = ptk.TempArtifacts("mesh_convert_baked_lights", policy="scoped")
        self.tmp = artifacts.dir_path()
        self.addCleanup(artifacts.cleanup)

    def _prebuilt(self, lightmapped, version=None):
        gltf = {
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [0, 1]}],
            "scene": 0,
            "nodes": [
                {"name": "KEY", "extensions": {"KHR_lights_punctual": {"light": 0}}},
                {"name": "ROOM", "mesh": 0},
            ],
            "meshes": [{"primitives": [{"attributes": {"TEXCOORD_1": 0}}]}],
            "extensionsUsed": ["KHR_lights_punctual"],
            "extensions": {
                "KHR_lights_punctual": {
                    "lights": [{"type": "spot", "intensity": 2.0, "spot": {}}]
                }
            },
        }
        if lightmapped:
            record = _lightmaps_record("ROOM", version=version)
            gltf["extras"] = {SceneRecords.LIGHTMAPS.key: record}
        path = _write_glb_file(os.path.join(self.tmp, "prebuilt.glb"), gltf)
        return _ride_media(path, ["ROOM_LM.exr"]) if lightmapped else path

    @staticmethod
    def _gltf(path):
        return MeshConvert._read_glb(path).gltf

    def _converted(self, lightmapped):
        return self._gltf(_convert_prebuilt(self.tmp, self._prebuilt(lightmapped)))

    def test_a_lightmapped_glb_ships_no_punctual_light(self):
        gltf = self._converted(lightmapped=True)
        self.assertNotIn("KHR_lights_punctual", gltf.get("extensions") or {})
        self.assertNotIn("KHR_lights_punctual", gltf.get("extensionsUsed") or [])
        for node in gltf["nodes"]:
            self.assertNotIn("KHR_lights_punctual", node.get("extensions") or {})
        self.assertEqual([n["name"] for n in gltf["nodes"]], ["KEY", "ROOM"])

    def test_an_unbaked_glb_keeps_its_lights(self):
        gltf = self._converted(lightmapped=False)
        self.assertEqual(len(gltf["extensions"]["KHR_lights_punctual"]["lights"]), 1)
        self.assertIn("KHR_lights_punctual", gltf["nodes"][0]["extensions"])

    def test_a_record_lighting_nothing_here_strips_no_light(self):
        """Keyed on what the record lights IN THIS FILE: a record whose map
        rides nothing here (a node of the file it names, but its bake not
        shipped) leaves the file its lights -- then its only lighting.
        Added: 2026-11-07"""
        path = self._prebuilt(lightmapped=True)
        with MeshConvert.open_glb(path) as edit:
            for view in edit.gltf["bufferViews"]:
                view.pop(SceneRecords.GLB_MEDIA.name, None)
            edit.dirty = True
        self.assertEqual(MeshConvert.strip_glb_baked_lights(path), [])

    def test_a_block_of_another_version_strips_no_light(self):
        """The lightmaps are read off the record in its current schema alone
        (``SceneRecords.glb_block``): a file whose record is another version --
        an older one too -- keeps its lights until it is re-exported, rather
        than losing them to a guess. Added: 2026-10-20"""
        current = SceneRecords.LIGHTMAPS.version
        for version in (current - 1, current + 1):
            with self.subTest(version=version):
                path = self._prebuilt(lightmapped=True, version=version)
                self.assertEqual(MeshConvert.strip_glb_baked_lights(path), [])
                self.assertIn(
                    "KHR_lights_punctual", self._gltf(path).get("extensions") or {}
                )


class TestMaterialsRecord(unittest.TestCase):
    """The ``materials`` record an FBX carries, read and applied to its GLB.

    Measured on a production assembly (2026-10-20): a GLB built from its FBX
    alone -- the WebXR preview of a delivered file -- halved every StingrayPBS
    albedo, dropped the emission and left a Phong lens opaque, because only a
    live scene's sidecar repaired what FBX2glTF misread; Unity read the same
    lens with other metal and roughness again. The export now states every
    material in the file (the hosts' producers), a file without that record
    is read into the same families, and every reader applies the one
    statement. Added: 2026-10-20
    """

    def setUp(self):
        from pythontk import TempArtifacts

        artifacts = TempArtifacts("mesh_convert_materials", policy="scoped")
        self.tmp = artifacts.dir_path()
        self.addCleanup(artifacts.cleanup)

    @staticmethod
    def _png(color, mode="RGB", size=8):
        from PIL import Image

        return encoded(Image.new(mode, (size, size), color))

    def _stingray(self, name="stingray.fbx", maps=None, props=None, records=None):
        """A StingrayPBS quad as Maya writes a lamp: colour, emission, metal and
        roughness maps in use, its inert ``base_color`` 0.5, intensity 0.75."""
        if maps is None:
            maps = {
                "color": ("Lamp_BaseColor.png", self._png((180, 120, 60))),
                "emissive": ("Lamp_Emissive.png", self._png((255, 255, 255))),
                "metallic": ("Lamp_Metallic.png", self._png(30, "L")),
                "roughness": ("Lamp_Roughness.png", self._png(200, "L")),
            }
        stated = {
            "base_color": (0.5, 0.5, 0.5),
            "emissive": (0.0, 0.0, 0.0),
            "emissive_intensity": 0.75,
            **(props or {}),
        }
        return build_stingray_fbx(
            os.path.join(self.tmp, name), maps, records=records, props=stated
        )

    def _legacy(self, name, model, rows, maps=None):
        return build_stingray_fbx(
            os.path.join(self.tmp, f"{name}.fbx"), maps or {}, shading=(model, rows), name=name
        )

    @staticmethod
    def _entry(fbx):
        """The file's one material, flat (:func:`flat_material`)."""
        record = MeshConvert.fbx_materials(fbx)
        (material,) = record["materials"]
        return flat_material(record, material)

    @staticmethod
    def _rounded(values):
        return [round(v, 6) for v in values]

    # ------------------------------------------- reading a file's own materials
    def test_a_stingray_material_is_stated_by_its_switches(self):
        """A map in use stands in for its value: the colour map is the albedo
        (not times the inert 0.5 FBX2glTF multiplied it by), the emission the
        map at ``emissive_intensity`` (where FBX2glTF read 0); its metal and
        roughness maps are named as the one image they pack into."""
        entry = self._entry(self._stingray())
        self.assertEqual(entry["baseColorFactor"], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(entry["baseColorTexture"], "Lamp_BaseColor.png")
        self.assertEqual(
            (entry["metallicFactor"], entry["roughnessFactor"]), (1.0, 1.0)
        )
        self.assertEqual(entry["metallicRoughnessTexture"], "Lamp_ORM.png")
        self.assertEqual(self._rounded(entry["emissiveFactor"]), [0.75] * 3)
        self.assertEqual(entry["emissiveTexture"], "Lamp_Emissive.png")
        self.assertEqual(entry["alphaMode"], "OPAQUE")

    def test_a_map_switched_off_states_the_value(self):
        fbx = self._stingray(
            props={
                "use_color_map": 0.0,
                "use_emissive_map": 0.0,
                "base_color": (0.2, 0.3, 0.4),
                "emissive": (1.0, 0.5, 0.0),
            }
        )
        entry = self._entry(fbx)
        self.assertEqual(self._rounded(entry["baseColorFactor"]), [0.2, 0.3, 0.4, 1.0])
        self.assertNotIn("baseColorTexture", entry)
        self.assertEqual(self._rounded(entry["emissiveFactor"]), [0.75, 0.375, 0.0])
        self.assertNotIn("emissiveTexture", entry)

    def test_the_opacity_graph_states_its_alpha_mode(self):
        """The transparent graph carries ``opacity``, the masked one
        ``mask_threshold``: the file says which, so a reader need not guess
        from a colour map's alpha."""
        clear = self._entry(
            self._stingray("t.fbx", props={"use_opacity_map": 0.0, "opacity": 0.4})
        )
        self.assertEqual((clear["alphaMode"], round(clear["baseColorFactor"][3], 6)), ("BLEND", 0.4))
        cut = self._entry(self._stingray("m.fbx", props={"mask_threshold": 0.3}))
        self.assertEqual((cut["alphaMode"], round(cut["alphaCutoff"], 6)), ("MASK", 0.3))

    def test_a_phong_lens_is_stated_as_the_converter_converts_it_and_blends(self):
        """The legacy conversion every approved deliverable was seen through
        -- colour times its factor, opacity one minus the transparent colour's
        mean, metallic 0.4, roughness sqrt(2 / (2 + shininess)) -- except that
        the lens blends: FBX2glTF judged it by a colour map it has none of,
        and shipped it opaque."""
        entry = self._entry(
            self._legacy(
                "GLASS",
                "phong",
                [
                    ("DiffuseColor", "Color", 0.5, 0.5, 0.5),
                    ("DiffuseFactor", "Number", 0.8),
                    ("TransparentColor", "Color", 0.9, 0.9, 0.9),
                    ("TransparencyFactor", "Number", 1.0),
                    ("Shininess", "double", 30.0),
                ],
            )
        )
        self.assertEqual(entry["name"], "GLASS")
        self.assertEqual(self._rounded(entry["baseColorFactor"]), [0.4, 0.4, 0.4, 0.1])
        self.assertAlmostEqual(entry["metallicFactor"], 0.4)
        self.assertAlmostEqual(entry["roughnessFactor"], math.sqrt(2.0 / 32.0))
        self.assertEqual(entry["alphaMode"], "BLEND")

    def test_a_decals_transparency_map_is_its_colour_maps_alpha(self):
        decal = self._png((10, 10, 10, 128), "RGBA")
        entry = self._entry(
            self._legacy(
                "DECAL",
                "phong",
                [("DiffuseColor", "Color", 1, 1, 1), ("DiffuseFactor", "Number", 0.8)],
                maps={
                    "DiffuseColor": ("Stain.png", decal),
                    "TransparentColor": ("Stain.png", decal),
                },
            )
        )
        self.assertEqual(entry["baseColorTexture"], "Stain.png")
        self.assertEqual(self._rounded(entry["baseColorFactor"]), [0.8, 0.8, 0.8, 1.0])
        self.assertEqual(entry["alphaMode"], "BLEND")

    def test_a_lambert_takes_the_converters_other_constants(self):
        entry = self._entry(
            self._legacy(
                "RED", "lambert", [("DiffuseColor", "Color", 1, 0, 0), ("DiffuseFactor", "Number", 1.0)]
            )
        )
        self.assertEqual(
            (entry["metallicFactor"], entry["roughnessFactor"], entry["alphaMode"]),
            (0.2, 0.8, "OPAQUE"),
        )

    def test_a_standard_surface_states_its_weights(self):
        """FBX2glTF reads nothing of one: it arrived flat white. Its emission
        above 1 is the strength extension's (glTF caps the factor at 1)."""
        entry = self._entry(
            self._legacy(
                "SS",
                "unknown",
                [
                    ("Maya", "Compound"),
                    ("Maya|TypeId", "int", 1398031443),
                    ("Maya|base", "Number", 0.5),
                    ("Maya|baseColor", "Color", 0.8, 0.4, 0.2),
                    ("Maya|metalness", "Number", 1.0),
                    ("Maya|specularRoughness", "Number", 0.25),
                    ("Maya|emission", "Number", 2.0),
                    ("Maya|emissionColor", "Color", 1.0, 0.5, 0.0),
                    ("Maya|opacity", "Color", 1, 1, 1),
                ],
            )
        )
        self.assertEqual(self._rounded(entry["baseColorFactor"]), [0.4, 0.2, 0.1, 1.0])
        self.assertEqual((entry["metallicFactor"], entry["roughnessFactor"]), (1.0, 0.25))
        self.assertEqual(self._rounded(entry["emissiveFactor"]), [2.0, 1.0, 0.0])
        self.assertEqual(entry["emissiveStrength"], 2.0)
        self.assertEqual(entry["alphaMode"], "OPAQUE")

    def test_the_record_validates_against_its_declared_shape(self):
        from pythontk import SceneRecords

        record = MeshConvert.fbx_materials(self._stingray())
        self.assertEqual(record["version"], SceneRecords.MATERIALS.version)
        result = SceneRecords.shape(SceneRecords.MATERIALS).validate(record)
        self.assertTrue(result.ok, result.errors)

    # ------------------------------------------------ the record a file carries
    @staticmethod
    def _unlit_record(*names):
        """A record stating an unlit grey for each of *names*."""
        return MaterialFamilies.payload(
            MaterialFamilies.entry(
                "unlit", name, {"color": (0.2, 0.2, 0.2) if i == 0 else (1, 0, 0)}
            )
            for i, name in enumerate(names)
        )

    def test_the_files_record_states_each_material_it_names(self):
        """Whole, over the file's own reading -- an unlit its material cannot
        say -- and an entry for a material the file does not carry is out of
        scope."""
        carried = self._unlit_record("orm", "GONE")
        record = MeshConvert.fbx_materials(self._stingray(records={"materials": carried}))
        self.assertEqual([e["name"] for e in record["materials"]], ["orm"])
        entry = flat_material(record, record["materials"][0])
        self.assertTrue(entry["unlit"])
        self.assertEqual(entry["baseColorFactor"], [0.2, 0.2, 0.2, 1.0])

    def test_a_material_the_record_does_not_name_is_read_from_the_file(self):
        carried = self._unlit_record("GONE")
        entry = self._entry(self._stingray(records={"materials": carried}))
        self.assertNotIn("unlit", entry)
        self.assertEqual(entry["baseColorTexture"], "Lamp_BaseColor.png")

    def test_a_newer_schema_is_never_guessed_at(self):
        """A record of a schema this reader does not know may mean other
        things by its fields: the file's own reading stands, said."""
        carried = dict(self._unlit_record("orm"), version=99)
        fbx = self._stingray(records={"materials": carried})
        with self.assertLogs("pythontk", level="WARNING"):
            entry = self._entry(fbx)
        self.assertNotIn("unlit", entry)

    def test_a_record_its_declaration_refuses_is_never_guessed_at(self):
        """REGRESSION (2026-10-24): the reader took any record whose version
        was not above its own -- one stating none read as 0 -- and never
        checked the shape, where every other record is read through
        ``SceneRecords.problems``. A record stating no version, breaking its
        shape, or of version 1 (the flat shape a re-export replaces) is
        refused whole, said, as a newer one is."""
        no_version = dict(self._unlit_record("orm"))
        no_version.pop("version")
        broken = dict(self._unlit_record("orm"), materials=[{"name": "orm"}])
        flat = {"version": 1, "materials": [{"name": "orm", "unlit": True}]}
        for carried in (no_version, broken, flat):
            with self.subTest(carried=carried):
                fbx = self._stingray(records={"materials": carried})
                with self.assertLogs("pythontk", level="WARNING") as logs:
                    entry = self._entry(fbx)
                self.assertNotIn("unlit", entry)
                self.assertTrue(
                    any("materials record" in line for line in logs.output), logs.output
                )
        self.assertTrue(any("re-export" in line for line in logs.output), logs.output)

    def test_the_record_is_read_off_a_carrier_alone(self):
        """The scene's own carrier's record wins, else a module's carrier's
        (``NS:data_export``) -- the precedence every carrier channel is read
        by. A property of the name on any other node is an artist's, never
        the record (2026-10-24: it was read as one)."""
        from types import SimpleNamespace

        def parsed(*held):
            return SimpleNamespace(
                user_properties=lambda key, named=True, paths=False: [
                    (owner, json.dumps({"version": 2, "materials": [], "by": owner}).encode())
                    for owner in held
                ]
            )

        read = MeshConvert._carried_record
        self.assertIsNone(read(parsed("Crate")))
        self.assertEqual(read(parsed("Crate", "MOD:data_export"))["by"], "MOD:data_export")
        self.assertEqual(
            read(parsed("MOD:data_export", "data_export"))["by"], "data_export"
        )
        # A grouped module's own-named carrier, ahead of the scene's: a
        # module's (under the old first-in-file-order rule it was read).
        self.assertEqual(
            read(parsed("Module|data_export", "data_export"))["by"], "data_export"
        )
        self.assertEqual(
            read(parsed("Crate", "Module|data_export"))["by"], "Module|data_export"
        )

    def test_a_record_stating_no_material_is_no_other_schema(self):
        """REGRESSION (2026-10-24): a current record stating no material
        (``materials: []``) was warned as one of another schema. It states
        nothing over the file's own reading, and says nothing."""
        fbx = self._stingray(records={"materials": MaterialFamilies.payload([])})
        from pythontk.file_utils.mesh_convert import _fbx_materials

        with unittest.mock.patch.object(_fbx_materials.logger, "warning") as warned:
            entry = self._entry(fbx)
        self.assertEqual(entry["baseColorTexture"], "Lamp_BaseColor.png")
        self.assertFalse(
            [c for c in warned.call_args_list if "materials record" in str(c)],
            warned.call_args_list,
        )

    # -------------------------------------------------------------- applying
    def _glb(self, material, images):
        """A GLB whose one material is *material* (or whose materials are a
        list of them); *images* ``{name: bytes}`` embedded as data URIs,
        texture i sampling image i."""
        import base64

        names = list(images)
        gltf = {
            "asset": {"version": "2.0"},
            "materials": material if isinstance(material, list) else [material],
            "images": [
                {
                    "name": name,
                    "mimeType": "image/png",
                    "uri": "data:image/png;base64," + base64.b64encode(images[name]).decode(),
                }
                for name in names
            ],
            "textures": [{"source": i, "sampler": 0} for i in range(len(names))],
            "samplers": [{}],
        }
        return _write_glb_file(os.path.join(self.tmp, "converted.glb"), gltf)

    def _maps(self, **named):
        """Write each ``name=bytes`` map to disk; ``{file name: path}``."""
        paths = {}
        for name, data in named.items():
            path = os.path.join(self.tmp, name + ".png")
            with open(path, "wb") as fh:
                fh.write(data)
            paths[name + ".png"] = path
        return paths

    def test_apply_copies_the_record_material_whole(self):
        """The GLB's material IS the record's: its factors, alpha mode and
        sidedness, the converter's same-file binding reused, a stale map
        unbound, and the packed metallicRoughness image the record names --
        a lone metal map packed with roughness white, so the factor carries
        the constant -- embedded from the file the record names."""
        glb = self._glb(
            {
                "name": "orm",
                "alphaMode": "BLEND",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.5, 0.5, 0.5, 1.0],
                    "baseColorTexture": {"index": 0},
                },
                "emissiveTexture": {"index": 1},
                "emissiveFactor": [1.0, 1.0, 1.0],
            },
            {"Lamp_BaseColor.png": self._png((180, 120, 60)), "Stale.png": self._png((9, 9, 9))},
        )
        maps = self._maps(Lamp_Metallic=self._png(40, "L"))
        record, maps = stated_record(
            maps,
            {
                "name": "orm",
                "baseColorTexture": "Lamp_BaseColor.png",
                "metallicFactor": 1.0,
                "metallicTexture": "Lamp_Metallic.png",
                "roughnessFactor": 0.3,
                "doubleSided": True,
            },
        )
        written = MeshConvert.apply_glb_materials(glb, record, maps)
        self.assertEqual((written[0]["missing"], written[0]["orm"]), ([], True))
        with MeshConvert.open_glb(glb) as edit:
            (mat,) = edit.materials
            pbr = mat["pbrMetallicRoughness"]
            self.assertEqual(pbr["baseColorFactor"], [1.0, 1.0, 1.0, 1.0])
            self.assertEqual(pbr["baseColorTexture"], {"index": 0})
            self.assertNotIn("emissiveTexture", mat)
            self.assertEqual(mat["emissiveFactor"], [0.0, 0.0, 0.0])
            self.assertEqual((mat["alphaMode"], mat["doubleSided"]), ("OPAQUE", True))
            self.assertEqual((pbr["metallicFactor"], pbr["roughnessFactor"]), (1.0, 0.3))
            image = edit.image_for_texture(pbr["metallicRoughnessTexture"]["index"])
            self.assertEqual(edit.image_label(image), "Lamp_ORM.png")
            self.assertEqual(edit.channel_extrema(image, "G"), (255, 255))
            self.assertEqual(edit.channel_extrema(image, "B"), (40, 40))
            self.assertNotIn("occlusionTexture", mat)
            assert_materials_equal(self, edit, record)

    def test_a_colour_map_it_cannot_find_is_unbound_and_reported(self):
        """The record says which file the slot samples: one the GLB cannot
        hold is no binding, never the converter's other file."""
        glb = self._glb(
            {"name": "orm", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}},
            {"Old.png": self._png((1, 2, 3))},
        )
        record, _maps = stated_record({}, {"name": "orm", "baseColorTexture": "New.png"})
        with self.assertLogs("pythontk", level="WARNING"):
            written = MeshConvert.apply_glb_materials(glb, record, {})
        self.assertEqual(written[0]["missing"], ["New.png"])
        with MeshConvert.open_glb(glb) as edit:
            self.assertNotIn(
                "baseColorTexture", edit.materials[0]["pbrMetallicRoughness"]
            )

    def test_a_packed_map_it_cannot_find_reads_the_absent_values(self):
        """The converter's own packing is never trusted in its place -- it is
        the one measured solid white, metal everywhere: a missing packed map
        reads dielectric where its metal factor stood in for it and matte (its
        roughness factor over white), as an absent channel does in every
        packing, and the material is not exempt from the metallic=1 guard."""
        glb = self._glb(
            {
                "name": "orm",
                "pbrMetallicRoughness": {"metallicRoughnessTexture": {"index": 0}},
            },
            {"converter_orm.png": self._png((255, 255, 255))},
        )
        record = MaterialFamilies.payload(
            [
                MaterialFamilies.entry(
                    "pbr",
                    "orm",
                    {},
                    {"metallic": "Lost_Metallic.png", "roughness": "Lost_Roughness.png"},
                )
            ]
        )
        with self.assertLogs("pythontk", level="WARNING"):
            (written,) = MeshConvert.apply_glb_materials(glb, record, {})
        self.assertEqual(written["missing"], ["Lost_ORM.png"])
        self.assertFalse(written["orm"])
        with MeshConvert.open_glb(glb) as edit:
            pbr = edit.materials[0]["pbrMetallicRoughness"]
            self.assertNotIn("metallicRoughnessTexture", pbr)
            self.assertEqual((pbr["metallicFactor"], pbr["roughnessFactor"]), (0.0, 1.0))

    def test_a_missing_map_reads_as_the_one_rule(self):
        """``MaterialFamilies.absent_map``, which unitytk mirrors: a missing
        packed map keeps a constant metal and the roughness factor; a
        missing emission map is no glow (never its factor over white).
        Added: 2026-11-05"""
        glb = self._glb([{"name": "cast"}, {"name": "glow"}], {})
        record = MaterialFamilies.payload(
            [
                MaterialFamilies.entry(
                    "pbr", "cast", {"metallic": 0.3}, {"roughness": "Lost_Roughness.png"}
                ),
                MaterialFamilies.entry(
                    "pbr",
                    "glow",
                    {"emission_weight": 4.0},
                    {"emission": "Lost_Emissive.png"},
                ),
            ]
        )
        with self.assertLogs("pythontk", level="WARNING"):
            MeshConvert.apply_glb_materials(glb, record, {})
        with MeshConvert.open_glb(glb) as edit:
            cast, glow = edit.materials
            pbr = cast["pbrMetallicRoughness"]
            self.assertEqual((pbr["metallicFactor"], pbr["roughnessFactor"]), (0.3, 1.0))
            self.assertEqual(glow["emissiveFactor"], [0.0, 0.0, 0.0])
            self.assertNotIn("emissiveTexture", glow)
            self.assertNotIn("extensions", glow)
            self.assertNotIn(
                "KHR_materials_emissive_strength", edit.gltf.get("extensionsUsed") or []
            )

    def test_emission_past_one_rides_the_strength_extension(self):
        glb = self._glb({"name": "orm"}, {})
        record, _maps = stated_record(
            {}, {"name": "orm", "emissiveFactor": [4.0, 2.0, 0.0], "unlit": True}
        )
        MeshConvert.apply_glb_materials(glb, record, {})
        with MeshConvert.open_glb(glb) as edit:
            (mat,) = edit.materials
            self.assertEqual(mat["emissiveFactor"], [1.0, 0.5, 0.0])
            self.assertEqual(
                mat["extensions"]["KHR_materials_emissive_strength"], {"emissiveStrength": 4.0}
            )
            self.assertIn("KHR_materials_unlit", mat["extensions"])
            self.assertIn("KHR_materials_unlit", edit.gltf["extensionsUsed"])

    def test_a_texture_transform_rides_each_map_and_the_extension_is_listed(self):
        """KHR_texture_transform, copied with each map. Added: 2026-11-05"""
        glb = self._glb({"name": "orm"}, {})
        maps = self._maps(Tile_BaseColor=self._png((10, 20, 30)))
        record, maps = stated_record(
            maps,
            {
                "name": "orm",
                "baseColorTexture": "Tile_BaseColor.png",
                "uvTransform": {"offset": [0.0, 0.0], "scale": [4.0, 4.0]},
            },
        )
        MeshConvert.apply_glb_materials(glb, record, maps)
        with MeshConvert.open_glb(glb) as edit:
            (mat,) = edit.materials
            info = mat["pbrMetallicRoughness"]["baseColorTexture"]
            self.assertEqual(
                info["extensions"]["KHR_texture_transform"],
                {"offset": [0.0, -3.0], "scale": [4.0, 4.0]},
            )
            self.assertIn("KHR_texture_transform", edit.gltf["extensionsUsed"])
            assert_materials_equal(self, edit, record)

    def test_an_extension_no_material_uses_leaves_the_lists(self):
        """REGRESSION (2026-10-24): ``extensionsUsed`` was only added to, so a
        record that un-unlit a material or stated no emission past one left
        the file declaring an extension nothing used -- and requiring it, where
        the converter had. The lists state what the materials use."""
        glb = self._glb(
            {
                "name": "orm",
                "extensions": {
                    "KHR_materials_unlit": {},
                    "KHR_materials_emissive_strength": {"emissiveStrength": 4.0},
                },
            },
            {},
        )
        both = ["KHR_materials_emissive_strength", "KHR_materials_unlit"]
        with MeshConvert.open_glb(glb) as edit:
            edit.gltf["extensionsUsed"] = both + ["KHR_texture_transform", "KHR_other"]
            edit.gltf["extensionsRequired"] = list(both)
            edit.dirty = True
        record, _maps = stated_record({}, {"name": "orm", "emissiveFactor": [0.5, 0.0, 0.0]})
        MeshConvert.apply_glb_materials(glb, record, {})
        with MeshConvert.open_glb(glb) as edit:
            (mat,) = edit.materials
            self.assertNotIn("extensions", mat)
            self.assertEqual(edit.gltf["extensionsUsed"], ["KHR_other"])
            self.assertNotIn("extensionsRequired", edit.gltf)

    def test_a_mask_states_its_cutoff(self):
        """FBX2glTF reads ``alphaMode`` off the base colour's alpha alone, so a
        cutout reaches the GLB as BLEND: the record's MASK and cutoff win."""
        glb = self._glb({"name": "orm", "alphaMode": "BLEND"}, {})
        record, _maps = stated_record(
            {}, {"name": "orm", "alphaMode": "MASK", "alphaCutoff": 0.25}
        )
        MeshConvert.apply_glb_materials(glb, record, {})
        with MeshConvert.open_glb(glb) as edit:
            (mat,) = edit.materials
            self.assertEqual((mat["alphaMode"], mat["alphaCutoff"]), ("MASK", 0.25))

    def test_any_other_mode_drops_a_stale_cutoff(self):
        """glTF reads ``alphaCutoff`` under MASK alone; a blend keeps none."""
        glb = self._glb({"name": "orm", "alphaMode": "MASK", "alphaCutoff": 0.5}, {})
        record, _maps = stated_record({}, {"name": "orm", "alphaMode": "BLEND"})
        MeshConvert.apply_glb_materials(glb, record, {})
        with MeshConvert.open_glb(glb) as edit:
            (mat,) = edit.materials
            self.assertEqual(mat["alphaMode"], "BLEND")
            self.assertNotIn("alphaCutoff", mat)


@unittest.skipUnless(_FBX2GLTF, "FBX2glTF is not installed (nothing is downloaded here)")
class TestTakeClipsSpanTheirWindows(unittest.TestCase):
    """Every take the FBX declares is a clip of its name spanning its window.

    Bug: a take that holds still -- a hold, a title shot, which can still
    carry a description -- has no glTF animation (glTF forbids an empty one
    and FBX2glTF drops it), so it vanished from the page's picker and its
    sequence; at HEAD it was a window in the whole-timeline take. And a take
    ending off the converter's 1/30 s grid lost its last key, its clip cut
    short. Fixed: 2026-11-07 -- the build makes each declared take a clip,
    in take order, and gives one whose channels end before its window (or
    that has none) a STEP hold on a scene root holding its rest, to the
    window's end. Real FBX2glTF over a real FBX (``build_curve_fbx``)."""

    FPS = 30.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_takespans_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _convert(self, takes, windows, **kwargs):
        from test_fbx_media import build_curve_fbx

        fbx = build_curve_fbx(
            os.path.join(self.tmp, "takes.fbx"),
            takes,
            windows=windows,
            poses={"box": {"Lcl Translation": (11.0, 0.0, 0.0)}},
        )
        glb = os.path.join(self.tmp, "takes.glb")
        MeshConvert.fbx_to_glb(fbx, dst=glb, overwrite=True, prompt=False, lightmaps=False, **kwargs)
        return glb

    @staticmethod
    def _clips(glb):
        """``[(name, (first, last) seconds, [(node name, path, interpolation)])]``."""
        with MeshConvert.open_glb(glb) as edit:
            gltf = edit.gltf
            out = []
            for animation in gltf.get("animations") or []:
                channels = [
                    (
                        gltf["nodes"][c["target"]["node"]].get("name"),
                        c["target"]["path"],
                        animation["samplers"][c["sampler"]].get("interpolation", "LINEAR"),
                    )
                    for c in animation["channels"]
                ]
                out.append((animation["name"], MeshConvert._animation_time_span(edit, animation), channels))
        return out

    #: Three takes: ShotA moves; ShotB holds the box at rest (the converter
    #: drops the take); ShotC keys nothing.
    STILL_TAKES = {
        "ShotA": {("box", "Lcl Translation", "d|X"): [(f, float(f)) for f in range(1, 12)]},
        "ShotB": {("box", "Lcl Translation", "d|X"): [(f, 11.0) for f in range(12, 22)]},
        "ShotC": {},
    }
    STILL_WINDOWS = {"ShotA": (1, 11), "ShotB": (12, 21), "ShotC": (22, 30)}

    def test_a_still_take_is_a_clip_spanning_its_window(self):
        """With the three declared shots: each is a clip, in take order,
        spanning its window from its t=0; the moving take keeps its own
        channel alone, and each still one plays one STEP hold on the root."""
        glb = self._convert(
            self.STILL_TAKES,
            self.STILL_WINDOWS,
            data_export={SceneRecords.SHOTS.key: _shot_record(["ShotA", "ShotB", "ShotC"])},
        )
        clips = self._clips(glb)
        self.assertEqual([c[0] for c in clips], ["ShotA", "ShotB", "ShotC"])
        spans = {name: span for name, span, _c in clips}
        self.assertAlmostEqual(spans["ShotA"][1], 10 / self.FPS, places=5)
        self.assertAlmostEqual(spans["ShotB"][1], 9 / self.FPS, places=5)
        self.assertAlmostEqual(spans["ShotC"][1], 8 / self.FPS, places=5)
        channels = {name: chans for name, _s, chans in clips}
        self.assertEqual(channels["ShotA"], [("box", "translation", "LINEAR")], "no hold where none is needed")
        for still in ("ShotB", "ShotC"):
            self.assertEqual(channels[still], [("RootNode", "translation", "STEP")], still)

    def test_a_take_no_shot_names_gains_no_clip(self):
        """Without shots, a take the converter wrote no clip for gains none: a
        static write's (Maya declares a take for every write), or one whose
        motion the converter lost, is no shot to keep -- as before."""
        clips = self._clips(self._convert(self.STILL_TAKES, self.STILL_WINDOWS))
        self.assertEqual([c[0] for c in clips], ["ShotA"])
        self.assertEqual(clips[0][2], [("box", "translation", "LINEAR")])

    def test_a_take_whose_last_key_the_converter_drops_spans_its_window(self):
        """The take ends at frame 10.6, off the 1/30 s grid: the converter's
        clip ends at frame 9 of its 9.6. The hold carries it to the window's
        end; the converter's channel is left as it wrote it."""
        glb = self._convert(
            {"Off": {("box", "Lcl Translation", "d|X"): [(1, 0.0), (5, 4.0), (10.6, 9.6)]}},
            {"Off": (1, 10.6)},
        )
        ((name, span, channels),) = self._clips(glb)
        self.assertEqual(name, "Off")
        self.assertAlmostEqual(span[1], 9.6 / self.FPS, places=5)
        self.assertEqual(
            channels, [("box", "translation", "LINEAR"), ("RootNode", "translation", "STEP")]
        )

    def test_the_no_clip_warning_names_only_a_take_split_miss(self):
        """With every declared take a clip, the shots that have none are the
        ones the file carries no take of -- a take split that missed."""
        record = _shot_record(["ShotA", "ShotB", "ShotX"])
        with self.assertLogs("pythontk", level="WARNING") as caught:
            self._convert(
                {
                    "ShotA": {("box", "Lcl Translation", "d|X"): [(f, float(f)) for f in range(1, 12)]},
                    "ShotB": {},
                },
                {"ShotA": (1, 11), "ShotB": (12, 21)},
                data_export={SceneRecords.SHOTS.key: record},
            )
        missed = [m for m in caught.output if "have no clip in the GLB" in m]
        self.assertEqual(len(missed), 1, caught.output)
        self.assertIn("(ShotX)", missed[0])


@unittest.skipUnless(_FBX2GLTF, "FBX2glTF is not installed (nothing is downloaded here)")
class TestMaterialsRecordThroughTheConverter(unittest.TestCase):
    """A GLB built from an FBX alone states its materials as the FBX does.

    The reported case (2026-10-20): opening a delivered FBX in the WebXR
    preview showed no emission and no glass, and every albedo halved -- the
    converter's reading, which only a live scene's sidecar repaired.
    """

    setUp = TestMaterialsRecord.setUp
    _png = staticmethod(TestMaterialsRecord._png)
    _stingray = TestMaterialsRecord._stingray

    def test_a_glb_built_from_the_fbx_alone_reads_its_materials(self):
        from pythontk import GlbPipeline

        fbx = self._stingray()
        built = GlbPipeline.build(fbx, os.path.join(self.tmp, "alone.glb"), downsize=False)
        self.assertEqual([r["material"] for r in built["materials"]], ["orm"])
        with MeshConvert.open_glb(built["glb"]) as edit:
            (mat,) = edit.materials
            pbr = mat["pbrMetallicRoughness"]
            self.assertEqual(pbr["baseColorFactor"], [1.0, 1.0, 1.0, 1.0])
            self.assertEqual([round(c, 6) for c in mat["emissiveFactor"]], [0.75] * 3)
            self.assertIn("emissiveTexture", mat)
            image = edit.image_for_texture(pbr["metallicRoughnessTexture"]["index"])
            self.assertEqual(edit.channel_extrema(image, "B"), (30, 30))
            self.assertEqual(edit.channel_extrema(image, "G"), (200, 200))

    def test_a_verifying_reader_vouches_for_the_orms_the_record_packed(self):
        """The ORM the record packed is the authored maps' (metal 30,
        roughness 200), not the converter's white one, so a verifying reader
        -- which decodes every ORM and reports only a destructive one -- finds
        nothing to say of it (measured on the 2026-10-20 production export:
        3 images over 56 materials flagged while an envelope judged them)."""
        fbx = self._stingray()
        dst = os.path.join(self.tmp, "verified.glb")
        MeshConvert.fbx_to_glb(
            fbx,
            dst,
            sidecar=HandoffRecord.block("glb", [], {"application": "test"}),
            auto_install=False,
        )
        report = MeshConvert.verify_glb(dst)
        self.assertFalse(report["problems"], report["problems"])
        self.assertNotIn("orm", report)
        self.assertEqual(report["handoff"]["source"], {"application": "test"})

    def test_a_map_only_the_record_names_reaches_the_glb_built_from_the_file(self):
        """An openPBRSurface reaches the FBX as a placeholder with no map, so
        the map its ``materials`` entry names is in the file only once the
        dependency pass embeds it; a GLB built from the delivered file alone
        -- no folder holding the map -- then binds it from the file itself.
        Added: 2026-10-31"""
        from pythontk import FbxMedia, GlbPipeline

        maps = os.path.join(self.tmp, "sourceimages")
        os.makedirs(maps)
        with open(os.path.join(maps, "OP_Base.png"), "wb") as fh:
            fh.write(self._png((40, 160, 220)))
        record, _maps = stated_record(
            {}, {"name": "orm", "baseColorTexture": "OP_Base.png", "metallicFactor": 0.0, "roughnessFactor": 0.5}
        )
        fbx = self._stingray(maps={}, records={"materials": record})
        report = FbxMedia.embed_dependencies(fbx, search_dirs=[maps])
        self.assertEqual(report["embedded"], ["OP_Base.png"])
        shutil.rmtree(maps)  # the file alone
        built = GlbPipeline.build(fbx, os.path.join(self.tmp, "alone.glb"), downsize=False)
        with MeshConvert.open_glb(built["glb"]) as edit:
            (mat,) = edit.materials
            texture = mat["pbrMetallicRoughness"].get("baseColorTexture")
            self.assertIsNotNone(texture, "the record's map did not reach the GLB")
            # The map's own pixels (the build re-encodes it: a code value apart).
            image = edit.image_for_texture(texture["index"])
            for channel, value in (("R", 40), ("G", 160), ("B", 220)):
                low, high = edit.channel_extrema(image, channel)
                self.assertLessEqual(abs(low - value), 2, channel)
                self.assertLessEqual(abs(high - value), 2, channel)

    def test_the_record_the_fbx_carries_states_the_glb_and_leaves_its_extras(self):
        """The export's statement wins over the file's own reading (an unlit
        the material cannot say), and the GLB keeps no second account of its
        materials in the converter's transcription of the carrier: its
        materials ARE the record's."""
        from pythontk import GlbPipeline

        carried = TestMaterialsRecord._unlit_record("orm")
        fbx = self._stingray(records={"materials": carried})
        built = GlbPipeline.build(fbx, os.path.join(self.tmp, "stated.glb"), downsize=False)
        with MeshConvert.open_glb(built["glb"]) as edit:
            (mat,) = edit.materials
            self.assertIn("KHR_materials_unlit", mat.get("extensions") or {})
            self.assertEqual(
                [round(c, 6) for c in mat["pbrMetallicRoughness"]["baseColorFactor"]],
                [0.2, 0.2, 0.2, 1.0],
            )
            self.assertIsNone(
                MeshConvert.data_export_channel(edit.gltf, MeshConvert.MATERIALS_KEY)
            )
            assert_materials_equal(self, edit, carried)

    def test_the_glbs_materials_are_the_records_and_the_validator_reads_them(self):
        """Every feature the record carries -- a packed metallicRoughness
        image, emission above 1, a texture transform -- reaches the GLB built
        from the FBX as the record states it, and the Khronos validator reads
        the GLB clean of errors. Added: 2026-11-05"""
        from pythontk import GlbPipeline, GltfValidator

        folder = os.path.join(self.tmp, "maps")
        os.makedirs(folder)
        maps = {}
        for name, value, mode in (
            ("Hot_BaseColor.png", (200, 80, 40), "RGB"),
            ("Hot_Metallic.png", 180, "L"),
            ("Hot_Roughness.png", 60, "L"),
            ("Hot_Emissive.png", (255, 255, 255), "RGB"),
        ):
            maps[name] = os.path.join(folder, name)
            with open(maps[name], "wb") as fh:
                fh.write(self._png(value, mode))
        record, maps = stated_record(
            maps,
            {
                "name": "orm",
                "baseColorTexture": "Hot_BaseColor.png",
                "metallicTexture": "Hot_Metallic.png",
                "roughnessTexture": "Hot_Roughness.png",
                "emissiveTexture": "Hot_Emissive.png",
                "emissiveFactor": [3.0, 3.0, 3.0],
                "uvTransform": {"offset": [0.5, 0.0], "scale": [2.0, 2.0]},
            },
        )
        fbx = self._stingray(maps={}, records={"materials": record})
        FbxMedia.embed_dependencies(fbx, search_dirs=[folder])
        shutil.rmtree(folder)
        built = GlbPipeline.build(fbx, os.path.join(self.tmp, "every.glb"), downsize=False)
        with MeshConvert.open_glb(built["glb"]) as edit:
            assert_materials_equal(self, edit, record)
        from conftest import gltf_validator

        report = GltfValidator.validate(built["glb"], gltf_validator())
        self.assertEqual(
            report["issues"]["numErrors"], 0, GltfValidator.messages(report, 0)
        )


class TestFacadeComposition(unittest.TestCase):
    """MeshConvert is a facade over its parts, not a class built from mixins:
    each job is a collaborator it builds on the class it is read through, and
    the whole public surface is the facade's own."""

    def test_the_facade_inherits_no_job(self):
        from pythontk import HelpMixin

        self.assertEqual(MeshConvert.__bases__, (HelpMixin,))

    def test_every_public_name_is_the_facades_own(self):
        from pythontk import HelpMixin

        inherited = sorted(
            name
            for name in dir(MeshConvert)
            if not name.startswith("_")
            and name not in vars(MeshConvert)
            and not hasattr(HelpMixin, name)
        )
        self.assertEqual(inherited, [])

    def test_a_subclass_override_reaches_a_parts_own_calls(self):
        class Stop(Exception):
            pass

        class Sub(MeshConvert):
            @classmethod
            def open_glb(cls, glb):
                raise Stop(glb)

        with self.assertRaises(Stop):
            Sub.read_glb_handoff("never_opened.glb")

    def test_a_subclass_dialect_row_reads_with_the_subclass_method(self):
        """``FBX_FAMILIES``' seam -- a new dialect is one row and its method --
        holds on a subclass: the row's method is found on the class read
        through, as the row is."""
        from test_fbx_media import build_stingray_fbx

        tmp = tempfile.mkdtemp(prefix="mc_dialect_")
        self.addCleanup(shutil.rmtree, tmp, True)
        path = build_stingray_fbx(
            os.path.join(tmp, "m.fbx"), {}, props={"custom_marker": 1.0}, name="m"
        )

        class Sub(MeshConvert):
            FBX_FAMILIES = (
                ("pbr", "Maya|custom_marker", "_fbx_custom_props"),
            ) + MeshConvert.FBX_FAMILIES

            @staticmethod
            def _fbx_custom_props(props, notes, name):
                return {"base_color": (1.0, 0.0, 0.0), "metallic": 0.0}

        (entry,) = Sub.read_fbx_materials(path)["materials"]
        self.assertEqual(
            entry["pbrMetallicRoughness"]["baseColorFactor"][:3], [1.0, 0.0, 0.0]
        )

    def test_every_delegation_forwards_its_own_signature(self):
        """Each one-statement delegation hands its part a parameter list the
        part's method declares the same (names, kinds, defaults), every
        parameter by its own name: the facade's copy of a signature cannot
        drift from the implementation's."""
        import ast
        import inspect

        from pythontk.core_utils.class_property import ClassProperty
        from pythontk.file_utils.mesh_convert import _mesh_convert as module

        parts = {
            name: value.getter
            for name, value in vars(MeshConvert).items()
            if isinstance(value, ClassProperty) and isinstance(value.getter, type)
        }
        tree = ast.parse(inspect.getsource(module))
        facade = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MeshConvert"
        )
        checked, problems = 0, []
        for fn in facade.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            body = [
                s
                for s in fn.body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
            ]
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            call = body[0].value
            if not isinstance(call, ast.Call):
                continue
            chain, node = [], call.func
            while isinstance(node, ast.Attribute):
                chain.insert(0, node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                chain.insert(0, node.id)
            if len(chain) == 3 and chain[1] in parts:  # cls.<part>.<name>
                target = getattr(parts[chain[1]], chain[2], None)
            elif len(chain) == 2 and chain[0] in vars(module) and chain[0] != "cls":
                target = getattr(vars(module)[chain[0]], chain[1], None)  # Part.name
            else:
                continue  # an inlined statement, not a delegation
            checked += 1
            if target is None:
                problems.append(f"{fn.name}: no {'.'.join(chain)}")
                continue
            wanted = [
                (p.name, p.kind, p.default)
                for p in inspect.signature(target).parameters.values()
                if p.name not in ("self", "cls")
            ]
            given = [
                (p.name, p.kind, p.default)
                for p in inspect.signature(
                    getattr(MeshConvert, fn.name)
                ).parameters.values()
            ]
            if given != wanted:
                problems.append(f"{fn.name}: signature {given} != {wanted}")
            names = [p[0] for p in wanted]
            passed = [
                (names[i] if i < len(names) else None, ast.unparse(a).lstrip("*"))
                for i, a in enumerate(call.args)
            ] + [(k.arg, ast.unparse(k.value)) for k in call.keywords]
            for name, expr in passed:
                if name is not None and name != expr:
                    problems.append(f"{fn.name}: {name} <- {expr}")
            if sorted(e for _, e in passed) != sorted(names):
                problems.append(f"{fn.name}: forwards {passed}, declares {names}")
        self.assertGreater(checked, 80)
        self.assertEqual(problems, [])


class TestNonFiniteKeys(unittest.TestCase):
    """A converted GLB's animation values and key times are finite, as glTF
    requires.

    Repro (``test_assets/fbx/maya_tangents.fbx``, FBX2glTF 0.13.1): the file
    keys node ``tangents`` at scale 0 on frame 1, and the converter, which bakes
    each node's local matrix per frame and decomposes it, writes that frame's
    rotation as a NaN quaternion -- a degenerate matrix has no rotation to
    recover. The GLB then failed its own NaN gate, and ``gltf_to_fbx`` crashed
    on it ("cannot convert float NaN to integer").
    """

    LOGGER = "pythontk.file_utils.mesh_convert._fbx2gltf"
    NAN, INF = float("nan"), float("inf")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_nonfinite_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    @staticmethod
    def _turn(degrees):
        """A rotation of *degrees* about +Y, as a glTF quaternion."""
        half = math.radians(degrees) / 2.0
        return (0.0, math.sin(half), 0.0, math.cos(half))

    def _glb(self, clips, image=False, name="anim.glb"):
        """Nodes ``spin``, ``move``, ``void`` and *clips* ``{clip name: [(node,
        path, values, options)]}``, three keys a frame apart unless *options*
        says ``times``; ``interpolation`` sets the sampler's, ``bounds`` keeps
        the output's min/max, ``share`` reads the output of the clip's channel
        at that position, ``pointer`` targets a KHR_animation_pointer. With
        *image*, a PNG lands in the BIN after every animation accessor."""
        from test_gltf2fbx import GltfBuilder, png_bytes

        b = GltfBuilder()
        b.gltf["nodes"] = [{"name": "spin"}, {"name": "move"}, {"name": "void"}]
        b.gltf["scenes"] = [{"nodes": [0, 1, 2]}]
        animations = []
        for clip, channels in clips.items():
            samplers, wired, outputs = [], [], []
            for node, path, values, options in channels:
                times = options.get("times", [0.0, 1 / 24.0, 2 / 24.0])
                source = b.accessor(times, "SCALAR")
                finite = [t for t in times if math.isfinite(t)]
                b.accessors[source]["min"] = [min(finite)]
                b.accessors[source]["max"] = [max(finite)]
                if "share" in options:
                    output = outputs[options["share"]]
                else:
                    width = len(values[0])
                    kind = {1: "SCALAR", 3: "VEC3", 4: "VEC4"}[width]
                    output = b.accessor(values, kind)
                    if options.get("bounds"):
                        b.accessors[output]["min"] = [
                            min(v[i] for v in values) for i in range(width)
                        ]
                        b.accessors[output]["max"] = [
                            max(v[i] for v in values) for i in range(width)
                        ]
                    else:
                        b.accessors[output].pop("min", None)
                        b.accessors[output].pop("max", None)
                outputs.append(output)
                sampler = {"input": source, "output": output}
                if "interpolation" in options:
                    sampler["interpolation"] = options["interpolation"]
                samplers.append(sampler)
                target = {"node": node, "path": path}
                if "pointer" in options:
                    target = {
                        "path": "pointer",
                        "extensions": {
                            "KHR_animation_pointer": {"pointer": options["pointer"]}
                        },
                    }
                wired.append({"sampler": len(samplers) - 1, "target": target})
            animations.append({"name": clip, "samplers": samplers, "channels": wired})
        b.gltf["animations"] = animations
        if image:
            b.gltf["images"] = []
            b.image(png_bytes((10, 20, 30, 255)), "after.png")
        return b.write_glb(os.path.join(self.tmp, name))

    @staticmethod
    def _channel(glb, node, path, clip=0):
        """``(times, values, output accessor)`` of one channel, as written."""
        reader = GlbReader.load(glb)
        animation = reader.gltf["animations"][clip]
        for channel in animation["channels"]:
            if channel["target"] == {"node": node, "path": path}:
                sampler = animation["samplers"][channel["sampler"]]
                return (
                    [t[0] for t in reader.accessor(sampler["input"])],
                    reader.accessor(sampler["output"]),
                    reader.gltf["accessors"][sampler["output"]],
                )
        return None

    def _repair(self, glb, level="WARNING"):
        with self.assertLogs(self.LOGGER, level) as logs:
            report = MeshConvert.fix_glb_nonfinite_keys(glb)
        return report, logs.output

    def assertClose(self, got, want, places=5):
        self.assertEqual(len(got), len(want))
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=places, msg=f"{got} != {want}")

    # ---------------------------------------------------------------- values
    def test_an_interior_key_is_interpolated_by_its_time(self):
        """Between two finite keys, the value the motion had there: a slerp
        for a rotation, a lerp otherwise, by time (here a third of the way)."""
        times = [0.0, 1 / 24.0, 3 / 24.0]
        glb = self._glb(
            {
                "Take": [
                    (
                        0,
                        "rotation",
                        [self._turn(0), (self.NAN,) * 4, self._turn(30)],
                        {"times": times},
                    ),
                    (
                        1,
                        "translation",
                        [(0, 0, 0), (self.INF, 0, 0), (3, 0, 0)],
                        {"times": times, "bounds": True},
                    ),
                ]
            }
        )
        report, said = self._repair(glb)
        self.assertEqual(report, {"keys": 2, "channels": 2, "dropped": 0})
        self.assertClose(self._channel(glb, 0, "rotation")[1][1], self._turn(10))
        _, moved, accessor = self._channel(glb, 1, "translation")
        self.assertClose(moved[1], (1.0, 0.0, 0.0))
        # The bounds an output states are its values' bounds again.
        self.assertEqual(
            (accessor["min"], accessor["max"]), ([0.0, 0.0, 0.0], [3.0, 0.0, 0.0])
        )
        joined = "\n".join(said)
        self.assertIn("'Take'", joined)
        self.assertIn("'spin'", joined)
        self.assertIn("'move'", joined)
        self.assertEqual(GlbReader.load(glb).nan_findings(deep=True), [])

    def test_an_end_key_holds_its_nearest_finite_key(self):
        glb = self._glb(
            {
                "Take": [
                    (
                        0,
                        "rotation",
                        [(self.NAN,) * 4, (self.NAN,) * 4, self._turn(30)],
                        {},
                    ),
                    (1, "translation", [(0, 0, 0), (1, 0, 0), (self.NAN, 0, 0)], {}),
                ]
            }
        )
        report, said = self._repair(glb)
        self.assertEqual(report, {"keys": 3, "channels": 2, "dropped": 0})
        turned = self._channel(glb, 0, "rotation")[1]
        self.assertClose(turned[0], self._turn(30))
        self.assertClose(turned[1], self._turn(30))
        self.assertClose(self._channel(glb, 1, "translation")[1][2], (1.0, 0.0, 0.0))
        # Only a rotation's report says why a converter writes one.
        (moved,) = [line for line in said if "'move'" in line]
        self.assertNotIn("rotation to recover", moved)
        (spun,) = [line for line in said if "'spin'" in line]
        self.assertIn("rotation to recover", spun)

    def test_a_cubic_key_is_interpolated_with_flat_tangents(self):
        zero, one, two = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)
        glb = self._glb(
            {
                "Take": [
                    (
                        1,
                        "translation",
                        [zero, zero, one, one, (self.NAN, 0, 0), one, one, two, zero],
                        {"interpolation": "CUBICSPLINE"},
                    )
                ]
            }
        )
        report, _ = self._repair(glb)
        self.assertEqual(report, {"keys": 1, "channels": 1, "dropped": 0})
        values = self._channel(glb, 1, "translation")[1]
        self.assertEqual(
            values[3:6], [zero, one, zero]
        )  # in-tangent, value, out-tangent
        self.assertEqual(values[:3] + values[6:], [zero, zero, one, one, two, zero])

    def test_a_shared_output_is_repaired_once_and_said_once(self):
        glb = self._glb(
            {
                "A": [
                    (
                        0,
                        "rotation",
                        [self._turn(0), (self.NAN,) * 4, self._turn(20)],
                        {},
                    )
                ],
                "B": [
                    (1, "translation", [(0, 0, 0)] * 3, {}),
                    (0, "rotation", None, {"share": 0}),
                ],
            }
        )
        # Clip B's second channel reads clip B's first output: rewire it to A's.
        with MeshConvert.open_glb(glb) as edit:
            first = edit.gltf["animations"][0]["samplers"][0]["output"]
            edit.gltf["animations"][1]["samplers"][1]["output"] = first
            edit.dirty = True
        report, said = self._repair(glb)
        self.assertEqual(report, {"keys": 1, "channels": 2, "dropped": 0})
        self.assertEqual(len(said), 1, said)
        self.assertIn("'A'", said[0])
        self.assertIn("'B'", said[0])
        self.assertEqual(said[0].count("'spin'"), 1, said[0])
        self.assertClose(
            self._channel(glb, 0, "rotation", clip=1)[1][1], self._turn(10)
        )

    def test_a_pointer_channel_is_named_by_its_pointer(self):
        pointer = "/materials/0/pbrMetallicRoughness/baseColorFactor"
        glb = self._glb(
            {
                "Fade": [
                    (
                        None,
                        None,
                        [(1, 1, 1, 1), (self.NAN,) * 4, (1, 1, 1, 0)],
                        {"pointer": pointer},
                    )
                ]
            }
        )
        report, said = self._repair(glb)
        self.assertEqual(report["keys"], 1)
        self.assertIn(repr(pointer), said[0])
        self.assertNotIn("None", said[0])

    # --------------------------------------------------------- key times
    def test_a_non_finite_key_time_takes_its_place_in_the_spacing(self):
        """A key time has no value to borrow: an interior one takes its place
        between its finite neighbours by index, an end one its neighbours'
        spacing; the input's bounds follow."""
        glb = self._glb(
            {
                "Take": [
                    (
                        1,
                        "translation",
                        [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
                        {"times": [0.0, self.NAN, 2 / 24.0]},
                    ),
                    (
                        0,
                        "rotation",
                        [self._turn(0)] * 3,
                        {"times": [0.0, 1 / 24.0, self.INF]},
                    ),
                ]
            }
        )
        report, said = self._repair(glb)
        self.assertEqual(report, {"keys": 2, "channels": 2, "dropped": 0})
        self.assertClose(
            self._channel(glb, 1, "translation")[0], [0.0, 1 / 24.0, 2 / 24.0]
        )
        times = self._channel(glb, 0, "rotation")[0]
        self.assertClose(times, [0.0, 1 / 24.0, 2 / 24.0])
        reader = GlbReader.load(glb)
        source = reader.gltf["animations"][0]["samplers"][1]["input"]
        self.assertAlmostEqual(
            reader.gltf["accessors"][source]["max"][0], 2 / 24.0, places=6
        )
        self.assertTrue(any("key time" in line for line in said), said)

    # ------------------------------------------------------- what is dropped
    def test_a_channel_with_no_finite_key_is_dropped_and_the_file_stays_whole(self):
        """The hollow channel is FIRST, so dropping it renumbers the samplers
        the rest read; an image sits after the animation bytes, so the
        compaction moves it; a clip left with nothing is pruned."""
        bounded = [self._turn(0), self._turn(10), self._turn(20)]
        glb = self._glb(
            {
                "Take": [
                    (2, "scale", [(self.NAN,) * 3] * 3, {}),
                    (0, "rotation", bounded, {"bounds": True}),
                ],
                "Only": [(2, "translation", [(self.INF, 0, 0)] * 3, {})],
            },
            image=True,
        )
        before = MeshConvert._read_glb(glb)
        image = before.image_bytes(before.gltf["images"][0])
        report, said = self._repair(glb)
        self.assertEqual(report, {"keys": 0, "channels": 0, "dropped": 2})
        self.assertTrue(all("'void'" in line for line in said), said)
        after = MeshConvert._read_glb(glb)
        (animation,) = after.gltf["animations"]
        self.assertEqual(animation["name"], "Take")
        self.assertEqual(
            animation["channels"],
            [{"sampler": 0, "target": {"node": 0, "path": "rotation"}}],
        )
        self.assertEqual(len(animation["samplers"]), 1)
        _, turned, accessor = self._channel(glb, 0, "rotation")
        for got, want in zip(turned, bounded):
            self.assertClose(got, want)
        self.assertIn("min", accessor)
        self.assertEqual(after.image_bytes(after.gltf["images"][0]), image)
        self.assertEqual(GlbReader.load(glb).nan_findings(deep=True), [])

    def test_a_finite_file_is_left_as_it_is(self):
        glb = self._glb({"Take": [(0, "rotation", [(0, 0, 0, 1)] * 3, {})]})
        with open(glb, "rb") as fh:
            before = fh.read()
        self.assertEqual(
            MeshConvert.fix_glb_nonfinite_keys(glb),
            {"keys": 0, "channels": 0, "dropped": 0},
        )
        with open(glb, "rb") as fh:
            self.assertEqual(fh.read(), before)

    # ------------------------------------------------------ the converter
    @unittest.skipUnless(
        _FBX2GLTF, "FBX2glTF is not installed; this test downloads nothing"
    )
    def test_the_fixtures_conversion_ships_only_finite_keys(self):
        src = os.path.join(
            os.path.dirname(__file__), "test_assets", "fbx", "maya_tangents.fbx"
        )
        fbx = os.path.join(self.tmp, "maya_tangents.fbx")
        shutil.copy(src, fbx)
        with self.assertLogs(self.LOGGER, "WARNING") as logs:
            glb = MeshConvert.fbx_to_glb(fbx, os.path.join(self.tmp, "out.glb"))
        self.assertEqual(GlbReader.load(glb).nan_findings(deep=True), [])
        said = [line for line in logs.output if "non-finite" in line]
        self.assertEqual(len(said), 1, logs.output)
        for word in ("'Take 001'", "'tangents'", "rotation"):
            self.assertIn(word, said[0])

    @unittest.skipUnless(
        _FBX2GLTF, "FBX2glTF is not installed; this test downloads nothing"
    )
    def test_a_mid_animation_zero_scale_keeps_the_authored_turn(self):
        """The round trip that showed the hold was wrong: a node squashed flat
        in the middle of a spin (scale Y through 0 at 0.5 s) comes back turned
        as authored there, not paused at a neighbour's angle."""
        from test_gltf2fbx import GltfBuilder

        b = GltfBuilder()
        times = [k / 24.0 for k in range(25)]
        source = b.accessor(times, "SCALAR")
        b.accessors[source].update(min=[0.0], max=[1.0])
        spin = b.accessor([self._turn(10 * k) for k in range(25)], "VEC4")
        squash = b.accessor([(1.0, abs(k - 12) / 12.0, 1.0) for k in range(25)], "VEC3")
        for index in (spin, squash):
            b.accessors[index].pop("min", None)
            b.accessors[index].pop("max", None)
        b.gltf.update(
            scenes=[{"nodes": [0]}],
            nodes=[{"name": "squash"}],
            animations=[
                {
                    "name": "Take",
                    "samplers": [
                        {"input": source, "output": spin},
                        {"input": source, "output": squash},
                    ],
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "rotation"}},
                        {"sampler": 1, "target": {"node": 0, "path": "scale"}},
                    ],
                }
            ],
        )
        authored = b.write_glb(os.path.join(self.tmp, "squash.glb"))
        fbx = MeshConvert.gltf_to_fbx(authored, os.path.join(self.tmp, "squash.fbx"))
        glb = MeshConvert.fbx_to_glb(fbx, os.path.join(self.tmp, "squash_rt.glb"))
        reader = GlbReader.load(glb)
        self.assertEqual(reader.nan_findings(deep=True), [])
        node = reader.node_index("squash")
        animation = reader.gltf["animations"][0]
        (channel,) = [
            c
            for c in animation["channels"]
            if c["target"] == {"node": node, "path": "rotation"}
        ]
        sampler = animation["samplers"][channel["sampler"]]
        keys = [t[0] for t in reader.accessor(sampler["input"])]
        at = min(range(len(keys)), key=lambda k: abs(keys[k] - 0.5))
        self.assertAlmostEqual(keys[at], 0.5, places=4)
        got = reader.accessor(sampler["output"])[at]
        want = self._turn(120)
        self.assertAlmostEqual(
            abs(sum(a * b for a, b in zip(got, want))), 1.0, places=4
        )

    @unittest.skipUnless(
        _FBX2GLTF, "FBX2glTF is not installed; this test downloads nothing"
    )
    def test_a_flat_nodes_tilt_survives_the_converter(self):
        """A node scaled flat on one axis for the whole clip decomposes to a
        NaN rotation on every key, and FBX2glTF drops that channel itself:
        the node played unrotated before any repair here could run. The tilt
        is read back from the FBX (``MeshConvert.restore_glb_rotations``) --
        here on a take the converter left with no channel, so no time input
        either: one is minted on its bake grid."""
        for args, fps in (
            (None, MeshConvert.CONVERSION_BAKE_FPS),
            (["--anim-framerate", "bake60"], 60.0),
        ):
            with self.subTest(fps=fps):
                self._assert_flat_tilt(args, fps)

    def _assert_flat_tilt(self, args, fps):
        from test_gltf2fbx import GltfBuilder

        b = GltfBuilder()
        source = b.accessor([k / 24.0 for k in range(6)], "SCALAR")
        b.accessors[source].update(min=[0.0], max=[5 / 24.0])
        half = [math.radians(9 * k) for k in range(6)]  # 18 degrees a key about X
        tilt = b.accessor([(math.sin(h), 0.0, 0.0, math.cos(h)) for h in half], "VEC4")
        b.gltf.update(
            scenes=[{"nodes": [0]}],
            nodes=[{"name": "flat", "scale": [1.0, 0.0, 1.0]}],
            animations=[
                {
                    "name": "Take",
                    "samplers": [{"input": source, "output": tilt}],
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "rotation"}}
                    ],
                }
            ],
        )
        stem = os.path.join(self.tmp, f"flat_{fps:g}")
        authored = b.write_glb(stem + ".glb")
        fbx = MeshConvert.gltf_to_fbx(authored, stem + ".fbx")
        glb = MeshConvert.fbx_to_glb(fbx, stem + "_rt.glb", extra_args=args)
        reader = GlbReader.load(glb)
        node = reader.node_index("flat")
        rotations = [
            (a, c)
            for a in reader.gltf.get("animations") or []
            for c in a["channels"]
            if c["target"] == {"node": node, "path": "rotation"}
        ]
        self.assertTrue(rotations, "the flat node's tilt did not survive")
        ((animation, channel),) = rotations
        sampler = animation["samplers"][channel["sampler"]]
        times = [t[0] for t in reader.accessor(sampler["input"])]
        # The converter's grid: the take's 5/24 s, a frame of the bake rate.
        count = int(5 / 24.0 * fps) + 1
        self.assertEqual(len(times), count)
        for k, t in enumerate(times):
            self.assertAlmostEqual(t, k / fps, places=6)
        self.assertAlmostEqual(
            reader.gltf["accessors"][sampler["input"]]["max"][0], times[-1], places=6
        )
        # The authored tilt wherever it is read: 18 degrees a 1/24 s key.
        for t, (x, y, z, w) in zip(times, reader.accessor(sampler["output"])):
            half = math.radians(18 * 24 * t) / 2
            self.assertAlmostEqual(
                abs(x * math.sin(half) + w * math.cos(half)), 1.0, places=6
            )
        self.assertEqual(reader.nan_findings(deep=True), [])


@unittest.skipUnless(
    _FBX2GLTF, "FBX2glTF is not installed; this test downloads nothing"
)
class TestFlatNodeRotations(unittest.TestCase):
    """The rotation FBX2glTF drops from a node scaled flat on one axis is read
    back from its FBX, composed as the converter composes it.

    Each fixture pairs a flat node with a ROUND twin keyed alike: the
    converter writes the twin's rotation itself, so the twin is the reference
    for every convention at once -- rotation order, pre/post rotation, the axis
    turn of a root child, each take's time input and t = 0.
    """

    #: Two unit quaternions this close (``1 - |dot|``) are one rotation, to
    #: the float32 the converter writes.
    CLOSE = 1e-6

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshconvert_flat_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _converted(self, fbx):
        """FBX2glTF's own output for *fbx*, as :meth:`MeshConvert.fbx_to_glb`
        runs it, before any pass."""
        base = os.path.join(self.tmp, "raw")
        subprocess.run(
            [_FBX2GLTF, "-i", fbx, "-o", base, "--binary", "--user-properties"],
            check=True,
            capture_output=True,
        )
        return base + ".glb"

    @staticmethod
    def _channels(glb):
        """``{(animation, node name, path): (input, output, values)}``."""
        reader = GlbReader.load(glb)
        names = [node.get("name") for node in reader.gltf["nodes"]]
        found = {}
        for animation in reader.gltf.get("animations") or []:
            for channel in animation["channels"]:
                sampler = animation["samplers"][channel["sampler"]]
                key = (
                    animation["name"],
                    names[channel["target"]["node"]],
                    channel["target"]["path"],
                )
                found.setdefault(key, []).append(
                    (
                        sampler["input"],
                        sampler["output"],
                        reader.accessor(sampler["output"]),
                    )
                )
        return found

    @staticmethod
    def _rest(glb, name):
        reader = GlbReader.load(glb)
        return reader.gltf["nodes"][reader.node_index(name)].get("rotation")

    def assertSameRotations(self, got, want):
        self.assertEqual(len(got), len(want))
        for a, b in zip(got, want):
            self.assertLess(1 - abs(sum(x * y for x, y in zip(a, b))), self.CLOSE)

    def assertTwins(self, before, after, take, flat, twin):
        """*flat* gained one rotation channel in *take*, on the input its
        *twin*'s shares, turning as the converter turned the twin."""
        self.assertNotIn((take, flat, "rotation"), before)
        ((source, _output, values),) = after[(take, flat, "rotation")]
        ((twin_source, _twin_output, twin_values),) = after[(take, twin, "rotation")]
        self.assertEqual(source, twin_source)
        self.assertSameRotations(values, twin_values)

    def test_each_take_gets_the_channel_the_converter_dropped_and_no_other(self):
        """A node flat at rest loses every rotation (its rest too); one flat at
        every key of take A only that take's. Take B starts off the bake grid
        (frame 10.5), where the converter's t = 0 is the frame it truncates to.
        Every channel the converter wrote stays as it wrote it, and a second
        run changes nothing."""
        from test_fbx_media import build_curve_fbx

        def ramp(first, step):
            return [(first + f, step * f) for f in range(8)]

        takes = {
            "A": {
                ("round", "Lcl Rotation", "d|X"): ramp(0, 18.0),
                ("flat", "Lcl Rotation", "d|X"): ramp(0, 18.0),
                ("squash_round", "Lcl Rotation", "d|Z"): ramp(0, -11.0),
                ("squash", "Lcl Rotation", "d|Z"): ramp(0, -11.0),
                ("squash", "Lcl Scaling", "d|X"): [(f, 0.0) for f in range(8)],
                ("squash", "Lcl Scaling", "d|Y"): [(f, 1.0) for f in range(8)],
                ("squash", "Lcl Scaling", "d|Z"): [(f, 1.0) for f in range(8)],
            },
            "B": {
                ("round", "Lcl Rotation", "d|X"): ramp(10.5, 5.0),
                ("flat", "Lcl Rotation", "d|X"): ramp(10.5, 5.0),
                ("squash", "Lcl Rotation", "d|Z"): ramp(10.5, 7.0),
            },
        }
        tilt = {"Lcl Rotation": (12.0, 0.0, 0.0)}
        fbx = build_curve_fbx(
            os.path.join(self.tmp, "takes.fbx"),
            takes,
            poses={
                "round": tilt,
                "flat": {**tilt, "Lcl Scaling": (1.0, 0.0, 1.0)},
                "squash_round": {"Lcl Scaling": (1.0, 1.0, 1.0)},
                "squash": {"Lcl Scaling": (1.0, 1.0, 1.0)},
            },
        )
        glb = self._converted(fbx)
        before = self._channels(glb)
        self.assertIsNone(self._rest(glb, "flat"))  # the defect, as shipped

        report = MeshConvert.restore_glb_rotations(glb, fbx)

        self.assertEqual(report, {"nodes": 1, "channels": 3})
        after = self._channels(glb)
        for key, written in before.items():
            self.assertEqual(after[key], written, key)  # never rewritten
        for key in set(after) - set(before):
            self.assertEqual(len(after[key]), 1, key)  # never doubled
        self.assertTwins(before, after, "A", "flat", "round")
        self.assertTwins(before, after, "B", "flat", "round")
        self.assertTwins(before, after, "A", "squash", "squash_round")
        # Not flat in B: the converter's own channel, the only one.
        self.assertIn(("B", "squash", "rotation"), before)
        self.assertSameRotations(
            [self._rest(glb, "flat")], [self._rest(glb, "round")]
        )
        with open(glb, "rb") as fh:
            restored = fh.read()
        self.assertEqual(
            MeshConvert.restore_glb_rotations(glb, fbx), {"nodes": 0, "channels": 0}
        )
        with open(glb, "rb") as fh:
            self.assertEqual(fh.read(), restored)

    def test_order_pre_post_rotation_and_the_axis_turn_compose_as_converted(self):
        """In a Z-up file: a root child with a ZXY order and pre/post rotation
        (both in XYZ order), its child in YZX, and a model whose order and
        pre/post rotation RotationActive switches off."""
        from test_fbx_media import build_curve_fbx

        posed = {
            "RotationOrder": 4,
            "PreRotation": (10.0, 20.0, 30.0),
            "PostRotation": (5.0, -15.0, 25.0),
            "Lcl Rotation": (30.0, 5.0, 0.0),
        }
        poses = {
            "root": {**posed, "RotationActive": 1},
            "kid": {"RotationOrder": 2, "RotationActive": 1},
            "idle": {**posed, "RotationActive": 0},
        }
        flat = {"Lcl Scaling": (1.0, 0.0, 1.0)}
        keys = {
            axis: [(f, step * f) for f in range(6)]
            for axis, step in (("d|X", 18.0), ("d|Y", -7.0), ("d|Z", 11.0))
        }
        takes = {
            "Take": {
                (f"{model}{suffix}", "Lcl Rotation", axis): curve
                for model in poses
                for suffix in ("_round", "_flat")
                for axis, curve in keys.items()
            }
        }
        fbx = build_curve_fbx(
            os.path.join(self.tmp, "posed.fbx"),
            takes,
            poses={
                f"{model}{suffix}": {**pose, **(flat if suffix == "_flat" else {})}
                for model, pose in poses.items()
                for suffix in ("_round", "_flat")
            },
            parents={"kid_round": "root_round", "kid_flat": "root_round"},
            axes={"UpAxis": 2, "UpAxisSign": 1, "FrontAxis": 1, "FrontAxisSign": -1},
        )
        glb = self._converted(fbx)
        before = self._channels(glb)

        report = MeshConvert.restore_glb_rotations(glb, fbx)

        self.assertEqual(report, {"nodes": 2, "channels": 3})
        after = self._channels(glb)
        for model in poses:
            with self.subTest(model=model):
                self.assertTwins(
                    before, after, "Take", f"{model}_flat", f"{model}_round"
                )
                twin = self._rest(glb, f"{model}_round")
                rest = self._rest(glb, f"{model}_flat")
                if twin is None or abs(twin[3]) == 1.0:
                    self.assertIsNone(rest)  # nothing to restore: no tilt at rest
                else:
                    self.assertSameRotations([rest], [twin])

    def test_a_file_with_no_flat_node_lacking_a_rotation_never_reads_its_fbx(self):
        """A flat node that states its rotation lost nothing, and a curve
        proxy -- FBX-only transport carrying a ramp on ``scale.x`` -- is flat
        wherever the ramp reads 0: neither costs the FBX's curves a read."""
        from test_gltf2fbx import GltfBuilder

        b = GltfBuilder()
        marker = {MeshConvert.CURVE_PROXY_MARKER: {"type": "Bool", "value": True}}
        proxy = {"fromFBX": {"userProperties": marker}}
        b.gltf.update(
            scenes=[{"nodes": [0, 1, 2]}],
            nodes=[
                {"name": "round"},
                {
                    "name": "tilted",
                    "scale": [1.0, 0.0, 1.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                },
                {"name": "fade__opacity", "scale": [0.0, 1.0, 1.0], "extras": proxy},
            ],
        )
        glb = b.write_glb(os.path.join(self.tmp, "round.glb"))
        with open(glb, "rb") as fh:
            before = fh.read()
        missing = os.path.join(self.tmp, "no_such.fbx")
        self.assertEqual(
            MeshConvert.restore_glb_rotations(glb, missing), {"nodes": 0, "channels": 0}
        )
        with open(glb, "rb") as fh:
            self.assertEqual(fh.read(), before)


if __name__ == "__main__":
    unittest.main()
