# !/usr/bin/python
# coding=utf-8
"""Tests for ``ptk.FolderDeliverer`` and ``ptk.FileDeliveryBridge``: a file
delivered by copying it into a folder another application watches."""

import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk.core_utils.handoff import folder_delivery  # noqa: E402
from pythontk.core_utils.handoff.app_handoff import (  # noqa: E402
    HandoffRequest,
    Payload,
)
from pythontk.core_utils.handoff.folder_delivery import (  # noqa: E402
    FileDeliveryBridge,
    FolderDeliverer,
)
from pythontk.core_utils.operation_catalog.op_context import OpContext  # noqa: E402
from pythontk.file_utils.mesh_convert.fbx_file import FBX_MAGIC  # noqa: E402
from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia  # noqa: E402
from test_fbx_media import build_carrier_fbx  # noqa: E402


class _Assets(FolderDeliverer):
    """A target's deliverer as an app shapes it: its own folder and keys."""

    DEFAULT_SUBDIR = "Imported"
    SUBDIR_PARAM = "ASSETS_SUBDIR"
    NAME_PARAM = "ASSET_NAME"

    def target(self, bridge):
        folder = os.path.join(bridge.target_folder or "", "Assets")
        return folder if os.path.isdir(folder) else None


class FolderDeliveryTest(unittest.TestCase):
    """Added: 2026-11-04"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.model = os.path.join(self.root, "robot.obj")
        with open(self.model, "w", encoding="utf-8") as f:
            f.write("v 0 0 0\n")
        self.target = os.path.join(self.root, "target")
        os.makedirs(self.target)

    def _bridge(self, deliverer=None, folder=None):
        return FileDeliveryBridge(target_folder=folder or self.target, deliverer=deliverer)

    def test_a_file_lands_under_its_subfolder_by_its_own_name(self):
        result = self._bridge().send(self.model, params={"SUBDIR": "Models"})
        self.assertEqual(result["file"], os.path.join(self.target, "Models", "robot.obj"))
        self.assertTrue(os.path.isfile(result["file"]))
        self.assertEqual(result["subdir"], "Models")

    def test_a_name_and_a_subfolder_never_leave_the_folder(self):
        result = self._bridge().send(
            self.model, params={"SUBDIR": "../..\\C:evil/./x", "NAME": "a/b:c"}
        )
        self.assertTrue(os.path.abspath(result["file"]).startswith(os.path.abspath(self.target)))
        self.assertEqual(os.path.basename(result["file"]), "a_b_c.obj")

    def test_a_missing_folder_is_refused_before_anything_is_made(self):
        bridge = self._bridge(folder=os.path.join(self.root, "nowhere"))
        self.assertIsNone(bridge.send(self.model))
        self.assertFalse(os.path.exists(os.path.join(self.root, "nowhere")))

    def test_a_dry_run_lists_the_copy_and_writes_nothing(self):
        ctx = OpContext(dry_run=True)
        result = self._bridge().send(self.model, params={"SUBDIR": "Models"}, ctx=ctx)
        self.assertFalse(os.path.exists(result["file"]))
        self.assertTrue(any(c.target == result["file"] for c in ctx.changes))

    def test_an_app_shapes_the_folder_its_default_and_its_keys(self):
        os.makedirs(os.path.join(self.target, "Assets"))
        result = self._bridge(_Assets()).send(self.model, params={"ASSET_NAME": "Bot"})
        self.assertEqual(result["file"], os.path.join(self.target, "Assets", "Imported", "Bot.obj"))

    def test_an_fbx_takes_its_files_through_the_embed_it_was_given(self):
        """The embedding is injected: none copies the FBX as it is; one writes
        the staged copy that lands. Added: 2026-11-04"""
        fbx = os.path.join(self.root, "prop.fbx")
        with open(fbx, "wb") as f:
            f.write(b"plain")
        plain = self._bridge().send(fbx)
        with open(plain["file"], "rb") as f:
            self.assertEqual(f.read(), b"plain")

        calls = []

        def embed(source, dest, search_dirs=(), logger=None):
            calls.append(source)
            with open(dest, "wb") as f:
                f.write(b"embedded")
            return {"embedded": ["lightmap.exr"]}

        result = self._bridge(FolderDeliverer(embed=embed)).send(fbx, params={"SUBDIR": "x"})
        with open(result["file"], "rb") as f:
            self.assertEqual(f.read(), b"embedded")
        self.assertEqual(calls, [fbx])

    def test_the_delivery_modes_are_one_copy(self):
        self.assertEqual(self._bridge().list_template_modes(), list(FolderDeliverer.DELIVERY_MODES))


class _Logger:
    """Captures a bridge's messages by level."""

    def __init__(self):
        self.errors, self.warnings, self.infos = [], [], []

    def error(self, m, *args):
        self.errors.append(m % args if args else m)

    def warning(self, m, *args):
        self.warnings.append(m % args if args else m)

    def info(self, m, *args):
        self.infos.append(m % args if args else m)

    def debug(self, m, *args):
        pass


class _Bridge:
    """The contract the deliverer reads off a bridge: its folder and logger."""

    def __init__(self, target_folder):
        self.target_folder = target_folder
        self.logger = _Logger()


class _HostBridge(_Bridge):
    """A DCC bridge: it says where its host keeps the maps the records name
    (the optional ``lightmap_search_dirs`` hook every DCC export mixin answers)."""

    def __init__(self, target_folder, search_dirs):
        super().__init__(target_folder)
        self._dirs = search_dirs

    def lightmap_search_dirs(self):
        return list(self._dirs)


class AtomicCopyTest(unittest.TestCase):
    """Ported from the project-folder deliverer's own tests. Added: 2026-11-04"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.target = os.path.join(self.tmp.name, "target")
        os.makedirs(self.target)
        self.model = os.path.join(self.tmp.name, "robot.fbx")
        with open(self.model, "wb") as fh:
            fh.write(b"NEW")

    def test_a_copy_cut_short_never_replaces_the_file_the_target_reads(self):
        """The copy lands whole or not at all (``FileUtils.atomic_write``): a
        copy cut short (a full disk, a sync client holding the file) left the
        target a truncated file under the name it reads."""
        dest = os.path.join(self.target, "Models", "Robot.fbx")
        os.makedirs(os.path.dirname(dest))
        with open(dest, "wb") as fh:
            fh.write(b"OLD")

        def cut_short(src, dst, *args, **kwargs):
            with open(dst, "wb") as fh:
                fh.write(b"HALF")
            raise OSError(28, "No space left on device")

        bridge = _Bridge(self.target)
        with mock.patch.object(folder_delivery.shutil, "copy2", side_effect=cut_short):
            result = FolderDeliverer().deliver(
                bridge,
                Payload(primary=self.model, extras={"default_asset_name": "Robot"}),
                HandoffRequest(params={"SUBDIR": "Models"}),
            )
        self.assertIsNone(result)
        self.assertTrue(bridge.logger.errors)
        with open(dest, "rb") as fh:
            self.assertEqual(fh.read(), b"OLD")
        self.assertEqual(os.listdir(os.path.dirname(dest)), ["Robot.fbx"])


class SelfContainedDeliveryTest(unittest.TestCase):
    """The real embed (``FbxMedia.embed_dependencies``) injected: the FBX lands
    carrying every file its records name, so nothing is copied beside it.
    Ported from the project-folder deliverer's own tests. Added: 2026-11-04"""

    RECORDS = {
        "lightmaps": {
            "version": 2,
            "unit_scale": 0.01,
            "objects": [{"name": "Crate", "map": "Crate_Lightmap.exr", "scaleOffset": [1.0, 1.0, 0.0, 0.0], "hierarchy": ["Crate"]}],
            "probe": {"map": "Room_Probe.exr", "position": [0, 0, 0], "box": None},
        },
        "shadows": {
            "version": 3,
            "planes": [
                {
                    "name": "Crate_shadow",
                    "silhouette": {"texture": "Crate_shadow.png"},
                    "horizon": {"texture": "Crate_horizon.png"},
                }
            ],
        },
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        self.target = os.path.join(root, "target")
        os.makedirs(self.target)
        self.maps = os.path.join(root, "sourceimages")
        os.makedirs(self.maps)
        self.files = {}
        for name in ("Crate_Lightmap.exr", "Room_Probe.exr", "Crate_shadow.png", "Crate_horizon.png"):
            self.files[name] = name.encode() * 8
            with open(os.path.join(self.maps, name), "wb") as fh:
                fh.write(self.files[name])
        self.src_dir = os.path.join(root, "export")
        os.makedirs(self.src_dir)
        self.fbx = build_carrier_fbx(os.path.join(self.src_dir, "export.fbx"), self.RECORDS)
        with open(self.fbx, "rb") as fh:
            self.source = fh.read()
        self.deliverer = FolderDeliverer(embed=FbxMedia.embed_dependencies)

    def _deliver(self, bridge, path=None, deliverer=None):
        return (deliverer or self.deliverer).deliver(
            bridge,
            Payload(primary=path or self.fbx, extras={"default_asset_name": "Crate"}),
            HandoffRequest(),
        )

    @staticmethod
    def _read(path):
        with open(path, "rb") as fh:
            return fh.read()

    def test_the_fbx_carries_every_file_its_records_name(self):
        """Each name resolves where the bridge's host keeps it (its
        ``lightmap_search_dirs`` hook); the source is only read."""
        bridge = _HostBridge(self.target, [self.maps])
        result = self._deliver(bridge)
        self.assertEqual(
            {row["name"]: row["bytes"] for row in FbxMedia.embedded(result["file"])},
            {name: len(data) for name, data in self.files.items()},
        )
        self.assertEqual(os.listdir(self.target), ["Crate.fbx"])
        self.assertEqual(self._read(self.fbx), self.source)
        self.assertFalse(bridge.logger.warnings, bridge.logger.warnings)

    def test_a_bridge_without_the_hook_looks_beside_the_source(self):
        for name, data in self.files.items():
            with open(os.path.join(self.src_dir, name), "wb") as fh:
                fh.write(data)
        result = self._deliver(_Bridge(self.target))
        self.assertEqual(
            sorted(row["name"] for row in FbxMedia.embedded(result["file"])),
            sorted(self.files),
        )

    def test_a_hook_that_raises_is_warned_and_the_source_folder_still_answers(self):
        for name, data in self.files.items():
            with open(os.path.join(self.src_dir, name), "wb") as fh:
                fh.write(data)
        bridge = _Bridge(self.target)
        bridge.lightmap_search_dirs = mock.Mock(side_effect=RuntimeError("no scene"))
        result = self._deliver(bridge)
        self.assertIn("no scene", "\n".join(bridge.logger.warnings))
        self.assertEqual(len(FbxMedia.embedded(result["file"])), len(self.files))

    def test_an_embed_pass_that_fails_ships_the_fbx_as_exported(self):
        """The pass is an addition to the delivery, never a reason to fail it."""
        failing = FolderDeliverer(embed=mock.Mock(side_effect=struct.error("cut short")))
        bridge = _HostBridge(self.target, [self.maps])
        result = self._deliver(bridge, deliverer=failing)
        self.assertEqual(self._read(result["file"]), self.source)
        self.assertIn("cut short", "\n".join(bridge.logger.warnings))

    def test_a_truncated_fbx_ships_as_it_is(self):
        """The magic but not the body: nothing to embed into, the copy stands."""
        cut = FBX_MAGIC + struct.pack("<I", 7700) + b"\x10" * 20
        with open(self.fbx, "wb") as fh:
            fh.write(cut)
        bridge = _HostBridge(self.target, [self.maps])
        result = self._deliver(bridge)
        self.assertEqual(self._read(result["file"]), cut)
        self.assertTrue(bridge.logger.warnings)

    def test_the_embed_decides_by_content_not_by_extension(self):
        """The gate is the embed's (the FBX code's): the deliverer called it
        only for a ``.fbx`` name, so an ASCII ``.fbx`` was warned "not a binary
        FBX" and a binary FBX under another name went in bare."""
        ascii_fbx = os.path.join(self.src_dir, "ascii.fbx")
        with open(ascii_fbx, "wb") as fh:
            fh.write(b"; FBX 7.7.0 project file (ASCII)\n" * 20)
        bridge = _HostBridge(self.target, [self.maps])
        result = self._deliver(bridge, ascii_fbx)
        self.assertEqual(self._read(result["file"]), self._read(ascii_fbx))
        self.assertFalse(bridge.logger.warnings, bridge.logger.warnings)

        renamed = os.path.join(self.src_dir, "export.bin")
        with open(renamed, "wb") as fh:
            fh.write(self.source)
        bridge = _HostBridge(self.target, [self.maps])
        result = self._deliver(bridge, renamed)
        self.assertEqual(os.path.basename(result["file"]), "Crate.bin")
        self.assertEqual(len(FbxMedia.embedded(result["file"])), len(self.files))
        self.assertFalse(bridge.logger.warnings, bridge.logger.warnings)

    def test_a_file_that_is_no_fbx_is_copied_as_it_is_quietly(self):
        obj = os.path.join(self.src_dir, "scan.obj")
        with open(obj, "w", encoding="utf-8") as fh:
            fh.write("v 0 0 0\n")
        bridge = _HostBridge(self.target, [self.maps])
        result = self._deliver(bridge, obj)
        self.assertEqual(self._read(result["file"]), self._read(obj))
        self.assertFalse(bridge.logger.warnings, bridge.logger.warnings)


if __name__ == "__main__":
    unittest.main()
