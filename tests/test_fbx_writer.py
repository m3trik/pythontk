# !/usr/bin/python
# coding=utf-8
"""FbxWriter -- binary FBX 7.4 encoding, read back through FbxFile.

The writer is only as good as a reader's view of it, so every assertion goes
through :class:`FbxFile` (the ecosystem's independent binary-FBX parser):
typed scalars, raw and compressed arrays, nesting, the null-record rules and
the fixed footer. Added: 2026-10-09
"""

import os
import struct
import unittest
from unittest import mock

from pythontk.file_utils.mesh_convert._fbx_writer import (
    FBX_VERSION,
    FbxNode,
    FbxWriter,
)
from pythontk.file_utils.mesh_convert.fbx_file import FbxFile
from pythontk.file_utils.temp_artifacts import TempArtifacts


class FbxWriterTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = TempArtifacts("test_fbx_writer", policy="scoped")

    def tearDown(self):
        self.temp.cleanup()

    def _round_trip(self, records):
        path = FbxWriter.write(self.temp.path(extension=".fbx"), records)
        return path, FbxFile.load(path, decode_arrays=True)

    @staticmethod
    def _child(record, name):
        return next(c for c in record["children"] if c["name"] == name)

    def test_scalars_keep_their_types(self):
        node = FbxNode(
            "Thing",
            FbxWriter.i64(7),
            "Name\x00\x01Model",
            True,
            42,
            2**40,
            1.5,
            FbxWriter.f32(0.25),
            FbxWriter.i16(-3),
            FbxWriter.raw(b"\x00\x01\x02"),
        )
        _, fbx = self._round_trip([node])
        props = fbx.roots[0]["props"]
        self.assertEqual(props[0], 7)
        self.assertEqual(props[1], b"Name\x00\x01Model")
        self.assertTrue(props[2])
        self.assertEqual(props[3:5], [42, 2**40])
        self.assertAlmostEqual(props[5], 1.5)
        self.assertAlmostEqual(props[6], 0.25)
        self.assertEqual(props[7], -3)
        self.assertEqual(bytes(props[8]), b"\x00\x01\x02")

    def test_arrays_round_trip_raw_and_compressed(self):
        big = [float(i) * 0.5 for i in range(1000)]  # past the compression floor
        node = FbxNode("Geometry")
        node.add("Vertices", FbxWriter.array("d", big))
        node.add("Index", FbxWriter.array("i", [0, 1, -3]))
        node.add("Times", FbxWriter.array("l", [0, 46186158000]))
        node.add("Values", FbxWriter.array("f", [0.5, 1.5]))
        path, fbx = self._round_trip([node])
        geo = fbx.roots[0]
        self.assertEqual(list(self._child(geo, "Vertices")["props"][0]), big)
        self.assertEqual(list(self._child(geo, "Index")["props"][0]), [0, 1, -3])
        self.assertEqual(list(self._child(geo, "Times")["props"][0]), [0, 46186158000])
        self.assertEqual(list(self._child(geo, "Values")["props"][0]), [0.5, 1.5])
        # Compressed: 8000 bytes of doubles must not be stored verbatim.
        self.assertLess(os.path.getsize(path), 8000)

    def test_nesting_and_properties70_rows(self):
        model = FbxNode("Model", FbxWriter.i64(1), "Box\x00\x01Model", "Mesh")
        model.add("Properties70").p(
            "Lcl Translation", "Lcl Translation", "", "A", 1.0, 2.0, 3.0
        )
        model.add("Shading", True)
        _, fbx = self._round_trip([FbxNode("Objects"), model])
        row = self._child(self._child(fbx.roots[1], "Properties70"), "P")
        self.assertEqual(row["props"][0], b"Lcl Translation")
        self.assertEqual(row["props"][4:], [1.0, 2.0, 3.0])

    def test_header_records_and_footer(self):
        records = FbxWriter.header_records("pythontk test") + [FbxNode("Objects")]
        path, fbx = self._round_trip(records)
        self.assertEqual(fbx.version, FBX_VERSION)
        self.assertEqual(
            [r["name"] for r in fbx.roots[:4]],
            ["FBXHeaderExtension", "FileId", "CreationTime", "Creator"],
        )
        with open(path, "rb") as fh:
            data = fh.read()
        self.assertEqual(data[-16:], bytes.fromhex("f85a8c6adef5d97eece90ce3758f290b"))
        self.assertEqual(struct.unpack("<I", data[-140:-136])[0], FBX_VERSION)

    def test_an_animation_stack_closes_with_a_null_record(self):
        """A stack or layer with no children still ends in the 13-byte null
        record the readers this targets expect of them."""
        stack = FbxNode("AnimationStack", FbxWriter.i64(5), "Take\x00\x01AnimStack", "")
        data = FbxWriter.encode([stack, FbxNode("Other", 1)])
        end = struct.unpack_from("<I", data, 27)[0]
        self.assertEqual(data[end - 13 : end], b"\x00" * 13)

    def test_untyped_values_are_refused(self):
        with self.assertRaises(TypeError):
            FbxWriter.encode([FbxNode("Bad", [1, 2])])
        with self.assertRaises(ValueError):
            FbxWriter.array("x", [1])

    def test_write_leaves_no_partial_file(self):
        path = self.temp.path(extension=".fbx")
        FbxWriter.write(path, [FbxNode("Objects")])
        self.assertTrue(os.path.isfile(path))
        self.assertFalse(os.path.exists(path + ".part"))

    def test_a_failed_write_keeps_the_previous_file_and_strands_nothing(self):
        """A promotion that fails (a disk full, a lock that never lifts)
        leaves the previous FBX whole and no ``.part`` beside it."""
        path = self.temp.path(extension=".fbx")
        FbxWriter.write(path, [FbxNode("Objects")])
        with open(path, "rb") as fh:
            before = fh.read()
        folder = os.path.dirname(path)
        listing = sorted(os.listdir(folder))
        with mock.patch("os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                FbxWriter.write(path, [FbxNode("Objects"), FbxNode("More", 1)])
        self.assertEqual(sorted(os.listdir(folder)), listing)
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
