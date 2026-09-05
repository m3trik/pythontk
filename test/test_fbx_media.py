# !/usr/bin/python
# coding=utf-8
"""Unit tests for :class:`pythontk.FbxMedia` -- the binary-FBX embedded-media writer.

The fixture is a synthetic 7700 FBX built the way the SDK lays one out
(64-bit records, nested NULL sentinels, the 16-byte footer id, 16-byte
alignment padding, version echo, magic). The round-trip identity that pins
the writer here was also checked by hand against two Maya-written files
(a 21 MB rigged prop and a 366 MB production assembly): both re-serialised
byte for byte.
"""

import io
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk.file_utils.mesh_convert.fbx_file import FbxFile  # noqa: E402
from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia  # noqa: E402
from pythontk.file_utils.temp_artifacts import TempArtifacts  # noqa: E402

MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
FOOTER_ID = bytes(range(16))
FOOT_MAGIC = b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"


class Raw(bytes):
    """A property written with the ``R`` (raw binary) tag rather than ``S``."""


def _prop(value) -> bytes:
    if isinstance(value, Raw):
        return b"R" + struct.pack("<I", len(value)) + value
    if isinstance(value, bytes):
        return b"S" + struct.pack("<I", len(value)) + value
    if isinstance(value, int):
        return b"L" + struct.pack("<q", value)
    raise TypeError(type(value))


def _record(name: str, props, children, offset: int) -> bytes:
    prop_bytes = b"".join(_prop(p) for p in props)
    name_bytes = name.encode("ascii")
    header_len = 24 + 1 + len(name_bytes)
    body = b""
    child_offset = offset + header_len + len(prop_bytes)
    for child in children:
        chunk = _record(child[0], child[1], child[2], child_offset)
        body += chunk
        child_offset += len(chunk)
    if children:
        body += b"\x00" * 25
    end = offset + header_len + len(prop_bytes) + len(body)
    return (
        struct.pack("<QQQB", end, len(props), len(prop_bytes), len(name_bytes))
        + name_bytes
        + prop_bytes
        + body
    )


def png_bytes(size, mode="RGB") -> bytes:
    from PIL import Image

    image = Image.new(mode, size)
    # A gradient, so a resize changes the pixels rather than the header only.
    image.putdata(
        [
            (x % 256, y % 256, (x * y) % 256)[: len(mode)] if len(mode) > 1 else x % 256
            for y in range(size[1])
            for x in range(size[0])
        ]
    )
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def jpeg_bytes(size) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", size, (200, 40, 40)).save(out, format="JPEG", quality=95)
    return out.getvalue()


def build_fbx(path: str, media, models: int = 1, takes=None) -> str:
    """Write a binary FBX whose ``Video`` objects embed *media* ``{name: bytes}``.

    *models* ``Model`` records are written, and *takes* -- ``{name: (start,
    end)}`` in FBX ticks -- becomes the ``Takes`` section the SDK writes.
    """
    objects = [
        ("Model", [1001 + i, f"cube{i}\x00\x01Model".encode(), b"Mesh"], [])
        for i in range(models)
    ]
    connections = []
    for index, (name, payload) in enumerate(media.items()):
        video_id, texture_id = 5001 + index, 6001 + index
        objects.append(
            (
                "Video",
                [video_id, name.encode() + b"\x00\x01Video", b"Clip"],
                [
                    ("Type", [b"Clip"], []),
                    ("Filename", [("C:/tex/" + name).encode()], []),
                    ("RelativeFilename", [name.encode()], []),
                    ("Content", [Raw(payload)] if payload else [], []),
                ],
            )
        )
        objects.append(
            ("Texture", [texture_id, name.encode() + b"\x00\x01Texture", b""], [])
        )
        connections.append(("C", [b"OO", video_id, texture_id], []))
    roots = [("Objects", [], objects), ("Connections", [], connections)]
    if takes:
        roots.append(
            (
                "Takes",
                [],
                [("Current", [next(iter(takes)).encode()], [])]
                + [
                    (
                        "Take",
                        [name.encode()],
                        [
                            (
                                "FileName",
                                [name.replace(" ", "_").encode() + b".tak"],
                                [],
                            ),
                            ("LocalTime", [start, end], []),
                            ("ReferenceTime", [start, end], []),
                        ],
                    )
                    for name, (start, end) in takes.items()
                ],
            )
        )
    body = b""
    offset = len(MAGIC) + 4
    for name, props, children in roots:
        chunk = _record(name, props, children, offset)
        body += chunk
        offset += len(chunk)
    body += b"\x00" * 25
    body += FOOTER_ID
    at = offset + 25 + 16
    pad = ((at + 15) & ~15) - at or 16
    body += (
        b"\x00" * pad
        + b"\x00" * 4
        + struct.pack("<I", 7700)
        + b"\x00" * 120
        + FOOT_MAGIC
    )
    with open(path, "wb") as handle:
        handle.write(MAGIC + struct.pack("<I", 7700) + body)
    return path


class FbxMediaTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = TempArtifacts("test_fbx_media", policy="scoped")
        self.dir = self.temp.dir_path()
        self.big = png_bytes((512, 256))
        self.small = png_bytes((64, 64))
        self.src = build_fbx(
            os.path.join(self.dir, "asset.fbx"),
            {"wall_Base_color.png": self.big, "trim_Normal.png": self.small},
        )

    def tearDown(self):
        self.temp.cleanup()

    def _read(self, path):
        with open(path, "rb") as fh:
            return fh.read()

    def test_rewrite_is_byte_identical(self):
        out = os.path.join(self.dir, "copy.fbx")
        FbxMedia.rewrite(self.src, out)
        self.assertEqual(self._read(self.src), self._read(out))

    def test_embedded_lists_every_video_payload(self):
        rows = FbxMedia.embedded(self.src)
        self.assertEqual(
            [(r["name"], r["format"], r["size"], r["bytes"]) for r in rows],
            [
                ("wall_Base_color.png", "PNG", (512, 256), len(self.big)),
                ("trim_Normal.png", "PNG", (64, 64), len(self.small)),
            ],
        )

    def test_a_failed_write_strands_no_partial_file(self):
        """``downsize`` streams into a sibling ``.part`` and only consumes it
        on the success path's ``os.replace``.

        A raise inside ``_write`` -- disk full is the realistic case for a
        multi-hundred-MB payload -- left a partial file beside the target with
        nothing to sweep it, and allocated it raw besides. It is now a tracked
        ``TempArtifacts`` allocation, so a process that dies mid-write is
        swept by the next store on the same prefix rather than never.
        """
        from unittest import mock

        out = os.path.join(self.dir, "failed.fbx")
        with mock.patch.object(
            FbxMedia, "_write", side_effect=OSError("No space left on device")
        ):
            with self.assertRaises(OSError):
                FbxMedia.downsize(self.src, out, max_size=128)
        strays = [n for n in os.listdir(self.dir) if n.endswith(".part")]
        self.assertEqual(strays, [], f"partial file left behind: {strays}")
        self.assertFalse(os.path.exists(out), "a failed run must not leave a target")

    def test_downsize_resizes_only_images_over_the_ceiling(self):
        out = os.path.join(self.dir, "small.fbx")
        report = FbxMedia.downsize(self.src, out, max_size=128)
        self.assertEqual(report["images"], 2)
        self.assertEqual(report["resized"], 1)
        self.assertLess(report["after"], report["before"])
        rows = {r["name"]: r for r in FbxMedia.embedded(out)}
        self.assertEqual(rows["wall_Base_color.png"]["size"], (128, 64))
        self.assertEqual(rows["wall_Base_color.png"]["format"], "PNG")
        self.assertEqual(rows["trim_Normal.png"]["bytes"], len(self.small))
        # Everything but the payload survived: same objects, same wiring.
        before, after = FbxFile.load(self.src), FbxFile.load(out)
        self.assertEqual(before.objects_census(), after.objects_census())
        self.assertEqual(before.connections(), after.connections())
        self.assertLess(os.path.getsize(out), os.path.getsize(self.src))
        # The untouched source is still the file it was.
        self.assertEqual(FbxMedia.embedded(self.src)[0]["size"], (512, 256))

    def test_zero_ceiling_and_exempt_names_write_nothing(self):
        out = os.path.join(self.dir, "untouched.fbx")
        self.assertEqual(FbxMedia.downsize(self.src, out, max_size=0)["resized"], 0)
        self.assertFalse(os.path.exists(out))
        report = FbxMedia.downsize(
            self.src, out, max_size=128, exempt=["C:/elsewhere/wall_Base_color.png"]
        )
        self.assertEqual(report["resized"], 0)
        self.assertFalse(os.path.exists(out))

    def test_in_place_downsize_replaces_the_source(self):
        report = FbxMedia.downsize(self.src, max_size=100)
        self.assertEqual(report["resized"], 1)
        self.assertFalse(os.path.exists(self.src + ".part"))
        self.assertEqual(FbxMedia.embedded(self.src)[0]["size"], (100, 50))

    def test_jpeg_keeps_its_container_and_other_formats_are_left_alone(self):
        src = build_fbx(
            os.path.join(self.dir, "mixed.fbx"),
            {
                "photo.jpg": jpeg_bytes((400, 300)),
                "cube.dds": b"DDS \x7c\x00\x00\x00" + b"\x00" * 200,
            },
        )
        out = os.path.join(self.dir, "mixed_small.fbx")
        report = FbxMedia.downsize(src, out, max_size=200)
        self.assertEqual(report, {**report, "images": 2, "resized": 1})
        rows = {r["name"]: r for r in FbxMedia.embedded(out)}
        self.assertEqual(
            (rows["photo.jpg"]["format"], rows["photo.jpg"]["size"]),
            ("JPEG", (200, 150)),
        )
        self.assertEqual(
            (rows["cube.dds"]["format"], rows["cube.dds"]["bytes"]), (None, 208)
        )

    def test_an_empty_content_record_is_not_an_image(self):
        src = build_fbx(os.path.join(self.dir, "ref.fbx"), {"linked.png": b""})
        self.assertEqual(FbxMedia.embedded(src), [])
        self.assertEqual(FbxMedia.downsize(src, max_size=16)["images"], 0)

    def test_not_an_fbx_raises(self):
        junk = os.path.join(self.dir, "junk.fbx")
        with open(junk, "wb") as fh:
            fh.write(b"not an fbx" * 40)
        with self.assertRaises(ValueError):
            FbxMedia.embedded(junk)


if __name__ == "__main__":
    unittest.main()
