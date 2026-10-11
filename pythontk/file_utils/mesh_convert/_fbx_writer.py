# !/usr/bin/python
# coding=utf-8
"""Zero-dependency binary-FBX writer: the encoding half of :class:`FbxFile`.

:class:`FbxFile` reads Kaydara's binary container; this writes one. A document
is a tree of :class:`FbxNode` records -- a name, typed properties, children --
serialized in the 7.4 layout (32-bit record offsets), which every importer the
ecosystem feeds reads: Maya's plug-in, Unity, Blender, FBX2glTF.

Typing is explicit where Python's is ambiguous. A bare ``bool`` is ``C``, an
``int`` is ``I`` (``L`` past 32 bits), a ``float`` ``D``, a ``str`` ``S`` and
``bytes`` ``R``; everything else is spelled through a constructor
(:meth:`FbxWriter.i64`, :meth:`FbxWriter.array` ...). Object ids are 64-bit by
format, so they are always spelled :meth:`FbxWriter.i64`, even when small.

The fixed header and footer bytes (``FileId``, ``CreationTime``, the footer id
and magic) are copied from a Blender 5.1 export (measured 2026-10-09), a set
Maya's importer reads without complaint.
"""

import struct
import sys
import zlib
from array import array
from pathlib import Path
from typing import Any, List, Sequence

from pythontk.file_utils.mesh_convert.fbx_file import FBX_MAGIC

#: The record layout written: 7.4, 32-bit offsets (7.5 widened them to 64).
FBX_VERSION = 7400

#: FBX time units per second (``FbxTime``'s tick rate).
FBX_TICKS_PER_SECOND = 46186158000

#: The 13-byte null record that closes a nested list (and the top level).
_SENTINEL = b"\x00" * 13
_FILE_ID = bytes.fromhex("28b32aebb624ccc2bfc8b02aa92bfcf1")
_CREATION_TIME = "1970-01-01 10:00:00:000"
_FOOT_ID = bytes.fromhex("fabcab09d0c8d466b176fb831cf7267e")
_FOOT_MAGIC = bytes.fromhex("f85a8c6adef5d97eece90ce3758f290b")
#: Records that close with a null record even when they hold no children --
#: the readers this targets expect it of an animation stack and layer.
_ALWAYS_SENTINEL = frozenset({"AnimationStack", "AnimationLayer"})
#: Arrays at least this many bytes long are zlib-compressed (encoding 1).
_COMPRESS_MIN = 128

_SCALAR_FORMATS = {"Y": "<h", "C": "<?", "I": "<i", "F": "<f", "D": "<d", "L": "<q"}
#: FBX array type -> :mod:`array` typecode of the same width.
_ARRAY_TYPECODES = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "b"}


class _Typed(object):
    """One property value with its FBX type code fixed."""

    __slots__ = ("code", "value")

    def __init__(self, code: str, value: Any):
        self.code = code
        self.value = value


class FbxNode(object):
    """One record: ``name``, its properties, and its child records.

    Example:
        >>> geo = FbxNode("Geometry", FbxWriter.i64(1), "Cube\\x00\\x01Geometry")
        >>> geo.add("Vertices", FbxWriter.array("d", [0.0, 0.0, 0.0]))
    """

    __slots__ = ("name", "props", "children")

    def __init__(self, name: str, *props: Any):
        self.name = name
        self.props = list(props)
        self.children: List["FbxNode"] = []

    def add(self, name: str, *props: Any) -> "FbxNode":
        """Append a child record and return it."""
        node = FbxNode(name, *props)
        self.children.append(node)
        return node

    def p(self, name: str, kind: str, label: str, flags: str, *values: Any) -> None:
        """Append one ``P`` entry -- the shape of every ``Properties70`` row."""
        self.add("P", name, kind, label, flags, *values)


class FbxWriter(object):
    """Serialize :class:`FbxNode` trees as binary FBX 7.4.

    Example:
        >>> root = FbxNode("Objects")
        >>> FbxWriter.write("out.fbx", FbxWriter.header_records("me") + [root])
    """

    VERSION = FBX_VERSION
    TICKS_PER_SECOND = FBX_TICKS_PER_SECOND

    # ------------------------------------------------------------ typed values
    @staticmethod
    def i64(value: int) -> _Typed:
        """*value* as ``L`` -- object ids, ``KTime``."""
        return _Typed("L", int(value))

    @staticmethod
    def i32(value: int) -> _Typed:
        """*value* as ``I``."""
        return _Typed("I", int(value))

    @staticmethod
    def i16(value: int) -> _Typed:
        """*value* as ``Y``."""
        return _Typed("Y", int(value))

    @staticmethod
    def f32(value: float) -> _Typed:
        """*value* as ``F``."""
        return _Typed("F", float(value))

    @staticmethod
    def raw(value: bytes) -> _Typed:
        """*value* as ``R`` (raw bytes)."""
        return _Typed("R", bytes(value))

    @staticmethod
    def array(code: str, values: Sequence[Any]) -> _Typed:
        """*values* as an FBX array of type *code* (``f d l i b``).

        Raises:
            ValueError: *code* is not an FBX array type.
        """
        if code not in _ARRAY_TYPECODES:
            raise ValueError(f"Not an FBX array type: {code!r}")
        return _Typed(code, values)

    # ------------------------------------------------------------ document
    @classmethod
    def header_records(cls, creator: str) -> List[FbxNode]:
        """The records every FBX file opens with, ahead of ``GlobalSettings``."""
        ext = FbxNode("FBXHeaderExtension")
        ext.add("FBXHeaderVersion", 1003)
        ext.add("FBXVersion", cls.VERSION)
        ext.add("EncryptionType", 0)
        stamp = ext.add("CreationTimeStamp")
        for key, value in (
            ("Version", 1000),
            ("Year", 1970),
            ("Month", 1),
            ("Day", 1),
            ("Hour", 10),
            ("Minute", 0),
            ("Second", 0),
            ("Millisecond", 0),
        ):
            stamp.add(key, value)
        ext.add("Creator", creator)
        return [
            ext,
            FbxNode("FileId", cls.raw(_FILE_ID)),
            FbxNode("CreationTime", _CREATION_TIME),
            FbxNode("Creator", creator),
        ]

    @classmethod
    def encode(cls, records: Sequence[FbxNode]) -> bytes:
        """The complete file for top-level *records*.

        Raises:
            ValueError: The document outgrows the 7.4 layout's 32-bit offsets.
        """
        buf = bytearray(FBX_MAGIC)
        buf += struct.pack("<I", cls.VERSION)
        for i, record in enumerate(records):
            cls._write_record(buf, record, i == len(records) - 1)
        buf += _SENTINEL
        buf += _FOOT_ID
        buf += b"\x00" * 4
        pad = (16 - len(buf) % 16) or 16
        buf += b"\x00" * pad
        buf += struct.pack("<I", cls.VERSION)
        buf += b"\x00" * 120
        buf += _FOOT_MAGIC
        return bytes(buf)

    @classmethod
    def write(cls, path: str, records: Sequence[FbxNode]) -> str:
        """Encode *records* to *path* atomically.

        A reader sees the previous file or the complete new one, never a
        half-written FBX that would import as a truncated scene; a failed
        write leaves no partial file behind (:meth:`FileUtils.atomic_write`).

        Parameters:
            path: The FBX to produce or replace; its directory must exist.
            records: The top-level records, header first.

        Returns:
            *path*.

        Raises:
            ValueError: The document outgrows the 7.4 layout's 32-bit offsets.
            OSError: The file could not be written or promoted -- *path*
                unchanged.
        """
        from pythontk.file_utils._file_utils import FileUtils

        data = cls.encode(records)
        return FileUtils.atomic_write(path, lambda part: Path(part).write_bytes(data))

    # ------------------------------------------------------------ encoding
    @classmethod
    def _write_record(cls, buf: bytearray, node: FbxNode, is_last: bool) -> None:
        start = len(buf)
        buf += b"\x00" * 12  # end offset, property count, property bytes
        name = node.name.encode("ascii")
        buf.append(len(name))
        buf += name
        props_start = len(buf)
        for value in node.props:
            cls._write_property(buf, cls._typed(value))
        props_len = len(buf) - props_start
        if node.children:
            for i, child in enumerate(node.children):
                cls._write_record(buf, child, i == len(node.children) - 1)
            buf += _SENTINEL
        elif (not node.props and not is_last) or node.name in _ALWAYS_SENTINEL:
            buf += _SENTINEL
        if len(buf) > 0xFFFFFFFF:
            raise ValueError("FBX document exceeds the 7.4 layout's 4 GB limit.")
        struct.pack_into("<III", buf, start, len(buf), len(node.props), props_len)

    @staticmethod
    def _typed(value: Any) -> _Typed:
        if isinstance(value, _Typed):
            return value
        if isinstance(value, bool):
            return _Typed("C", value)
        if isinstance(value, int):
            return _Typed("I" if -(2**31) <= value < 2**31 else "L", value)
        if isinstance(value, float):
            return _Typed("D", value)
        if isinstance(value, str):
            return _Typed("S", value)
        if isinstance(value, (bytes, bytearray)):
            return _Typed("R", bytes(value))
        raise TypeError(f"No FBX type for {type(value).__name__}: {value!r}")

    @staticmethod
    def _write_property(buf: bytearray, prop: _Typed) -> None:
        code, value = prop.code, prop.value
        buf += code.encode("ascii")
        if code in _SCALAR_FORMATS:
            buf += struct.pack(_SCALAR_FORMATS[code], value)
        elif code in ("S", "R"):
            data = value.encode("utf-8") if isinstance(value, str) else value
            buf += struct.pack("<I", len(data))
            buf += data
        else:
            packed = array(_ARRAY_TYPECODES[code], value)
            if sys.byteorder != "little":
                packed.byteswap()
            data = packed.tobytes()
            encoding = 0
            if len(data) >= _COMPRESS_MIN:
                data, encoding = zlib.compress(data), 1
            buf += struct.pack("<III", len(packed), encoding, len(data))
            buf += data


__all__ = ["FbxNode", "FbxWriter", "FBX_VERSION", "FBX_TICKS_PER_SECOND"]
