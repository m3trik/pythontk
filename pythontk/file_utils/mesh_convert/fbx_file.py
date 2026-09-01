# !/usr/bin/python
# coding=utf-8
"""Zero-dependency binary-FBX reader: header, node records, objects, takes.

The FBX half of deliverable verification. :meth:`MeshConvert.fbx_to_glb`
*converts* FBX through an external binary but nothing in the ecosystem could
*look inside* one — so a truncated write, a dropped take or a missing
animation stack surfaced only after a GLB conversion or a DCC import. This
reads Kaydara's binary container directly (both the 32-bit pre-7500 and the
64-bit 7500+ record layouts), decodes property records, and answers the
census questions a verifier asks. Array payloads are **skipped by default** —
a 200 MB production FBX parses in well under a second because the geometry is
never inflated.

Read-only by design: there is no writer here, so this can never be what
damages the file it inspects.
"""

import struct
import zlib
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Tuple

#: The 23-byte magic every binary FBX starts with.
FBX_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"

#: FBX splits an object's display name from its class with this token:
#: ``b"Shot_1\\x00\\x01AnimStack"`` -> name ``Shot_1``, class ``AnimStack``.
_NAME_CLASS_SEPARATOR = b"\x00\x01"


class _FbxFileInternal:
    """Record parsing for :class:`FbxFile`."""

    _SCALAR_PROPS = {
        b"Y": ("<h", 2),
        b"C": ("<b", 1),
        b"I": ("<i", 4),
        b"F": ("<f", 4),
        b"D": ("<d", 8),
        b"L": ("<q", 8),
    }
    _ARRAY_PROPS = {
        b"f": ("<f", 4),
        b"d": ("<d", 8),
        b"l": ("<q", 8),
        b"i": ("<i", 4),
        b"b": ("<b", 1),
        b"c": ("<B", 1),
    }

    @classmethod
    def _read_property(cls, f, decode_arrays: bool) -> Any:
        kind = f.read(1)
        scalar = cls._SCALAR_PROPS.get(kind)
        if scalar:
            fmt, size = scalar
            return struct.unpack(fmt, f.read(size))[0]
        array = cls._ARRAY_PROPS.get(kind)
        if array:
            fmt, size = array
            count, encoding, byte_len = struct.unpack("<III", f.read(12))
            payload = f.read(byte_len)
            if not decode_arrays:
                return ("ARRAY", kind.decode(), count)
            if encoding == 1:
                payload = zlib.decompress(payload)
            return list(struct.unpack(f"<{count}{fmt[-1]}", payload))
        if kind in (b"S", b"R"):
            length = struct.unpack("<I", f.read(4))[0]
            return f.read(length)
        raise ValueError(f"Unknown FBX property type {kind!r}")

    @classmethod
    def _read_record(
        cls, f, wide: bool, decode_arrays: bool
    ) -> Optional[Dict[str, Any]]:
        """One node record, or ``None`` at a NULL sentinel."""
        if wide:
            header = f.read(24)
            if len(header) < 24:
                return None
            end, prop_count, _prop_len = struct.unpack("<QQQ", header)
        else:
            header = f.read(12)
            if len(header) < 12:
                return None
            end, prop_count, _prop_len = struct.unpack("<III", header)
        name_len = struct.unpack("<B", f.read(1))[0]
        name = f.read(name_len).decode("utf-8", "replace")
        if end == 0:
            return None
        props = [cls._read_property(f, decode_arrays) for _ in range(prop_count)]
        sentinel = 25 if wide else 13
        children: List[Dict[str, Any]] = []
        while f.tell() < end - sentinel:
            child = cls._read_record(f, wide, decode_arrays)
            if child is None:
                break
            children.append(child)
        f.seek(end)
        return {"name": name, "props": props, "children": children}

    @staticmethod
    def _display_name(raw: Any) -> Optional[str]:
        """The human half of an FBX object-name property, or ``None``."""
        if not isinstance(raw, (bytes, bytearray)):
            return None
        return bytes(raw).split(_NAME_CLASS_SEPARATOR, 1)[0].decode("utf-8", "replace")


class FbxFile(_FbxFileInternal):
    """A parsed binary FBX, held read-only.

    Example:
        >>> fbx = FbxFile.load("asset.fbx")
        >>> fbx.version
        7700
        >>> fbx.take_names()
        ['Shot_1', 'Shot_2']
        >>> fbx.objects_census()["AnimationCurveNode"]
        27989
    """

    def __init__(self, path: str, version: int, roots: List[Dict[str, Any]]):
        self.path = path
        self.version = version
        self.roots = roots
        self._by_name = {r["name"]: r for r in roots}

    @classmethod
    def load(cls, path: str, decode_arrays: bool = False) -> "FbxFile":
        """Parse *path*.

        Parameters:
            path: Binary FBX file.
            decode_arrays: When False (default), array properties are read as
                ``("ARRAY", type_char, count)`` placeholders and their
                payloads skipped — the fast census mode. True inflates them
                (zlib where encoded) into python lists.

        Raises:
            ValueError: Not a binary FBX, or truncated before the header ends.
        """
        with open(path, "rb") as f:
            magic = f.read(len(FBX_MAGIC))
            if magic != FBX_MAGIC:
                raise ValueError(f"Not a binary FBX: {path}")
            version = struct.unpack("<I", f.read(4))[0]
            wide = version >= 7500
            roots: List[Dict[str, Any]] = []
            while True:
                record = cls._read_record(f, wide, decode_arrays)
                if record is None:
                    break
                roots.append(record)
        return cls(path, version, roots)

    @staticmethod
    def is_fbx(path: str) -> bool:
        """True when *path* starts with the binary-FBX magic."""
        try:
            with open(path, "rb") as f:
                return f.read(len(FBX_MAGIC)) == FBX_MAGIC
        except OSError:
            return False

    # ---- navigation -------------------------------------------------------

    def section(self, name: str) -> Optional[Dict[str, Any]]:
        """Top-level record *name* (``"Objects"``, ``"Connections"`` …)."""
        return self._by_name.get(name)

    def iter_objects(self) -> Iterator[Dict[str, Any]]:
        """Yield every child record of the ``Objects`` section."""
        objects = self.section("Objects")
        if objects:
            yield from objects["children"]

    # ---- census -----------------------------------------------------------

    def objects_census(self) -> Dict[str, int]:
        """``{record name: count}`` over the Objects section.

        Keys are FBX record names — ``Model``, ``Geometry``,
        ``AnimationStack``, ``AnimationLayer``, ``AnimationCurveNode``,
        ``AnimationCurve``, ``Deformer`` … — which is the census a
        deliverable check compares run over run.
        """
        return dict(Counter(record["name"] for record in self.iter_objects()))

    def object_names(self, kind: str) -> List[str]:
        """Display names of every Objects child whose record name is *kind*."""
        names: List[str] = []
        for record in self.iter_objects():
            if record["name"] != kind:
                continue
            for prop in record["props"]:
                name = self._display_name(prop)
                if name is not None:
                    names.append(name)
                    break
        return names

    def take_names(self) -> List[str]:
        """Animation take names — the ``AnimationStack`` display names."""
        return self.object_names("AnimationStack")

    def connections(self) -> List[Tuple[str, Any, Any, Optional[str]]]:
        """Every ``C`` record as ``(kind, child_id, parent_id, property)``.

        Property-name is ``None`` for object-object (``"OO"``) links and the
        target attribute for object-property (``"OP"``) ones.
        """
        section = self.section("Connections")
        rows: List[Tuple[str, Any, Any, Optional[str]]] = []
        for record in (section or {}).get("children", []):
            props = record["props"]
            if len(props) < 3:
                continue
            kind = (
                props[0].decode("ascii", "replace")
                if isinstance(props[0], (bytes, bytearray))
                else str(props[0])
            )
            prop_name = None
            if len(props) > 3 and isinstance(props[3], (bytes, bytearray)):
                prop_name = props[3].decode("utf-8", "replace")
            rows.append((kind, props[1], props[2], prop_name))
        return rows


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
