# !/usr/bin/python
# coding=utf-8
"""Rewrite the payload of a binary FBX -- no DCC, no FBX SDK.

:class:`FbxMedia` is the writer :mod:`fbx_file` deliberately is not, scoped to
the edits a hand-off pipeline needs on its scratch copy: replacing the
``Video`` objects' ``Content`` payloads -- the textures
``FBXExportEmbeddedTextures`` copies into the file at full authoring
resolution (:meth:`FbxMedia.downsize`), and the grayscale ones FBX2glTF would
pack as white (:meth:`FbxMedia.expand_grayscale`) -- dropping animation
takes the converter would bake for nothing (:meth:`FbxMedia.drop_takes`),
merging the takes an importer would otherwise play only one of into one
(:meth:`FbxMedia.merge_takes`), dropping the apparatus of a baked rig
(:meth:`FbxMedia.drop_apparatus`), and
embedding the files the scene records name by file name -- the lightmaps, the
reflection probe, the shadow maps -- so a deliverable needs nothing beside it
(:meth:`FbxMedia.embed_dependencies`). Everything else is copied byte for
byte. Only the record headers are re-serialised, because binary FBX stores
every record's end as an *absolute* offset, so one payload that changes size
moves every header after it.

Why it exists: measured on a production assembly, the FBX handed to FBX2glTF
was 366 MB, 353 MB of it 4096x4096 PNG, every byte of which the texture pass
afterwards resized to 2048 and threw away. Downsizing the images *in the FBX*
to that same ceiling loses nothing the deliverable would have kept, and it
runs on the pipeline's scratch payload, so the live scene is never touched --
the alternative, restaging the scene's file nodes and restoring them
afterwards, is exactly the pattern a DCC crash turns into corrupted texture
paths. Measured quiet, end to end: the push went from 419 s to 333 s -- the
converter ~365 -> ~290 s (a fifth: its cost is mostly the animation it bakes
over every node, not the images -- a 12 MB textureless export of the same
scene still took over 300 s), and every pass after it now reads 2K images
(sidecar 17 -> 9 s, optimize 21 -> 10 s), with ~600 MB less scratch per push.

Example:
    >>> FbxMedia.embedded("scene.fbx")[0]
    {'name': 'wall_Base_color.png', 'format': 'PNG', 'size': (4096, 4096), 'bytes': 19500000}
    >>> FbxMedia.downsize("scene.fbx", "scene_2k.fbx", max_size=2048)
    {'images': 29, 'resized': 27, 'before': 353200000, 'after': 61000000}
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import mmap
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from pythontk.core_utils.engines.scene_export.scene_records import Scope, SceneRecords
from pythontk.core_utils.engines.scene_export.scene_store import SceneStoreBase
from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.file_dependencies import FileDependencies
from pythontk.file_utils.mesh_convert.fbx_file import FBX_MAGIC, FbxFile

logger = logging.getLogger(__name__)


@dataclass
class _Record:
    """One binary-FBX node record, held as raw byte spans."""

    name: bytes
    prop_count: int
    #: The property list, verbatim (type tags included).
    props: Any
    children: List["_Record"] = field(default_factory=list)
    #: Bytes between the last child and the record's end -- the NULL sentinel
    #: a record with children carries, copied rather than reasoned about.
    tail: bytes = b""
    #: A replacement property list, when this record is being rewritten.
    replaced: Optional[bytes] = None
    size: int = 0

    @property
    def payload(self) -> Any:
        return self.props if self.replaced is None else self.replaced


class _FbxMediaInternal:
    """Parse and re-serialise helpers for :class:`FbxMedia`."""

    #: Containers rewritten in place. Anything else (DDS, TGA, EXR, ...) is
    #: left as found: the point is to shrink what the converter will read and
    #: re-embed, and those it either ignores or cannot decode.
    REWRITABLE_FORMATS = ("PNG", "JPEG")

    #: Pixel modes resized as they are; every other mode is converted to RGB
    #: or RGBA first (palette, bilevel), or skipped (16-bit, float) so a depth
    #: the FBX carried is never silently halved here.
    RESIZABLE_MODES = ("L", "LA", "RGB", "RGBA")

    #: Pixel modes a PNG or JPEG decodes to with no green or blue channel --
    #: what :meth:`FbxMedia.expand_grayscale` rewrites. A palette image is not
    #: one: it decodes to RGB(A).
    GRAYSCALE_MODES = ("1", "L", "LA", "I;16", "I;16B", "I;16L")

    _FOOTER_TAIL = 4 + 4 + 120 + 16  # zeros, version, zeros, magic

    #: The objects an animation take owns, owner first: a stack holds layers,
    #: a layer holds curve nodes, a curve node holds curves.
    TAKE_CHAIN = (
        b"AnimationStack",
        b"AnimationLayer",
        b"AnimationCurveNode",
        b"AnimationCurve",
    )

    @staticmethod
    def _header_fmt(wide: bool) -> Tuple[str, int]:
        return ("<QQQ", 24) if wide else ("<III", 12)

    @classmethod
    def _parse(cls, buf, pos: int, wide: bool) -> Tuple[Optional[_Record], int]:
        """Parse the record at *pos*; ``(None, end)`` for a NULL record.

        Raises:
            ValueError: A record that ends before it starts, past the buffer
                or past its parent -- a truncated or corrupt file.
        """
        fmt, hlen = cls._header_fmt(wide)
        end, count, plen = struct.unpack_from(fmt, buf, pos)
        if end and not pos < end <= len(buf):
            raise ValueError("truncated or corrupt FBX record")
        name_len = buf[pos + hlen]
        name_start = pos + hlen + 1
        name = bytes(buf[name_start : name_start + name_len])
        props_start = name_start + name_len
        if end == 0:
            return None, props_start
        props = buf[props_start : props_start + plen]
        cur = props_start + plen
        sentinel = hlen + 1
        children: List[_Record] = []
        while cur < end - sentinel:
            if struct.unpack_from(fmt, buf, cur)[0] == 0:
                break
            child, cur = cls._parse(buf, cur, wide)
            if cur > end:
                # Its overrun would swallow its siblings and the next record
                # into its tail, which a rewrite then emits twice.
                raise ValueError("truncated or corrupt FBX record")
            children.append(child)
        return _Record(name, count, props, children, bytes(buf[cur:end])), end

    @classmethod
    def _measure(cls, record: _Record, hlen: int) -> int:
        """Set and return the serialised size of *record* and its subtree."""
        size = hlen + 1 + len(record.name) + len(record.payload) + len(record.tail)
        for child in record.children:
            size += cls._measure(child, hlen)
        record.size = size
        return size

    @classmethod
    def _emit(cls, record: _Record, out, pos: int, wide: bool) -> int:
        """Write *record* at absolute *pos*; return the position after it."""
        fmt, hlen = cls._header_fmt(wide)
        payload = record.payload
        end = pos + record.size
        out.write(struct.pack(fmt, end, record.prop_count, len(payload)))
        out.write(bytes((len(record.name),)))
        out.write(record.name)
        out.write(payload)
        cur = pos + hlen + 1 + len(record.name) + len(payload)
        for child in record.children:
            cur = cls._emit(child, out, cur, wide)
        out.write(record.tail)
        return end

    @classmethod
    def _load(cls, buf) -> Tuple[int, List[_Record], bytes, bytes]:
        """``(version, roots, footer_id, footer_magic)`` of a binary FBX buffer.

        Raises:
            ValueError: Not a binary FBX, a truncated or corrupt one, or a
                footer this writer cannot re-emit faithfully (the caller then
                leaves the file alone).
        """
        if bytes(buf[: len(FBX_MAGIC)]) != FBX_MAGIC:
            raise ValueError("not a binary FBX")
        try:
            version = struct.unpack_from("<I", buf, len(FBX_MAGIC))[0]
            wide = version >= 7500
            pos = len(FBX_MAGIC) + 4
            roots: List[_Record] = []
            while True:
                record, pos = cls._parse(buf, pos, wide)
                if record is None:
                    break
                roots.append(record)
        except (struct.error, IndexError) as error:
            # A file cut short has the magic but not the body.
            raise ValueError(f"truncated or corrupt FBX: {error}") from error
        footer = bytes(buf[pos:])
        if len(footer) < 16 + cls._FOOTER_TAIL:
            raise ValueError("unrecognised FBX footer")
        tail = footer[-cls._FOOTER_TAIL :]
        stated = struct.unpack_from("<I", tail, 4)[0]
        if tail[:4] != b"\0" * 4 or stated != version or tail[8:128] != b"\0" * 120:
            raise ValueError("unrecognised FBX footer")
        return version, roots, footer[:16], footer[-16:]

    @classmethod
    def _write(cls, out, version: int, roots: List[_Record], footer_id, magic) -> None:
        wide = version >= 7500
        hlen = cls._header_fmt(wide)[1]
        out.write(FBX_MAGIC)
        out.write(struct.pack("<I", version))
        pos = len(FBX_MAGIC) + 4
        for record in roots:
            cls._measure(record, hlen)
            pos = cls._emit(record, out, pos, wide)
        out.write(b"\0" * (hlen + 1))  # the top-level NULL record
        out.write(footer_id)
        offset = pos + hlen + 1 + len(footer_id)
        # Alignment padding as the SDK writes it: up to the next 16-byte
        # boundary, and a full 16 when already aligned.
        pad = ((offset + 15) & ~15) - offset or 16
        out.write(b"\0" * pad)
        out.write(b"\0" * 4)
        out.write(struct.pack("<I", version))
        out.write(b"\0" * 120)
        out.write(magic)

    @classmethod
    def _video_records(
        cls, roots: Iterable[_Record], wired_only: bool = False
    ) -> List[Tuple[str, _Record]]:
        """``(basename, Content record)`` for every embedded ``Video`` object.

        *wired_only* keeps only the media something is connected under -- a
        material's texture. What nothing references is a file the scene
        records name (:meth:`FbxMedia.embed_dependencies`), which no
        converter reads as a texture.
        """
        roots = list(roots)
        wired = cls._wired_ids(roots) if wired_only else None
        found: List[Tuple[str, _Record]] = []
        for root in roots:
            if root.name != b"Objects":
                continue
            for obj in root.children:
                if obj.name != b"Video":
                    continue
                if wired is not None:
                    head = cls._scalars(obj)
                    if not head or head[0] not in wired:
                        continue
                content = None
                for child in obj.children:
                    if child.name == b"Content" and child.prop_count:
                        content = child
                if content is not None and bytes(content.props[:1]) == b"R":
                    found.append((cls._video_file(obj), content))
        return found

    @staticmethod
    def _video_file(video: _Record) -> str:
        """The file name a ``Video`` record's ``Filename`` names, folder-less;
        ``""`` when it names none."""
        filename = ""
        for child in video.children:
            if child.name == b"Filename" and len(child.props) > 5:
                filename = bytes(child.props[5:]).decode("utf-8", "replace")
        return os.path.basename(filename.replace("\\", "/"))

    @classmethod
    def _wired_ids(cls, roots: Iterable[_Record]) -> set:
        """Every object id a connection names as the CHILD -- wired under
        something (a Video under its Texture, a Texture under its material)."""
        wired = set()
        for root in roots:
            if root.name != b"Connections":
                continue
            for record in root.children:
                row = cls._scalars(record)
                if len(row) >= 3 and isinstance(row[1], int):
                    wired.add(row[1])
        return wired

    @staticmethod
    def _raw(record: _Record) -> bytes:
        """The bytes of a single ``R`` property."""
        return bytes(record.props[5:])

    @staticmethod
    def _pack_raw(data: bytes) -> bytes:
        return b"R" + struct.pack("<I", len(data)) + data

    @staticmethod
    def _pack_string(data: bytes) -> bytes:
        return b"S" + struct.pack("<I", len(data)) + data

    @staticmethod
    def _pack_long(value: int) -> bytes:
        return b"L" + struct.pack("<q", value)

    @staticmethod
    def _pack_int(value: int) -> bytes:
        return b"I" + struct.pack("<i", value)

    @staticmethod
    def _new_record(
        name: bytes, props: Sequence[bytes], children: Sequence[_Record], hlen: int
    ) -> _Record:
        """A record built here: packed *props*, and the NULL sentinel a record
        with children closes on (*hlen* is the layout's header length)."""
        return _Record(
            name,
            len(props),
            b"".join(props),
            list(children),
            b"\0" * (hlen + 1) if children else b"",
        )

    @classmethod
    def _media_record(cls, oid: int, name: str, data: bytes, hlen: int) -> _Record:
        """A ``Video`` holding *data* as *name*, wired to nothing.

        The fields an SDK writes for embedded media, every path the bare file
        name: the deliverable names no folder, and the name is what Unity's
        ``ModelImporter.ExtractTextures`` writes the bytes back under.
        """
        text = name.encode("utf-8")
        s = cls._pack_string

        def record(tag: bytes, *props: bytes, children=()) -> _Record:
            return cls._new_record(tag, props, children, hlen)

        def path(key: bytes) -> _Record:
            return record(b"P", s(key), s(b"KString"), s(b"XRefUrl"), s(b""), s(text))

        return record(
            b"Video",
            cls._pack_long(oid),
            s(text + b"\x00\x01Video"),
            s(b"Clip"),
            children=(
                record(b"Type", s(b"Clip")),
                record(b"Properties70", children=(path(b"Path"), path(b"RelPath"))),
                record(b"UseMipMap", cls._pack_int(0)),
                record(b"Filename", s(text)),
                record(b"RelativeFilename", s(text)),
                record(b"Content", cls._pack_raw(data)),
            ),
        )

    @staticmethod
    def _named(entry: Any) -> Tuple[str, str]:
        """``(file name, path)`` -- an entry :meth:`_add_media` embeds: a path
        goes in under its own file name, a ``(name, path)`` pair under *name*."""
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            return str(entry[0]), str(entry[1])
        path = str(entry)
        return os.path.basename(path.replace("\\", "/")), path

    @staticmethod
    def _media_id(name: str, taken: set) -> int:
        """An object id for the media *name*: the same for the same name, so
        two exports of one scene number it alike, and one *taken* never is."""
        digest = hashlib.sha256(b"pythontk.FbxMedia:" + name.casefold().encode())
        oid = int.from_bytes(digest.digest()[:8], "little") & 0x3FFF_FFFF_FFFF_FFFF
        while not oid or oid in taken:
            oid += 1
        return oid

    @classmethod
    def _shift_count(cls, record: _Record, delta: int, name: bytes = b"Count") -> None:
        """Move *record*'s *name* counter child by *delta* (never below zero),
        keeping its integer width."""
        for child in record.children:
            if child.name != name:
                continue
            head = cls._scalars(child)
            if head and isinstance(head[0], int):
                count = max(0, head[0] + delta)
                wide = bytes(child.payload[:1]) == b"L"
                child.replaced = cls._pack_long(count) if wide else cls._pack_int(count)

    @classmethod
    def _shift_definitions(
        cls, sections: Mapping[bytes, _Record], deltas: Mapping[bytes, int], hlen: int
    ) -> None:
        """Move the ``Definitions`` counts by *deltas* ``{object type: change}``
        -- each type's, and the total by what moved -- declaring a type the
        file had none of when objects of it were added. A file without the
        section is left alone."""
        definitions = sections.get(b"Definitions")
        if definitions is None:
            return
        total = 0
        for kind, delta in deltas.items():
            if not delta:
                continue
            declared = next(
                (
                    child
                    for child in definitions.children
                    if child.name == b"ObjectType" and cls._scalars(child)[:1] == [kind]
                ),
                None,
            )
            if declared is not None:
                cls._shift_count(declared, delta)
            elif delta > 0:
                definitions.children.append(
                    cls._new_record(
                        b"ObjectType",
                        [cls._pack_string(kind)],
                        [cls._new_record(b"Count", [cls._pack_int(delta)], (), hlen)],
                        hlen,
                    )
                )
            else:
                continue
            total += delta
        if total:
            cls._shift_count(definitions, total)

    @classmethod
    def _taken_ids(cls, roots: Iterable[_Record], objects: _Record) -> set:
        """Every object id the file uses -- its objects' and its connections'
        -- and 0, which none may take."""
        roots = list(roots)
        taken = {0}
        for record in objects.children:
            head = cls._scalars(record)
            if head and isinstance(head[0], int):
                taken.add(head[0])
        return taken | cls._wired_ids(roots)

    @classmethod
    def _remove_objects(
        cls,
        sections: Mapping[bytes, _Record],
        graph: Mapping[str, Any],
        removed: set,
        hlen: int,
        also: Optional[Callable[[_Record], bool]] = None,
    ) -> Dict[bytes, int]:
        """Remove every object whose id is in *removed* -- and each one *also*
        says goes, its id then joining *removed* -- every connection naming
        one, and their ``Definitions`` counts. *graph* is the file's
        :meth:`FbxMedia._model_graph`.

        Returns:
            ``{record name: count}`` of the objects removed.
        """
        objects = sections[b"Objects"]
        connections = sections.get(b"Connections")
        counts: Dict[bytes, int] = {}
        survivors = []
        for record in objects.children:
            oid = graph["id_of"].get(id(record))
            if oid in removed or (also is not None and also(record)):
                removed.add(oid)  # its connections go below
                counts[record.name] = counts.get(record.name, 0) + 1
            else:
                survivors.append(record)
        objects.children = survivors
        if connections is not None:
            connections.children = [
                record
                for record, row in graph["rows"]
                if not (len(row) >= 3 and (row[1] in removed or row[2] in removed))
            ]
        cls._shift_definitions(
            sections, {kind: -count for kind, count in counts.items()}, hlen
        )
        return counts

    #: The Model properties an importer moves a mesh by: Unity's (the FBX
    #: SDK's pivot reset) brings each node's pivots to its origin THROUGH its
    #: mesh, so every instance of one geometry must agree on them. The
    #: offsets (``RotationOffset``, ``ScalingOffset``) are translations the
    #: node keeps -- measured, instances that differ only in those land right.
    GEOMETRY_PIVOTS: Tuple[bytes, ...] = (
        b"RotationPivot",
        b"ScalingPivot",
        b"GeometricTranslation",
        b"GeometricRotation",
        b"GeometricScaling",
    )
    _PIVOT_DEFAULTS = {b"GeometricScaling": (1.0, 1.0, 1.0)}

    @classmethod
    def _pivot_set(cls, model: _Record) -> Tuple[Tuple[float, ...], ...]:
        """*model*'s :attr:`GEOMETRY_PIVOTS`, each its default where it writes none."""
        found: Dict[bytes, Tuple[float, ...]] = {}
        for child in model.children:
            if child.name != b"Properties70":
                continue
            for prop in child.children:
                head = cls._scalars(prop)
                if head and head[0] in cls.GEOMETRY_PIVOTS:
                    found[head[0]] = tuple(float(v) for v in cls._decoded(prop)[4:7])
        return tuple(
            found.get(key, cls._PIVOT_DEFAULTS.get(key, (0.0, 0.0, 0.0)))
            for key in cls.GEOMETRY_PIVOTS
        )

    #: Material slots that hold the host's own lighting, not the asset's:
    #: Maya's StingrayPBS wires Autodesk's environment cubes and BRDF table
    #: into every material. No engine samples them -- each lights its scene
    #: its own way.
    HOST_LIGHTING_SLOTS: Tuple[bytes, ...] = (
        b"Maya|TEX_global_diffuse_cube",
        b"Maya|TEX_global_specular_cube",
        b"Maya|TEX_brdf_lut",
    )

    @classmethod
    def _prop_span(cls, raw: bytes, index: int) -> Tuple[int, int]:
        """``(start, end)`` of the *index*-th property in a record's *raw* list
        (the type widths :class:`FbxFile`'s reader decodes by).

        Raises:
            ValueError: A property type this reader does not know.
        """
        pos = start = 0
        for _ in range(index + 1):
            start = pos
            kind = raw[pos : pos + 1]
            scalar = FbxFile._SCALAR_PROPS.get(kind)
            if scalar:
                pos += 1 + scalar[1]
            elif kind in (b"S", b"R"):
                pos += 5 + struct.unpack_from("<I", raw, pos + 1)[0]
            elif kind in FbxFile._ARRAY_PROPS:
                # length, encoding, then the stored (maybe compressed) bytes
                pos += 13 + struct.unpack_from("<I", raw, pos + 9)[0]
            else:
                raise ValueError(f"unknown FBX property type {kind!r}")
        return start, pos

    @classmethod
    def _replace_prop(cls, record: _Record, index: int, packed: bytes) -> None:
        """Replace *record*'s *index*-th property with *packed*, one of its own type.

        Raises:
            ValueError: The property is of another type, or not there.
        """
        raw = bytes(record.payload)
        start, end = cls._prop_span(raw, index)
        if end > len(raw) or raw[start : start + 1] != packed[:1]:
            raise ValueError(
                f"property {index} of {record.name!r} is not a {packed[:1]!r}"
            )
        record.replaced = raw[:start] + packed + raw[end:]

    @classmethod
    def _add_media(
        cls,
        sections: Dict[bytes, _Record],
        entries: Sequence[Tuple[str, str]],
        hlen: int,
    ) -> Dict[str, Any]:
        """Append a :meth:`_media_record` per ``(name, path)`` in *entries*.

        Each file goes in under *name* -- the spelling its record joins on,
        which a case-insensitive file system may have found under another. A
        name the file already embeds -- compared without case, as the file
        systems Unity extracts onto compare -- is reported ``present`` and
        left; so is a repeat in *entries*. The ``Definitions`` count each
        added object, declaring the ``Video`` type when the file had none.

        Raises:
            ValueError: The file has no ``Objects`` section.
            OSError: A file in *entries* cannot be read.
        """
        report: Dict[str, Any] = {"embedded": [], "present": [], "bytes": 0}
        objects = sections.get(b"Objects")
        if objects is None:
            raise ValueError("FBX has no Objects section")
        roots = list(sections.values())
        have = {name.casefold() for name, _record in cls._video_records(roots)}
        taken = cls._taken_ids(roots, objects)
        for name, path in entries:
            if not name:
                continue
            if name.casefold() in have:
                report["present"].append(name)
                continue
            with open(path, "rb") as fh:
                data = fh.read()
            oid = cls._media_id(name, taken)
            taken.add(oid)
            have.add(name.casefold())
            objects.children.append(cls._media_record(oid, name, data, hlen))
            report["embedded"].append(name)
            report["bytes"] += len(data)
        cls._shift_definitions(sections, {b"Video": len(report["embedded"])}, hlen)
        return report

    #: The scene's own carrier node; a module's is ``NS:data_export`` (or a
    #: renamed ``data_export1``).
    _OWN_CARRIER = SceneStoreBase.name(Scope.DELIVERABLE)

    @classmethod
    def _models(
        cls, roots: Iterable[_Record]
    ) -> Iterator[Tuple[Optional[str], _Record]]:
        """``(display name, record)`` of every ``Model``, in file order."""
        for root in roots:
            if root.name != b"Objects":
                continue
            for record in root.children:
                if record.name == b"Model":
                    head = cls._scalars(record)
                    name = FbxFile._display_name(head[1]) if len(head) > 1 else None
                    yield name, record

    @classmethod
    def _carrier_props(
        cls, roots: Iterable[_Record], keys: Iterable[str]
    ) -> Iterator[Tuple[Optional[str], str, Any, _Record]]:
        """:meth:`_carrier_channels`, each with the ``P`` record it was read from."""
        wanted = {key.encode("utf-8"): key for key in keys}
        for name, record in cls._models(roots):
            for child in record.children:
                if child.name != b"Properties70":
                    continue
                for prop in child.children:
                    head = cls._scalars(prop)
                    if len(head) < 5 or head[0] not in wanted:
                        continue
                    value = head[4]
                    if not isinstance(value, bytes) or not value.strip():
                        continue
                    try:
                        payload = json.loads(value.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        continue
                    yield name, wanted[head[0]], payload, prop

    @classmethod
    def _carrier_channels(
        cls, roots: Iterable[_Record], keys: Iterable[str]
    ) -> Iterator[Tuple[Optional[str], str, Any]]:
        """``(carrier name, record key, decoded payload)`` for every Model's
        user property among *keys*, in file order -- one per carrier (a
        referenced module ships its own). A blank or malformed value is
        skipped."""
        for name, key, payload, _prop in cls._carrier_props(roots, keys):
            yield name, key, payload

    @classmethod
    def _applied_channel(
        cls, roots: Iterable[_Record], key: str
    ) -> Optional[Tuple[_Record, Any]]:
        """The ``P`` record holding the *key* record a reader applies, and its
        payload (:meth:`_applied_channels`); ``None`` when no carrier holds
        one."""
        return cls._applied_channels(roots, (key,)).get(key)

    @classmethod
    def _applied_channels(
        cls, roots: Iterable[_Record], keys: Iterable[str]
    ) -> Dict[str, Tuple[_Record, Any]]:
        """``{key: (P record, payload)}`` for each of *keys* a reader applies,
        in one walk: the scene's own carrier's, else the first module's in
        file order -- the copy :meth:`FbxMedia.reader_records` takes. The
        handoff is the scene's own alone: one on a module's carrier describes
        that module, never the file (the FBX handoff's own rule)."""
        own: Dict[str, Tuple[_Record, Any]] = {}
        first: Dict[str, Tuple[_Record, Any]] = {}
        for name, key, payload, prop in cls._carrier_props(roots, tuple(keys)):
            if name == cls._OWN_CARRIER:
                own.setdefault(key, (prop, payload))
            elif key != SceneRecords.HANDOFF.key:
                first.setdefault(key, (prop, payload))
        return {**first, **own}


    @staticmethod
    def _field_values(payload: Any, path: Sequence[str]) -> Iterator[Any]:
        """The values at *path* in *payload*; ``"*"`` walks a list."""
        if not path:
            yield payload
            return
        step, rest = path[0], path[1:]
        if step == "*":
            items = payload if isinstance(payload, list) else []
        elif isinstance(payload, dict) and step in payload:
            items = [payload[step]]
        else:
            items = []
        for item in items:
            yield from _FbxMediaInternal._field_values(item, rest)

    @staticmethod
    def _scalars(record: _Record) -> List[Any]:
        """The record's leading ``L``/``I``/``S`` properties, decoded.

        Stops at the first property of any other type: the callers read object
        ids, display names and connection rows, which are all at the head.
        """
        raw = bytes(record.payload)
        out: List[Any] = []
        pos = 0
        while pos < len(raw):
            kind = raw[pos : pos + 1]
            if kind == b"L":
                out.append(struct.unpack_from("<q", raw, pos + 1)[0])
                pos += 9
            elif kind == b"I":
                out.append(struct.unpack_from("<i", raw, pos + 1)[0])
                pos += 5
            elif kind == b"S":
                length = struct.unpack_from("<I", raw, pos + 1)[0]
                out.append(raw[pos + 5 : pos + 5 + length])
                pos += 5 + length
            else:
                break
        return out

    @classmethod
    def _write_part(cls, target: str, version, roots, footer_id, magic):
        """Serialise into a tracked ``.part`` beside *target*; return its path.

        :meth:`FileUtils.atomic_write` with the promotion left to the caller:
        the replace has to wait until its mmap closes, because Windows refuses
        to replace a mapped file. A raise inside :meth:`_write` (disk full is
        realistic for a multi-hundred-MB payload) removes the partial file; a
        process that dies first leaves it to the store's age-gated sweep.
        """

        def write(part: str) -> None:
            with open(part, "wb") as out:
                cls._write(out, version, roots, footer_id, magic)

        return FileUtils.atomic_write(target, write, promote=False)

    @staticmethod
    def _promote(part: str, target: str) -> None:
        """Swap the finished *part* in for *target*, or remove it.

        The other half of :meth:`_write_part`, run once its mmap is closed. A
        replace still refused once :meth:`FileUtils.replace_file` has waited
        out the lock leaves *target* exactly as it was -- and takes the
        ``.part`` with it: left behind, it was a full-size copy of the file in
        the synced export folder, one per refused pass per run (production,
        2026-10-07: four, ~970 MB).
        """
        try:
            FileUtils.replace_file(part, target)
        except BaseException:
            try:
                os.remove(part)
            except OSError:
                pass
            raise

    @classmethod
    def _rewrite_images(
        cls,
        src: str,
        dst: Optional[str],
        edit: Callable[[str, Any], Any],
        *,
        count_key: str,
        workers: Optional[int],
        png_compress_level: int,
        jpeg_quality: int,
        smaller_only: bool = False,
    ) -> Dict[str, Any]:
        """Re-encode each embedded PNG/JPEG that *edit* returns a new image for.

        *edit* gets the image's basename and its opened Pillow image (header
        read, pixels not yet loaded) and returns the replacement, or ``None``
        to keep the embedded bytes. A replacement keeps its container, so the
        ``Filename`` the SDK extracts it under still describes the bytes; one
        that fails to decode or encode keeps its bytes, with a warning.
        *smaller_only* keeps them too when the re-encode is not smaller.

        Only media a material's texture references is read: the files the
        scene records name (:meth:`FbxMedia.embed_dependencies`) are wired to
        nothing and read by no converter -- a horizon map is data a resize
        destroys, and a lightmap EXR no image Pillow opens.

        Returns:
            ``{"images", <count_key>, "before", "after"}`` -- embedded image
            count, how many were replaced, and the embedded bytes before and
            after. When nothing was replaced the file is not written at all.
        """
        from PIL import Image

        with (
            open(src, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            version, roots, footer_id, magic = cls._load(buf)
            videos = cls._video_records(roots, wired_only=True)
            report = {
                "images": len(videos),
                count_key: 0,
                # The record length, not a copy of every payload just to size it.
                "before": sum(len(r.props) - 5 for _n, r in videos),
                "after": 0,
            }

            def rewrite(item: Tuple[str, _Record]) -> Optional[bytes]:
                name, record = item
                data = cls._raw(record)
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        fmt = image.format
                        if fmt not in cls.REWRITABLE_FORMATS:
                            return None
                        replacement = edit(name, image)
                        if replacement is None:
                            return None
                        out = io.BytesIO()
                        if fmt == "PNG":
                            replacement.save(
                                out, format="PNG", compress_level=png_compress_level
                            )
                        else:
                            if replacement.mode == "RGBA":
                                replacement = replacement.convert("RGB")
                            replacement.save(out, format="JPEG", quality=jpeg_quality)
                except Exception as error:  # noqa: BLE001 -- keep the authored bytes
                    logger.warning("FbxMedia: %s left as found: %s", name, error)
                    return None
                encoded = out.getvalue()
                if smaller_only and len(encoded) >= len(data):
                    return None
                return encoded

            from pythontk.img_utils._img_utils import ImgUtils

            count = max(1, min(ImgUtils.encode_workers(workers), len(videos) or 1))
            with ThreadPoolExecutor(
                max_workers=count, thread_name_prefix="ptk-fbx-media"
            ) as pool:
                results = list(pool.map(rewrite, videos))
            for (_name, record), encoded in zip(videos, results):
                if encoded is not None:
                    record.replaced = cls._pack_raw(encoded)
                    report[count_key] += 1
            report["after"] = sum(len(r.payload) - 5 for _n, r in videos)
            if not report[count_key]:
                return report

            target = dst or src
            part = cls._write_part(target, version, roots, footer_id, magic)
        # After the mmap closes: Windows refuses to replace a mapped file, and
        # `target` is `src` for an in-place run.
        cls._promote(part, target)
        return report

    # ---- takes ------------------------------------------------------------

    @classmethod
    def _head_string(cls, record: _Record) -> Optional[str]:
        """*record*'s first property as text, when it is a string."""
        head = cls._scalars(record)
        if head and isinstance(head[0], bytes):
            return head[0].decode("utf-8", "replace")
        return None

    @staticmethod
    def _decoded(record: _Record) -> List[Any]:
        """Every property of *record*, decoded (arrays as lists).

        Raises:
            ValueError: A property that does not decode -- a cut payload or a
                broken zlib stream -- as ``FbxFile.load`` says it.
        """
        stream = io.BytesIO(bytes(record.payload))
        try:
            return [
                FbxFile._read_property(stream, True) for _ in range(record.prop_count)
            ]
        except (struct.error, zlib.error) as error:
            raise ValueError(
                f"Corrupt FBX record {bytes(record.name)!r}: {error}"
            ) from error

    @classmethod
    def _stacks(cls, sections: Mapping[bytes, _Record]) -> Dict[int, str]:
        """``{uid: take name}`` of every ``AnimationStack``, in file order."""
        stacks: Dict[int, str] = {}
        objects = sections.get(b"Objects")
        for record in objects.children if objects is not None else ():
            if record.name != b"AnimationStack":
                continue
            head = cls._scalars(record)
            if len(head) > 1 and isinstance(head[0], int):
                stacks[head[0]] = FbxFile._display_name(head[1]) or ""
        return stacks

    @classmethod
    def _drop_stacks(
        cls, sections: Mapping[bytes, _Record], stacks: Mapping[int, str]
    ) -> Dict[str, Any]:
        """Remove the ``AnimationStack`` objects *stacks* names, and everything
        only they own -- :meth:`FbxMedia.drop_takes` on a parsed file.

        Returns:
            ``{"takes", "objects", "connections"}`` as ``drop_takes`` reports.
        """
        objects = sections[b"Objects"]
        alive = {
            name for uid, name in cls._stacks(sections).items() if uid not in stacks
        }
        kind_of: Dict[int, bytes] = {}
        id_of_record: Dict[int, int] = {}
        for record in objects.children:
            if record.name not in cls.TAKE_CHAIN:
                continue
            head = cls._scalars(record)
            if not head or not isinstance(head[0], int):
                continue
            kind_of[head[0]] = record.name
            id_of_record[id(record)] = head[0]

        connections = sections.get(b"Connections")
        rows = [
            (record, cls._scalars(record))
            for record in (connections.children if connections else [])
        ]
        parents: Dict[int, List[int]] = {}
        for _record, row in rows:
            if len(row) >= 3 and row[1] in kind_of:
                parents.setdefault(row[1], []).append(row[2])

        # Owner-first, so every owner's verdict is settled before its
        # members are judged.
        dropped = set(stacks)
        for level in range(1, len(cls.TAKE_CHAIN)):
            owner, member = cls.TAKE_CHAIN[level - 1], cls.TAKE_CHAIN[level]
            for oid, kind in kind_of.items():
                if kind != member:
                    continue
                owners = [p for p in parents.get(oid, ()) if kind_of.get(p) == owner]
                if owners and all(p in dropped for p in owners):
                    dropped.add(oid)

        removed: Dict[bytes, int] = {}
        survivors = []
        for record in objects.children:
            oid = id_of_record.get(id(record))
            if oid in dropped:
                removed[record.name] = removed.get(record.name, 0) + 1
            else:
                survivors.append(record)
        objects.children = survivors

        report: Dict[str, Any] = {"takes": list(stacks.values()), "objects": {}}
        report["connections"] = 0
        if connections is not None:
            connections.children = [
                record
                for record, row in rows
                if not (len(row) >= 3 and (row[1] in dropped or row[2] in dropped))
            ]
            report["connections"] = len(rows) - len(connections.children)

        # A Takes entry is a name: it goes with the last stack of that name.
        gone = set(stacks.values()) - alive
        takes = sections.get(b"Takes")
        if takes is not None:
            takes.children = [
                child
                for child in takes.children
                if not (child.name == b"Take" and cls._head_string(child) in gone)
            ]
            left = [cls._head_string(c) for c in takes.children if c.name == b"Take"]
            for child in takes.children:
                if child.name == b"Current" and cls._head_string(child) in gone:
                    if left:
                        child.replaced = cls._pack_string(left[0].encode("utf-8"))

        definitions = sections.get(b"Definitions")
        if definitions is not None:
            total = 0
            for object_type in definitions.children:
                if object_type.name != b"ObjectType":
                    continue
                head = cls._scalars(object_type)
                count = removed.get(head[0], 0) if head else 0
                if count:
                    cls._shift_count(object_type, -count)
                    total += count
            if total:
                cls._shift_count(definitions, -total)
        report["objects"] = {k.decode(): v for k, v in removed.items()}
        return report

    @classmethod
    def _take_rows(
        cls, objects: _Record, connections: Optional[_Record]
    ) -> Tuple[List[Tuple[str, int, str, Any]], List[Tuple[str, Any, Any, Any]]]:
        """``(rows, links)`` for :meth:`FbxFile._build_take_model`: each object
        (an ``AnimationCurve`` with its key arrays decoded) and each
        connection of a parsed file."""
        wanted = {name.encode("ascii") for name in FbxFile.CURVE_ARRAYS}
        rows = []
        for record in objects.children:
            head = cls._scalars(record)
            if not head or not isinstance(head[0], int):
                continue
            arrays = None
            if record.name == b"AnimationCurve":
                arrays = {
                    child.name.decode("ascii"): cls._decoded(child)[0]
                    for child in record.children
                    if child.name in wanted and child.prop_count
                }
            display = FbxFile._display_name(head[1]) if len(head) > 1 else None
            rows.append(
                (record.name.decode("ascii", "replace"), head[0], display or "", arrays)
            )
        links = []
        for record in connections.children if connections is not None else ():
            row = cls._scalars(record)
            if len(row) < 3:
                continue
            kind = (
                row[0].decode("ascii", "replace")
                if isinstance(row[0], bytes)
                else str(row[0])
            )
            prop = (
                row[3].decode("utf-8", "replace")
                if len(row) > 3 and isinstance(row[3], bytes)
                else None
            )
            links.append((kind, row[1], row[2], prop))
        return rows, links

    @classmethod
    def _takes_settings(
        cls, sections: Mapping[bytes, _Record]
    ) -> Tuple[Optional[str], Optional[float]]:
        """``(current take, frames per second)`` a parsed file declares."""
        current = None
        for child in getattr(sections.get(b"Takes"), "children", ()):
            if child.name == b"Current":
                current = cls._head_string(child) or None
        found: Dict[bytes, Any] = {}
        for child in getattr(sections.get(b"GlobalSettings"), "children", ()):
            if child.name != b"Properties70":
                continue
            for prop in child.children:
                values = cls._decoded(prop)
                if len(values) >= 5 and values[0] in (b"TimeMode", b"CustomFrameRate"):
                    found[values[0]] = values[4]
        fps = FbxFile._frame_rate(found.get(b"TimeMode"), found.get(b"CustomFrameRate"))
        return current, fps

    @staticmethod
    def _pack_array(code: str, values: Sequence[Any]) -> bytes:
        """An FBX array property, encoded as :class:`FbxWriter` writes one."""
        from pythontk.file_utils.mesh_convert._fbx_writer import FbxWriter

        buf = bytearray()
        FbxWriter._write_property(buf, FbxWriter.array(code, values))
        return bytes(buf)

    @classmethod
    def _write_curve(
        cls,
        record: _Record,
        times: Sequence[int],
        values: Sequence[float],
        flags: Sequence[int],
        data: Sequence[Tuple[float, ...]],
        counts: Sequence[int],
        hlen: int,
    ) -> None:
        """Replace an ``AnimationCurve``'s keys and attribute runs (*flags*,
        *data* and *counts* per run). Each array keeps the element type the
        file stored it in."""
        arrays = {
            b"KeyTime": ("l", list(times)),
            b"KeyValueFloat": ("f", list(values)),
            b"KeyAttrFlags": ("i", list(flags)),
            b"KeyAttrDataFloat": ("f", [x for d in data for x in d]),
            b"KeyAttrRefCount": ("i", list(counts)),
        }
        for child in record.children:
            if child.name in arrays:
                code, items = arrays.pop(child.name)
                stored = bytes(child.payload[:1]).decode("ascii", "replace")
                if stored and stored in "fdli":
                    code = stored
                child.replaced = cls._pack_array(code, items)
        for tag, (code, items) in arrays.items():  # a curve that lacked one
            record.children.append(
                cls._new_record(tag, [cls._pack_array(code, items)], (), hlen)
            )

    @classmethod
    def _connection(
        cls, child: int, parent: int, prop: Optional[str], hlen: int
    ) -> _Record:
        """A ``C`` record: ``OO`` child under parent, ``OP`` onto *prop*."""
        props = [
            cls._pack_string(b"OO" if prop is None else b"OP"),
            cls._pack_long(child),
            cls._pack_long(parent),
        ]
        if prop is not None:
            props.append(cls._pack_string(prop.encode("utf-8")))
        return cls._new_record(b"C", props, (), hlen)

    @staticmethod
    def _find_properties(record: _Record) -> Optional[_Record]:
        """*record*'s ``Properties70`` child, or ``None``."""
        return next((c for c in record.children if c.name == b"Properties70"), None)

    @classmethod
    def _properties(cls, record: _Record, hlen: int) -> _Record:
        """*record*'s ``Properties70`` child, added when it has none."""
        block = cls._find_properties(record)
        if block is None:
            block = _Record(b"Properties70", 0, b"", [], b"\0" * (hlen + 1))
            if not record.children and not record.tail:
                record.tail = b"\0" * (hlen + 1)  # a leaf gains a closing NULL
            record.children.append(block)
        return block

    @classmethod
    def _declare_component(
        cls, node: _Record, donor: Optional[_Record], component: str, hlen: int
    ) -> None:
        """Give curve node *node* the ``P`` row for *component* when it has
        none: the *donor* node's (the curve's former node), else a zero."""
        wanted = component.encode("utf-8")
        block = cls._properties(node, hlen)
        if any(cls._scalars(p)[:1] == [wanted] for p in block.children):
            return
        source_block = cls._find_properties(donor) if donor is not None else None
        for source in source_block.children if source_block is not None else ():
            if cls._scalars(source)[:1] == [wanted]:
                block.children.append(
                    _Record(source.name, source.prop_count, bytes(source.payload))
                )
                return
        s = cls._pack_string
        block.children.append(
            cls._new_record(
                b"P",
                [
                    s(wanted),
                    s(b"Number"),
                    s(b""),
                    s(b"A"),
                    b"D" + struct.pack("<d", 0.0),
                ],
                (),
                hlen,
            )
        )

    @classmethod
    def _set_times(cls, record: _Record, times: Mapping[bytes, int], hlen: int) -> None:
        """Set the ``KTime`` rows *times* names on *record*'s
        ``Properties70``, adding any it lacks."""
        block = cls._properties(record, hlen)
        pending = dict(times)
        s = cls._pack_string
        for prop in block.children:
            head = cls._scalars(prop)
            if len(head) >= 4 and head[0] in pending:
                value = pending.pop(head[0])
                prop.replaced = b"".join(s(h) for h in head[:4]) + cls._pack_long(value)
                prop.prop_count = 5
        for key, value in pending.items():
            block.children.append(
                cls._new_record(
                    b"P",
                    [s(key), s(b"KTime"), s(b"Time"), s(b""), cls._pack_long(value)],
                    (),
                    hlen,
                )
            )

    @classmethod
    def _rename_object(cls, record: _Record, old: str, new: str, kind: bytes) -> None:
        """Name the object *record* -- ``(uid, "<name>\\0\\x01<kind>", subclass)``
        -- *new*, when it is named *old*."""
        head = cls._scalars(record)
        if new == old or len(head) < 3 or record.prop_count != 3:
            return
        if FbxFile._display_name(head[1]) != old:
            return
        label = new.encode("utf-8") + b"\x00\x01" + kind
        record.replaced = (
            cls._pack_long(head[0])
            + cls._pack_string(label)
            + cls._pack_string(head[2])
        )

    @classmethod
    def _rename_take(
        cls,
        sections: Mapping[bytes, _Record],
        stack: _Record,
        old: str,
        new: str,
        span: Sequence[int],
        hlen: int,
    ) -> None:
        """Name the take on *stack* *new* everywhere the file names it -- the
        stack, its ``Takes`` entry (file name and times, set to *span*'s
        extent) and ``Takes/Current``."""
        s = cls._pack_string
        cls._rename_object(stack, old, new, b"AnimStack")
        takes = sections.get(b"Takes")
        if takes is None:
            return
        if not takes.children and not takes.tail:
            takes.tail = b"\0" * (hlen + 1)  # it gains a child below
        times = []
        if span:
            times = [cls._pack_long(min(span)), cls._pack_long(max(span))]
        entry = next(
            (
                c
                for c in takes.children
                if c.name == b"Take" and cls._head_string(c) == old
            ),
            None,
        )
        if entry is None:
            entry = cls._new_record(b"Take", [s(b"")], [], hlen)
            takes.children.append(entry)
        entry.replaced = s(new.encode("utf-8"))
        fields = {
            b"FileName": [s(new.replace(" ", "_").encode("utf-8") + b".tak")],
            **({b"LocalTime": times, b"ReferenceTime": times} if times else {}),
        }
        for child in entry.children:
            if child.name in fields:
                props = fields.pop(child.name)
                child.replaced, child.prop_count = b"".join(props), len(props)
        if fields and not entry.children and not entry.tail:
            entry.tail = b"\0" * (hlen + 1)
        for tag, props in fields.items():
            entry.children.append(cls._new_record(tag, props, (), hlen))
        current = next((c for c in takes.children if c.name == b"Current"), None)
        if current is None:
            current = cls._new_record(b"Current", [s(b"")], (), hlen)
            takes.children.insert(0, current)
        current.replaced = s(new.encode("utf-8"))

    @classmethod
    def _stack_times(cls, record: _Record) -> List[int]:
        """The ``KTime`` values of a stack's ``Properties70``."""
        found = []
        for child in record.children:
            if child.name != b"Properties70":
                continue
            for prop in child.children:
                head = cls._scalars(prop)
                if len(head) >= 5 and head[1] == b"KTime" and isinstance(head[4], int):
                    found.append(head[4])
        return found


class FbxMedia(_FbxMediaInternal):
    """Read and rewrite the media a binary FBX embeds."""

    @classmethod
    def embedded(cls, path: str) -> List[Dict[str, Any]]:
        """Every embedded image: ``{"name", "format", "size", "bytes"}``.

        ``format`` / ``size`` are ``None`` when Pillow cannot decode the
        payload (a DDS cube map, say); ``bytes`` is always the embedded length.
        """
        with (
            open(path, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            _version, roots, _fid, _magic = cls._load(buf)
            rows: List[Dict[str, Any]] = []
            try:
                from PIL import Image
            except ImportError:  # the byte counts are still an answer
                Image = None
            for name, record in cls._video_records(roots):
                data = cls._raw(record)
                fmt = size = None
                try:
                    if Image is None:
                        raise ValueError("Pillow unavailable")
                    with Image.open(io.BytesIO(data)) as image:
                        fmt, size = image.format, image.size
                except Exception:  # noqa: BLE001 -- undecodable is a valid answer
                    pass
                rows.append(
                    {"name": name, "format": fmt, "size": size, "bytes": len(data)}
                )
        return rows

    @classmethod
    def downsize(
        cls,
        src: str,
        dst: Optional[str] = None,
        *,
        max_size: int,
        exempt: Iterable[str] = (),
        png_compress_level: int = 1,
        jpeg_quality: int = 90,
        workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Resize every embedded PNG/JPEG whose longest edge exceeds *max_size*.

        The same rule :meth:`MeshConvert.optimize_glb_textures` applies to the
        GLB afterwards -- longest edge to *max_size*, aspect kept, Lanczos, a
        normal map (by its registry filename type here, its slot there)
        resampled as one (``ImgUtils.resize_image(normal_map=True)``) --
        so an image this shrinks is one that pass would have shrunk anyway;
        pass its ceiling and the deliverable is unchanged. Each image keeps
        its container, so the ``Filename`` the SDK extracts it under still
        describes the bytes. The PNG level defaults low because the result is
        transport: the converter re-embeds it and the texture pass re-encodes
        it, so deflate effort spent here is paid twice and kept nowhere.

        Parameters:
            src: The FBX to read.
            dst: Where to write; ``None`` rewrites *src* in place (through a
                sibling ``.part`` and an atomic replace).
            max_size: Longest-edge ceiling in pixels. ``0`` resizes nothing.
            exempt: Image basenames to leave at their authored size.
            png_compress_level: zlib level for re-encoded PNGs (0-9).
            jpeg_quality: Quality for re-encoded JPEGs.
            workers: Decode/encode threads; ``None`` takes the shared encode cap
                (:meth:`ImgUtils.encode_workers`).

        Returns:
            ``{"images", "resized", "before", "after"}`` -- embedded image
            count, how many were rewritten, and the embedded bytes before and
            after. When nothing qualifies the file is not rewritten at all.

        Raises:
            ValueError: *src* is not a binary FBX this writer can re-emit.
        """
        from pythontk.core_utils.engines.textures.map_registry import MapRegistry
        from pythontk.img_utils._img_utils import ImgUtils

        exempt = {os.path.basename(str(name)) for name in exempt}
        registry = MapRegistry()

        def shrink(name: str, image: Any) -> Any:
            if not max_size or name in exempt or max(image.size) <= max_size:
                return None
            image.load()
            if image.mode in ("P", "1"):
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            if image.mode not in cls.RESIZABLE_MODES:
                logger.debug(
                    "FbxMedia: %s left at %s (mode %s)", name, image.size, image.mode
                )
                return None
            scale = max_size / float(max(image.size))
            target = tuple(max(1, round(edge * scale)) for edge in image.size)
            # Only a filename here, no material slot: the registry classifies
            # it, and resize_image still declines a map whose texels are not
            # unit normals, so a misnamed one only costs the probe.
            normal = (
                registry.resolve_type_from_path(name)
                in MapRegistry.TANGENT_NORMAL_TYPES
            )
            return ImgUtils.resize_image(image, *target, normal_map=normal)

        return cls._rewrite_images(
            src,
            dst,
            shrink,
            count_key="resized",
            workers=workers,
            png_compress_level=png_compress_level,
            jpeg_quality=jpeg_quality,
            smaller_only=True,
        )

    @classmethod
    def expand_grayscale(
        cls,
        src: str,
        dst: Optional[str] = None,
        *,
        png_compress_level: int = 1,
        jpeg_quality: int = 95,
        workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Re-encode every embedded grayscale PNG/JPEG with colour channels.

        Each channel of the result carries the image's own gray value, and an
        alpha channel is kept, so a reader of red, green or blue reads what the
        map meant. A 16-bit map is rescaled over its full range: Pillow's own
        conversion clips everything above 255 to white.

        Why it exists: FBX2glTF 0.13.1 packs a material's occlusion-roughness-
        metallic texture from each map's red, green and blue, and reads a
        channel the decoded image lacks as white. Measured on a Stingray PBS
        quad (roughness a 60..220 ramp, metallic 8): stored as RGB the packed
        texture carries both; stored grayscale -- L, LA, 16-bit or a grayscale
        JPEG -- green and blue are 255 on every texel, roughness 1 and metallic
        1, which renders black under a lightmap. The fault is per map (a
        grayscale roughness beside an RGB metallic whitens roughness alone),
        and it is the common case: Maya's ``GameShader`` splits a packed ORM
        into grayscale maps. The converter reads the embedded copies its SDK
        extracts, so rewriting the payload fixes an FBX from any producer.

        Parameters:
            src: The FBX to read.
            dst: Where to write; ``None`` rewrites *src* in place.
            png_compress_level: zlib level for re-encoded PNGs (0-9), low for
                the reason :meth:`downsize` gives.
            jpeg_quality: Quality for re-encoded JPEGs -- high, since nothing
                is resized to hide the extra generation.
            workers: Decode/encode threads; ``None`` takes the shared encode cap
                (:meth:`ImgUtils.encode_workers`).

        Returns:
            ``{"images", "expanded", "before", "after"}`` -- embedded image
            count, how many were rewritten, and the embedded bytes before and
            after. When none is grayscale the file is not written at all.

        Raises:
            ValueError: *src* is not a binary FBX this writer can re-emit.
        """

        def expand(_name: str, image: Any) -> Any:
            if image.mode not in cls.GRAYSCALE_MODES:
                return None
            image.load()
            if image.mode.startswith("I;16"):
                image = image.convert("I").point(lambda v: v / 257 + 0.5).convert("L")
            return image.convert("RGBA" if image.mode == "LA" else "RGB")

        return cls._rewrite_images(
            src,
            dst,
            expand,
            count_key="expanded",
            workers=workers,
            png_compress_level=png_compress_level,
            jpeg_quality=jpeg_quality,
        )

    #: Where each scene record names a file by FILE NAME -- what a deliverable
    #: depends on beyond the textures its materials reference: ``{record key:
    #: (field path, ...)}``, ``"*"`` walking a list. A record that starts
    #: naming a file adds its paths here and :meth:`embed_dependencies` ships it.
    DEPENDENCY_FIELDS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
        SceneRecords.LIGHTMAPS.key: (
            ("objects", "*", "map"),
            ("probe", "map"),
            ("probe", "highlights", "*", "map"),
        ),
        SceneRecords.SHADOWS.key: (
            ("planes", "*", "texture"),
            ("planes", "*", "atlas", "texture"),
            ("planes", "*", "horizon", "texture"),
        ),
    }

    #: Where each record of :attr:`DEPENDENCY_FIELDS` lists the scene nodes it
    #: describes, one entry per node: ``{record key: (list field, name
    #: field)}`` -- what :meth:`reader_records` narrows a module's copy by.
    NODE_ENTRIES: Dict[str, Tuple[str, str]] = {
        SceneRecords.LIGHTMAPS.key: ("objects", "name"),
        SceneRecords.SHADOWS.key: ("planes", "name"),
    }

    @classmethod
    def _dependency_names(cls, roots: Iterable[_Record]) -> List[str]:
        """:meth:`dependencies` of an already-parsed file."""
        roots = list(roots)
        read = cls.reader_records(
            cls._carrier_channels(roots, cls.DEPENDENCY_FIELDS),
            nodes=[name for name, _record in cls._models(roots) if name],
        )
        return cls.named_files({key: [payload] for key, payload in read.items()})

    @classmethod
    def reader_records(
        cls,
        channels: Iterable[Tuple[Optional[str], str, Any]],
        nodes: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Each scene record as a reader of the file applies it.

        A file carries the scene's own ``data_export`` and, from an assembly
        export, one ``NS:data_export`` per referenced module: that module's
        records as it was last exported alone -- its whole bake, whatever the
        export set. A reader takes each record from the scene's own carrier,
        and a module's only where the scene's carries none, the first in file
        order (unitytk ``CarrierImport.Collect``,
        ``_Sidecar._data_export_carrier``), so a module's copy beside the
        scene's is never applied. A module's copy read alone keeps only the
        entries whose node the file carries (:attr:`NODE_ENTRIES`; the name
        exactly, else its namespace-stripped leaf -- the appliers' match), and
        a copy with none of them left names nothing: no reader binds it, so
        neither its maps nor its probe belong to the file. The scene's own
        copy is taken as written: its producer scopes it to the export set.

        What the post-write embed ships and the verifier's
        ``fbx_dependencies`` gate expects: an assembly export that left a
        module's objects out embedded the module's maps, or failed the gate
        on them where they could not be found.

        Parameters:
            channels: ``(carrier name, record key, decoded payload)`` for each
                carrier's channel, in file order.
            nodes: The names of the nodes the file carries; ``None`` takes a
                module's copy whole.

        Returns:
            ``{record key: payload}`` -- a module's copy narrowed to the nodes
            the file carries, and left out when none is.
        """
        read: Dict[str, Tuple[bool, Any]] = {}
        for carrier, key, payload in channels:
            own = carrier == cls._OWN_CARRIER
            held = read.get(key)
            if held is None or (own and not held[0]):
                read[key] = (own, payload)
        names = leaves = None
        if nodes is not None:
            names = {name for name in nodes if name}
            leaves = {name.rsplit(":", 1)[-1] for name in names}
        records: Dict[str, Any] = {}
        for key, (own, payload) in read.items():
            if not own and names is not None:
                payload = cls._carried_entries(key, payload, names, leaves)
                if payload is None:
                    continue
            records[key] = payload
        return records

    @classmethod
    def _carried_entries(
        cls, key: str, payload: Any, names: set, leaves: set
    ) -> Optional[Any]:
        """*payload* with only the :attr:`NODE_ENTRIES` entries whose node is
        among *names* (exactly, else by namespace-stripped leaf in *leaves*);
        ``None`` when none is. A record that lists no nodes is returned as it
        is."""
        spec = cls.NODE_ENTRIES.get(key)
        if spec is None:
            return payload
        field_name, node_field = spec
        entries = payload.get(field_name) if isinstance(payload, Mapping) else None
        if not isinstance(entries, list):
            return None
        kept = []
        for entry in entries:
            node = entry.get(node_field) if isinstance(entry, Mapping) else None
            if not isinstance(node, str) or not node:
                continue
            if node in names or node.rsplit(":", 1)[-1] in leaves:
                kept.append(entry)
        if not kept:
            return None
        if len(kept) == len(entries):
            return payload
        return {**payload, field_name: kept}

    @classmethod
    def named_files(cls, payloads: Mapping[str, Iterable[Any]]) -> List[str]:
        """The file names decoded scene records name (:attr:`DEPENDENCY_FIELDS`).

        The walk :meth:`dependencies` runs on an FBX, for a reader that
        already holds the channels -- the verifier's parsed file, a GLB's
        extras. In record order, each name once (compared without case); a
        name a manifest wrote with a folder (one published before 0.11.0) is
        its file name.

        Parameters:
            payloads: ``{record key: [decoded payload, ...]}`` -- for a
                delivered file, each record as a reader applies it
                (:meth:`reader_records`). Keys no record declares are
                ignored.

        Returns:
            The file names, folder-less, in record order.
        """
        names: List[str] = []
        seen = set()
        for key, paths in cls.DEPENDENCY_FIELDS.items():
            for payload in payloads.get(key) or ():
                for path in paths:
                    for value in cls._field_values(payload, path):
                        if not isinstance(value, str):
                            continue
                        name = os.path.basename(value.strip().replace("\\", "/"))
                        if name and name.casefold() not in seen:
                            seen.add(name.casefold())
                            names.append(name)
        return names

    @classmethod
    def dependencies(cls, path: str) -> List[str]:
        """The file names the FBX's scene records name (:attr:`DEPENDENCY_FIELDS`).

        Read off the file's ``data_export`` carriers as a reader applies them
        (:meth:`reader_records`: the scene's own record wins, a referenced
        module's names only the nodes the file carries): the lightmaps and
        the reflection probe ``lightmaps`` names, the silhouette,
        atlas and horizon maps ``shadows`` names. In record order,
        each name once (compared without case); a name a manifest wrote with
        a folder (one published before 0.11.0) is its file name.

        Parameters:
            path: The FBX to read.

        Returns:
            The file names, folder-less; empty when no record names one.

        Raises:
            ValueError: *path* is not a binary FBX this writer can read.
        """
        with (
            open(path, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            _version, roots, _fid, _magic = cls._load(buf)
            return cls._dependency_names(roots)

    @classmethod
    def audio_events(cls, path: str) -> List[str]:
        """The tracks the FBX's ``audio`` events play, by event name.

        An audio event names its TRACK, not a file (``SceneRecords.AUDIO``),
        so the host resolves each name to its file and hands it to
        :meth:`embed_dependencies` as ``files=[(name + ext, path)]`` -- the
        name the consumer matches the clip by. Read off the ONE carrier its
        readers take: the scene's own, else the first that carries the
        record (:meth:`reader_records`: the GLB applier's precedence,
        unitytk's ``CarrierImport.Collect``) -- the scene's own record is
        re-derived from the whole merged scene, so a module's or a stray
        ``data_export1``'s events are stale and their clips are not shipped.
        In event order, each name once; a blank or ``"none"`` name names
        nothing.

        Parameters:
            path: The FBX to read.

        Returns:
            The event names; empty when the file carries no audio events.

        Raises:
            ValueError: *path* is not a binary FBX this writer can read.
        """
        with (
            open(path, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            _version, roots, _fid, _magic = cls._load(buf)
            key = SceneRecords.AUDIO.key
            # The carrier precedence alone: an event names a track and its
            # clip, never a node, so there is nothing to narrow it by.
            payload = cls.reader_records(cls._carrier_channels(roots, (key,))).get(key)
        names: List[str] = []
        events = payload.get("events") if isinstance(payload, Mapping) else None
        for event in events if isinstance(events, list) else ():
            if not isinstance(event, Mapping):
                continue
            name = str(event.get("name") or "").strip()
            if name and name.lower() != "none" and name not in names:
                names.append(name)
        return names

    @classmethod
    def embed(
        cls, src: str, dst: Optional[str] = None, *, files: Iterable[Any]
    ) -> Dict[str, Any]:
        """Embed each of *files* in the FBX as media no material uses.

        Each file becomes a ``Video`` object holding its bytes under its FILE
        NAME (no folder: the deliverable names none) and wired to nothing, so
        every material imports as authored. That is the form Unity's FBX
        importer hands back: measured on 2022.3 and 6000.3 (2026-10-05),
        ``ModelImporter.ExtractTextures`` writes a bare ``Video``'s bytes out
        byte for byte under its file name -- EXR, HDR, 8- and 16-bit PNG and
        WAV alike -- while a Texture bound to a known material slot rebinds
        that slot. No form surfaces as a sub-asset of the model; extraction
        is the only way Unity holds embedded media.

        A file name the FBX already embeds (a material's own map, or an
        earlier run) is left, compared without case, so embedding again
        changes nothing.

        Parameters:
            src: The FBX to read.
            dst: Where to write; ``None`` rewrites *src* in place.
            files: The files to embed: paths, or ``(name, path)`` pairs to
                embed a file under another name.

        Returns:
            ``{"embedded", "present", "bytes"}`` -- the file names added and
            the ones already in the file, in *files*' order, and the bytes
            added. When nothing is added the file is not written at all.

        Raises:
            ValueError: *src* is not a binary FBX this writer can re-emit.
            OSError: A file in *files* cannot be read.
        """
        entries = [cls._named(path) for path in files]
        with (
            open(src, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            version, roots, footer_id, magic = cls._load(buf)
            sections = {record.name: record for record in roots}
            hlen = cls._header_fmt(version >= 7500)[1]
            report = cls._add_media(sections, entries, hlen)
            if not report["embedded"]:
                return report
            target = dst or src
            part = cls._write_part(target, version, roots, footer_id, magic)
        cls._promote(part, target)
        return report

    @classmethod
    def record(cls, path: str, key: str) -> Optional[Any]:
        """The scene record *key* the FBX carries, as a reader applies it.

        Read off the file's ``data_export`` carriers: the scene's own, else a
        referenced module's, the first in file order -- the copy
        :meth:`reader_records` takes (whole here, not narrowed to the nodes
        the file carries). ``None`` when no carrier holds one.

        Raises:
            ValueError: *path* is not a binary FBX this reader can parse.
        """
        with (
            open(path, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            _version, roots, _fid, _magic = cls._load(buf)
            found = cls._applied_channel(roots, key)
        return found[1] if found else None

    @classmethod
    def records(
        cls, path: str, keys: Optional[Iterable[str]] = None
    ) -> Dict[str, Any]:
        """Every scene record the FBX carries, as a reader applies it
        (:meth:`record`'s copy of each, in one read of the file).

        Parameters:
            path: The FBX.
            keys: The records to read; every declared deliverable record
                (``SceneRecords.deliverable()``) when ``None``.

        Returns:
            ``{key: payload}`` for each of *keys* a carrier holds.

        Raises:
            ValueError: *path* is not a binary FBX this reader can parse.
        """
        if keys is None:
            keys = [spec.key for spec in SceneRecords.deliverable()]
        with (
            open(path, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            _version, roots, _fid, _magic = cls._load(buf)
            found = cls._applied_channels(roots, keys)
        return {key: hit[1] for key, hit in found.items()}

    @classmethod
    def set_record(
        cls,
        src: str,
        key: str,
        payload: Any,
        dst: Optional[str] = None,
        *,
        files: Iterable[Any] = (),
    ) -> Dict[str, Any]:
        """Replace the scene record *key* the FBX carries, embedding *files* in the same write.

        The copy a reader applies (:meth:`record`) is rewritten, as JSON, in
        the carrier that holds it; *files* go in as :meth:`embed` puts them.
        One pass, so a record that starts naming a file is never written
        without it.

        Parameters:
            src: The FBX to read.
            key: The record (a ``SceneRecords`` key).
            payload: Its new decoded value.
            dst: Where to write; ``None`` rewrites *src* in place.
            files: Files to embed: paths, or ``(name, path)`` pairs.

        Returns:
            ``{"replaced", "embedded", "present", "bytes"}`` -- whether a
            record was replaced (``False``: the file carries no *key* record
            and is left as it was) and :meth:`embed`'s report.

        Raises:
            ValueError: *src* is not a binary FBX this writer can re-emit.
            OSError: A file in *files* cannot be read.
        """
        entries = [cls._named(path) for path in files]
        report: Dict[str, Any] = {
            "replaced": False,
            "embedded": [],
            "present": [],
            "bytes": 0,
        }
        with (
            open(src, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            version, roots, footer_id, magic = cls._load(buf)
            found = cls._applied_channel(roots, key)
            if found is None:
                return report
            cls._replace_prop(
                found[0], 4, cls._pack_string(json.dumps(payload).encode("utf-8"))
            )
            report["replaced"] = True
            sections = {record.name: record for record in roots}
            hlen = cls._header_fmt(version >= 7500)[1]
            report.update(cls._add_media(sections, entries, hlen))
            target = dst or src
            part = cls._write_part(target, version, roots, footer_id, magic)
        cls._promote(part, target)
        return report

    @classmethod
    def split_pivot_instances(
        cls, src: str, dst: Optional[str] = None, *, tolerance: float = 1e-3
    ) -> Dict[str, Any]:
        """Give each pivot set among a geometry's instances a geometry of its own.

        An instanced mesh is one ``Geometry`` under several Models, and an
        engine's importer brings each node's pivots to its origin through the
        mesh: Unity's moves the shared mesh by ONE instance's pivots, so an
        instance whose pivots differ lands off by the difference. Measured on
        a production office (2026-10-19): ``WALL_D``, its rotate pivot
        131.44 cm below its row's, imported exactly 1.3144 m low, onto the
        wall under it -- where the GLB, whose converter places each node
        itself, put it right. Each further set of :attr:`GEOMETRY_PIVOTS`
        (equal within *tolerance*, in the file's units) gets a copy of the
        geometry wired to its Models; instances that agree stay one mesh. A
        geometry a deformer drives (a skin, blend shapes) is left shared: its
        copy would deform nothing.

        Parameters:
            src: The FBX to read.
            dst: Where to write; ``None`` rewrites *src* in place.
            tolerance: How far two pivots may differ and still be one.

        Returns:
            ``{"models", "geometries", "deformed"}`` -- the Models moved onto
            a copy, how many copies were made, and the geometries left shared
            because a deformer drives them (both lists sorted). When nothing
            splits the file is not written at all.

        Raises:
            ValueError: *src* is not a binary FBX this writer can re-emit.
        """
        report: Dict[str, Any] = {"models": [], "geometries": 0, "deformed": []}
        with (
            open(src, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            version, roots, footer_id, magic = cls._load(buf)
            sections = {record.name: record for record in roots}
            objects = sections.get(b"Objects")
            connections = sections.get(b"Connections")
            if objects is None or connections is None:
                return report
            graph = cls._model_graph(objects, connections)
            kinds, models = graph["kinds"], graph["records"]
            geometries = {
                graph["id_of"][id(record)]: record
                for record in objects.children
                if record.name == b"Geometry" and id(record) in graph["id_of"]
            }
            users: Dict[int, List[int]] = {}
            driven = set()
            for child, parent in graph["links"]:
                if child in geometries and parent in models:
                    users.setdefault(child, []).append(parent)
                elif parent in geometries and kinds.get(child) == b"Deformer":
                    driven.add(parent)
            taken = cls._taken_ids(roots, objects)
            moves: Dict[Tuple[int, int], int] = {}  # (geometry, model) -> its copy
            after: Dict[int, List[_Record]] = {}  # source record -> its copies
            for gid, owners in users.items():
                sets: List[Tuple[Tuple[Tuple[float, ...], ...], List[int]]] = []
                for mid in owners:
                    pivots = cls._pivot_set(models[mid])
                    for known, members in sets:
                        if all(
                            abs(a - b) <= tolerance
                            for mine, theirs in zip(pivots, known)
                            for a, b in zip(mine, theirs)
                        ):
                            members.append(mid)
                            break
                    else:
                        sets.append((pivots, [mid]))
                if len(sets) < 2:
                    continue
                source = geometries[gid]
                if gid in driven:
                    head = cls._scalars(source)
                    label = head[1] if len(head) > 1 and isinstance(head[1], bytes) else b""
                    report["deformed"].append(FbxFile._display_name(label) or str(gid))
                    continue
                raw = bytes(source.payload)
                start, end = cls._prop_span(raw, 0)
                for _pivots, members in sets[1:]:
                    oid = cls._media_id(f"pivot-split:{gid}:{members[0]}", taken)
                    taken.add(oid)
                    after.setdefault(id(source), []).append(
                        _Record(
                            source.name,
                            source.prop_count,
                            raw[:start] + cls._pack_long(oid) + raw[end:],
                            source.children,
                            source.tail,
                        )
                    )
                    for mid in members:
                        moves[(gid, mid)] = oid
                        report["models"].append(graph["names"][mid])
            report["deformed"].sort()
            if not moves:
                return report
            objects.children = [
                kept
                for record in objects.children
                for kept in (record, *after.get(id(record), ()))
            ]
            for record, row in graph["rows"]:
                if len(row) >= 3 and row[0] == b"OO" and (row[1], row[2]) in moves:
                    cls._replace_prop(record, 1, cls._pack_long(moves[(row[1], row[2])]))
            report["geometries"] = sum(len(copies) for copies in after.values())
            report["models"].sort()
            cls._shift_definitions(
                sections,
                {b"Geometry": report["geometries"]},
                cls._header_fmt(version >= 7500)[1],
            )
            target = dst or src
            part = cls._write_part(target, version, roots, footer_id, magic)
        cls._promote(part, target)
        return report

    @classmethod
    def drop_redundant_properties(
        cls, src: str, markers: Iterable[str] = (), dst: Optional[str] = None
    ) -> Dict[str, int]:
        """Remove the user properties a delivered FBX carries that state
        nothing its records do not.

        Two kinds, both the export's own: each of *markers* on every model --
        a per-object property a record restates whole (the lightmap bake's
        ``lightmapInfo`` repeats its ``lightmaps`` entry) -- and every
        empty string on a ``data_export`` carrier, a record cleared in the
        scene whose attribute the FBX writer still ships as a channel holding
        nothing. Anything else a model carries -- another program's
        properties, an artist's own attribute -- stays.

        Parameters:
            src: The FBX to read.
            markers: The user-property names to drop from every model.
            dst: Where to write; ``None`` rewrites *src* in place.

        Returns:
            ``{"markers", "empty"}`` -- how many of each went. When none
            qualifies the file is not written at all.

        Raises:
            ValueError: *src* is not a binary FBX this writer can re-emit.
        """
        names = {name.encode("utf-8") for name in markers}
        report = {"markers": 0, "empty": 0}
        with (
            open(src, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            version, roots, footer_id, magic = cls._load(buf)
            for name, model in cls._models(roots):
                carrier = SceneStoreBase.is_carrier(name)
                for child in model.children:
                    if child.name != b"Properties70":
                        continue
                    kept = []
                    for prop in child.children:
                        head = cls._scalars(prop)
                        user = len(head) >= 4 and b"U" in bytes(head[3] or b"")
                        if user and head[0] in names:
                            report["markers"] += 1
                            continue
                        if (
                            user
                            and carrier
                            and len(head) >= 5
                            and head[1] == b"KString"
                            and not head[4]
                        ):
                            report["empty"] += 1
                            continue
                        kept.append(prop)
                    child.children = kept
            if not (report["markers"] or report["empty"]):
                return report
            target = dst or src
            part = cls._write_part(target, version, roots, footer_id, magic)
        cls._promote(part, target)
        return report

    @classmethod
    def drop_host_lighting_media(
        cls, src: str, dst: Optional[str] = None
    ) -> Dict[str, Any]:
        """Remove the textures only :attr:`HOST_LIGHTING_SLOTS` sample, with their media.

        An embedded-media export carries every file a material samples, and
        Maya's StingrayPBS samples Autodesk's own environment cubes and BRDF
        table in every material: a production office's FBX shipped
        ``specular_cube.png``, ``diffuse_cube.png`` and ``ibl_brdf_lut.png``
        once per material, and Unity extracted them beside the model, where
        nothing reads them (2026-10-19; the GLB build already sweeps them,
        :meth:`MeshConvert.prune_glb_unreferenced_textures`). A texture any
        other slot samples stays, and so does a file another ``Video`` or a
        scene record names (the SDK embeds a file's bytes once, in the first
        ``Video`` naming it), its media left wired to nothing. The slots
        themselves stay on the material, sampling nothing.

        Parameters:
            src: The FBX to read.
            dst: Where to write; ``None`` rewrites *src* in place.

        Returns:
            ``{"textures", "media"}`` -- how many textures went and the file
            names of the media that went with them (sorted). When none
            qualifies the file is not written at all.

        Raises:
            ValueError: *src* is not a binary FBX this writer can re-emit.
        """
        report: Dict[str, Any] = {"textures": 0, "media": []}
        with (
            open(src, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            version, roots, footer_id, magic = cls._load(buf)
            sections = {record.name: record for record in roots}
            objects = sections.get(b"Objects")
            connections = sections.get(b"Connections")
            if objects is None or connections is None:
                return report
            graph = cls._model_graph(objects, connections)
            kinds, owners = graph["kinds"], graph["owners"]
            slots: Dict[int, List[Optional[bytes]]] = {}
            for _record, row in graph["rows"]:
                if len(row) >= 3 and kinds.get(row[1]) == b"Texture":
                    slot = row[3] if row[0] == b"OP" and len(row) > 3 else None
                    slots.setdefault(row[1], []).append(slot)
            doomed = {
                texture
                for texture, used in slots.items()
                if all(slot in cls.HOST_LIGHTING_SLOTS for slot in used)
            }
            if not doomed:
                return report
            videos = {
                graph["id_of"][id(record)]: record
                for record in objects.children
                if record.name == b"Video" and id(record) in graph["id_of"]
            }
            # The SDK embeds a file's bytes once, in the first Video naming it:
            # a file a kept Video names (a user's texture on the same file) or
            # a record names (a probe plugged into a slot) may ride here
            # alone, so its Video stays, wired to nothing.
            held = {name.casefold() for name in cls._dependency_names(roots)}
            held |= {
                cls._video_file(record).casefold()
                for oid, record in videos.items()
                if not (owners.get(oid) and owners[oid] <= doomed)
            }
            held.discard("")
            media = {
                oid
                for oid, record in videos.items()
                if owners.get(oid)
                and owners[oid] <= doomed
                and cls._video_file(record).casefold() not in held
            }
            going = {id(child) for oid in media for child in videos[oid].children}
            report["media"] = sorted(
                name
                for name, content in cls._video_records(roots)
                if id(content) in going
            )
            counts = cls._remove_objects(
                sections, graph, doomed | media, cls._header_fmt(version >= 7500)[1]
            )
            report["textures"] = counts.get(b"Texture", 0)
            target = dst or src
            part = cls._write_part(target, version, roots, footer_id, magic)
        cls._promote(part, target)
        return report

    @classmethod
    def embed_dependencies(
        cls,
        src: str,
        dst: Optional[str] = None,
        *,
        search_dirs: Sequence[str] = (),
        walk_root: str = "",
        files: Iterable[Any] = (),
        logger: Any = None,
    ) -> Dict[str, Any]:
        """Embed every file the FBX's scene records name, so nothing ships beside it.

        The post-write step both Scene Exporters and the Unity hand-off run on
        an FBX deliverable. The records name their files by FILE NAME --
        ``lightmaps`` its maps and its reflection probe,
        ``shadows`` its silhouette, atlas and horizon maps
        (:meth:`dependencies`) -- and a lighting-only bake leaves its maps on
        no material, so the FBX used to carry none of them: the consumer
        copied them in by hand, or imported unlit with a silhouette shadow.
        Each name is resolved the way the GLB appliers resolve it: the first
        of *search_dirs* holding that file (the host's own answer to where
        its maps live), then the FBX's own folder, then -- for what those miss
        -- anywhere under *walk_root* (:meth:`FileDependencies.find_files`; a
        unique hit, never a guess between same-named copies), and goes in as
        :meth:`embed` puts a file -- under the record's own spelling of the
        name, the join key. A name the FBX already embeds is left as it is;
        one found nowhere is reported and warned, naming *walk_root* (the
        folders tried go to DEBUG, the run log's detail).

        Parameters:
            src: The FBX to read.
            dst: Where to write; ``None`` rewrites *src* in place.
            search_dirs: Folders to resolve the names against, in priority
                order (a DCC's ``LightmapRecords.search_dirs``).
            walk_root: The folder a name those miss is searched for under,
                recursively -- the project's texture root; ``""`` for none.
            files: Further files to embed, for a host whose records name
                their files by something other than a file name: a path (it
                goes in under its file name) or a ``(name, path)`` pair -- an
                audio event names its track, so the track's file rides under
                the name the consumer matches it by.
            logger: Where the outcome goes; this module's logger otherwise.

        Returns:
            ``{"named", "embedded", "present", "missing", "bytes"}`` -- the
            names the records give, the file names added, the ones already
            in the file, the named files found nowhere, and the bytes added.
            When nothing is added the file is not written at all.

        Raises:
            ValueError: *src* is not a binary FBX this writer can re-emit.
            OSError: A file found or named cannot be read.
        """
        log = logger or logging.getLogger(__name__)
        report: Dict[str, Any] = {
            "named": [],
            "embedded": [],
            "present": [],
            "missing": [],
            "bytes": 0,
        }
        dirs = [d for d in search_dirs if d]
        dirs.append(os.path.dirname(os.path.abspath(src)))
        walk_root = walk_root if walk_root and os.path.isdir(walk_root) else ""
        with (
            open(src, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            version, roots, footer_id, magic = cls._load(buf)
            sections = {record.name: record for record in roots}
            report["named"] = cls._dependency_names(roots)
            inside = {name.casefold() for name, _r in cls._video_records(roots)}
            found: List[Tuple[str, str]] = []
            for name in report["named"]:
                if name.casefold() in inside:
                    report["present"].append(name)
                    continue
                path = next(
                    (
                        p
                        for d in dirs
                        for p in [os.path.join(d, name)]
                        if os.path.isfile(p)
                    ),
                    None,
                )
                if path is None:
                    report["missing"].append(name)
                else:
                    found.append((name, path))
            twins: Dict[str, int] = {}
            if report["missing"] and walk_root:
                hits: Dict[str, List[str]] = {}
                for hit in FileDependencies.find_files(report["missing"], walk_root):
                    hits.setdefault(os.path.basename(hit).casefold(), []).append(hit)
                for name in list(report["missing"]):
                    paths = hits.get(name.casefold()) or []
                    if len(paths) == 1:
                        report["missing"].remove(name)
                        found.append((name, paths[0]))
                    elif paths:
                        twins[name] = len(paths)
            hlen = cls._header_fmt(version >= 7500)[1]
            extra = [cls._named(path) for path in files]
            added = cls._add_media(sections, [*found, *extra], hlen)
            report["embedded"] = added["embedded"]
            report["bytes"] = added["bytes"]
            report["present"] += [
                n for n in added["present"] if n not in report["present"]
            ]
            if report["embedded"]:
                target = dst or src
                part = cls._write_part(target, version, roots, footer_id, magic)
        if report["embedded"]:
            cls._promote(part, target)
            log.info(
                "FBX dependencies: embedded %d file(s), %.1f MB: %s.",
                len(report["embedded"]),
                report["bytes"] / 1e6,
                ", ".join(report["embedded"]),
            )
        if report["missing"]:
            log.debug(
                "FBX dependencies: looked in %s%s.",
                ", ".join(dirs),
                f", then under {walk_root}" if walk_root else "",
            )
            where = (
                f"{walk_root} or any of its subfolders"
                if walk_root
                else f"the {len(dirs)} folder(s) searched"
            )
            names = [
                f"{n} ({twins[n]} same-named files, none guessed)" if n in twins else n
                for n in report["missing"]
            ]
            log.warning(
                "FBX dependencies: not found in %s, so the FBX ships without: %s.",
                where,
                ", ".join(names),
            )
        return report

    @classmethod
    def extract_dependencies(cls, src: str, dst_dir: str) -> Dict[str, str]:
        """Write each file the FBX's records name AND embeds into *dst_dir*.

        The reverse of :meth:`embed_dependencies`, for a reader of a DELIVERED
        FBX that joins the records' names against folders -- the GLB build's
        lightmap and shadow appliers. Each file goes out byte for byte under
        the record's spelling of its name; a name the FBX does not embed is
        left out, and so is media no record names (a material's own map is
        the converter's to read).

        Parameters:
            src: The FBX to read.
            dst_dir: An existing folder; same-named files there are replaced.

        Returns:
            ``{file name: path written}``, in record order; empty when the
            records name nothing the file embeds.

        Raises:
            ValueError: *src* is not a binary FBX this writer can read.
            OSError: A file could not be written.
        """
        return cls.extract(src, dst_dir)

    @classmethod
    def extract(
        cls, src: str, dst_dir: str, names: Optional[Iterable[str]] = None
    ) -> Dict[str, str]:
        """Write each of *names* the FBX embeds into *dst_dir*, byte for byte.

        The general form of :meth:`extract_dependencies` (which is this with
        *names* ``None``: the files the records name): a reader that joins
        names against folders -- the GLB build binding a materials record's
        maps -- takes the deliverable's own bytes rather than whatever a disk
        holds under the name. Each file goes out under the spelling asked for,
        matched without case; a name the FBX does not embed is left out.

        Parameters:
            src: The FBX to read.
            dst_dir: An existing folder; same-named files there are replaced.
            names: The file names to write; ``None`` takes the records'.

        Returns:
            ``{file name: path written}``, in *names*' order.

        Raises:
            ValueError: *src* is not a binary FBX this writer can read.
            OSError: A file could not be written.
        """
        written: Dict[str, str] = {}
        with (
            open(src, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            _version, roots, _fid, _magic = cls._load(buf)
            wanted = cls._dependency_names(roots) if names is None else list(names)
            if not wanted:
                return written
            media: Dict[str, _Record] = {}
            for name, record in cls._video_records(roots):
                media.setdefault(name.casefold(), record)
            for name in wanted:
                record = media.get(str(name).casefold())
                if record is None or name in written:
                    continue
                path = os.path.join(dst_dir, name)
                with open(path, "wb") as out:
                    out.write(cls._raw(record))
                written[name] = path
        return written

    @classmethod
    def drop_takes(
        cls,
        src: str,
        dst: Optional[str] = None,
        *,
        names: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Remove the named animation takes, and everything only they own.

        Each named ``AnimationStack`` goes with every layer, curve node and
        curve reachable ONLY through dropped owners (an object another take
        still reaches is kept), every connection naming a removed object, its
        ``Takes`` entry, and the removed objects' ``Definitions`` counts. A
        ``Takes/Current`` that named a dropped take points at the first take
        left. Geometry, materials, media and the surviving takes are copied
        byte for byte.

        Why it exists: FBX2glTF bakes EVERY node at EVERY frame of EVERY take.
        A Maya take split writes each shot as its own stack beside the
        whole-timeline one, and the GLB's clips are then cut from that
        whole-timeline stack -- so the converter was baking the entire
        performance a second time, shot by shot, for animations the clip
        rebuild throws away. Measured on a production assembly (18 shots, 3723
        exported nodes): 1377 s -> 537 s of conversion, 491 MB -> 385 MB of
        converter input.

        Parameters:
            src: The FBX to read.
            dst: Where to write; ``None`` rewrites *src* in place.
            names: Take (``AnimationStack`` display) names to drop. Names the
                file does not carry are ignored. ``None`` drops every take:
                the scene at rest, for a build that reads only that
                (:meth:`MeshConvert.embed_fbx_highlight_probes`).

        Returns:
            ``{"takes", "objects", "connections"}`` -- the take names dropped
            (file order), ``{record name: count}`` removed, and the number of
            connections removed. When no named take is present the file is not
            written at all.

        Raises:
            ValueError: *src* is not a binary FBX this writer can re-emit.
        """
        wanted = None if names is None else {str(name) for name in names}
        report: Dict[str, Any] = {"takes": [], "objects": {}, "connections": 0}
        with (
            open(src, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            version, roots, footer_id, magic = cls._load(buf)
            sections = {record.name: record for record in roots}
            if sections.get(b"Objects") is None:
                return report
            stacks = {
                uid: name
                for uid, name in cls._stacks(sections).items()
                if wanted is None or name in wanted
            }
            if not stacks:
                return report
            report = cls._drop_stacks(sections, stacks)
            target = dst or src
            part = cls._write_part(target, version, roots, footer_id, magic)
        cls._promote(part, target)
        return report

    #: The name :meth:`merge_takes` gives the one take it leaves: the take an
    #: unsplit export carries, so an importer meets the plain one-take file
    #: the merge has made of it (Blender names each action ``<object>|Take
    #: 001``, as for any Maya file).
    MERGED_TAKE = "Take 001"

    @classmethod
    def merge_takes(
        cls,
        src: str,
        dst: Optional[str] = None,
        *,
        takes: Optional[Iterable[str]] = None,
        name: Optional[str] = MERGED_TAKE,
        tolerance: float = 1e-4,
        keep: Iterable[str] = (),
    ) -> Dict[str, Any]:
        """Rewrite the file's animation as ONE take: the union of *takes*.

        The import half of :meth:`FbxFile.take_layout`. No DCC merges takes on
        import, and none can be made to: Maya plays the one take
        ``FBXImportSetTake`` names (sequential merge imports REPLACE a curve,
        and a per-take namespace import pasted back lost 63 of 1735 channels
        on a production file), Blender assigns each object the first stack's
        action. So the file is made a one-take file first, and imported once.

        Every key of every merged take stays at its stored time with its
        stored value, flags and slopes (a computed tangent beside a boundary
        aside, below); a time two takes both key (a gap-0
        seam) is kept once. Each channel lands on the kept take's curve -- the
        file's current take when it is merged, else the first merged take --
        and a channel, curve node or layer only a later take has is re-wired
        onto the kept take's layer (curve node, stack). The kept stack's
        ``LocalStart`` / ``LocalStop`` (and ``Reference*``) and its ``Takes``
        entry widen to the union, it becomes ``Takes/Current``, and every
        other stack goes with everything only it owns (:meth:`drop_takes`).
        Geometry, materials and media are copied byte for byte. Between two
        takes' keys -- frames no take keys -- the curve interpolates as the
        earlier take's last key says. A key whose tangent the importer
        computes from its neighbours (Maya's spline, clamped, auto) and whose
        neighbours the merge changed -- a take's end key, the next take's
        first beside it -- is written as a user tangent carrying the slopes
        it plays in its own take (broken at a seam whose sides differ), so
        each take plays as it does alone, between frames too.

        A single-take or covered file needs no merge (``take_layout`` says
        so); merging one take only renames it -- its layer too, where the
        layer bears the take's name (Blender names each layer after its take,
        and its importer names an action ``<object>|<take>|<layer>`` once the
        two differ).

        *keep* leaves takes in the file whole, each its own stack beside the
        merged one: the ``stash`` of a layout, so ONE import carries both what
        plays and the alternatives an importer parks (Blender's makes an
        action per stack and object -- a second import of the same objects
        could only duplicate them).

        Parameters:
            src: The FBX to read.
            dst: Where to write; ``None`` rewrites *src* in place.
            takes: Take names to merge, in any order; ``None`` merges the
                layout's ``live`` takes.
            name: The merged take's name; ``None`` keeps the kept take's own.
            tolerance: As :meth:`FbxFile.take_layout` judges agreement.
            keep: Take names left as they are -- not merged, not dropped. None
                of them may be merged, be named *name*, or share an animation
                object (curve, curve node, layer) with a merged take.

        Returns:
            ``{"path", "take", "takes", "dropped", "channels", "keys",
            "seams", "rewired", "objects", "connections"}`` -- the file
            written, the merged take's name, the takes merged and the takes
            left out (file order; every stack but the kept one and *keep*'s
            goes), the channels written, their keys in all,
            the keys a shared time collapsed, the connections added to re-wire
            what only a dropped take held, ``{record name: count}`` removed
            and the connections removed. With nothing to merge the file is not
            written at all (``path`` ``None``).

        Raises:
            ValueError: *src* is not a binary FBX this writer can re-emit (a
                cut or corrupt record included), a named take is not in it,
                two of *takes* disagree -- the message names both takes, the
                channel and the frame -- or a *keep* take is merged, named
                *name* or shares an animation object with a merged take.
            OSError: The merged file could not be written or put in place.
        """
        report: Dict[str, Any] = {
            "path": None,
            "take": None,
            "takes": [],
            "dropped": [],
            "channels": 0,
            "keys": 0,
            "seams": 0,
            "rewired": 0,
            "objects": {},
            "connections": 0,
        }
        with (
            open(src, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            version, roots, footer_id, magic = cls._load(buf)
            sections = {record.name: record for record in roots}
            objects = sections.get(b"Objects")
            if objects is None:
                return report
            connections = sections.get(b"Connections")
            if connections is None:  # nothing is wired, so nothing is keyed
                return report
            hlen = cls._header_fmt(version >= 7500)[1]
            model = FbxFile._build_take_model(*cls._take_rows(objects, connections))
            current, fps = cls._takes_settings(sections)
            known = [take["name"] for take in model["takes"]]
            parked = [str(take) for take in keep]
            if takes is None:
                wanted = FbxFile._layout(model, current, tolerance, fps)["live"]
                named = list(parked)
            else:
                wanted = [str(take) for take in takes]
                named = wanted + parked
            missing = [take for take in named if take not in known]
            if missing:
                raise ValueError(
                    f"FBX {src!r} has no take {', '.join(map(repr, missing))}; "
                    f"it has {', '.join(map(repr, known)) or 'none'}"
                )
            both = [take for take in parked if take in wanted]
            if both:
                raise ValueError(
                    f"FBX takes {', '.join(map(repr, both))} cannot be both merged "
                    "and kept"
                )
            chosen = [take for take in model["takes"] if take["name"] in wanted]
            if not chosen:
                return report
            for i, take in enumerate(chosen[1:], 1):
                reason = FbxFile._first_conflict(
                    take, chosen[:i], model["labels"], tolerance, fps
                )
                if reason is not None:
                    raise ValueError(
                        f"FBX takes cannot merge: {reason} -- merge only takes "
                        "that agree (FbxFile.take_layout()['live'])"
                    )
            kept = next((t for t in chosen if t["name"] == current), chosen[0])
            final = name or kept["name"]
            if final in parked:
                raise ValueError(
                    f"FBX merged take name {final!r} is a kept take's; pass another "
                    "name"
                )

            def owned(take: Mapping[str, Any]) -> set:
                """The animation objects *take* reaches: layers, nodes, curves."""
                found = set(take["layers"]) | set(take["nodes"].values())
                for channel in take["channels"].values():
                    found.update((channel["node"], channel["id"]))
                return found

            held = [t for t in model["takes"] if t["name"] in parked]
            if held:
                # A kept take sharing an object with a merged one would be
                # rewritten by the merge (a curve's keys, a node's components).
                merging = set().union(*(owned(t) for t in chosen))
                for take in held:
                    if owned(take) & merging:
                        raise ValueError(
                            f"FBX take {take['name']!r} shares animation objects "
                            "with a merged take; it cannot be kept beside the merge"
                        )

            records: Dict[int, _Record] = {}
            for record in objects.children:
                head = cls._scalars(record)
                if head and isinstance(head[0], int):
                    records[head[0]] = record
            components: Dict[int, Dict[str, int]] = {}
            for take in model["takes"]:
                for key, channel in take["channels"].items():
                    components.setdefault(channel["node"], {}).setdefault(
                        key[2], channel["id"]
                    )
            layers = list(kept["layers"])
            nodes = dict(kept["nodes"])

            def wire(child: int, parent: int, prop: Optional[str] = None) -> None:
                connections.children.append(cls._connection(child, parent, prop, hlen))
                report["rewired"] += 1

            order: Dict[Tuple[Any, ...], None] = {}
            for take in chosen:
                order.update(dict.fromkeys(take["channels"]))
            first = last = None
            for key in order:
                target, prop, component, index = key
                sources = [take for take in chosen if key in take["channels"]]
                while len(layers) <= index:  # a layer the kept take lacks
                    donor = next(t for t in chosen if len(t["layers"]) > len(layers))
                    layer = donor["layers"][len(layers)]
                    wire(layer, kept["stack"])
                    layers.append(layer)
                    for (t, p, i), node in donor["nodes"].items():
                        if i == len(layers) - 1:
                            nodes.setdefault((t, p, i), node)
                node = nodes.get((target, prop, index))
                if node is None:  # a property the kept take does not animate
                    node = sources[0]["channels"][key]["node"]
                    wire(node, layers[index])
                    nodes[(target, prop, index)] = node
                curve_id = components.setdefault(node, {}).get(component)
                if curve_id is None:  # a component the kept node lacks
                    donor = sources[0]["channels"][key]
                    curve_id = donor["id"]
                    wire(curve_id, node, component)
                    components[node][component] = curve_id
                    cls._declare_component(
                        records[node], records.get(donor["node"]), component, hlen
                    )
                curves = [take["channels"][key]["curve"] for take in sources]
                times, values, *runs, seams = FbxFile._union_curves(curves)
                report["channels"] += 1
                report["keys"] += len(times)
                report["seams"] += seams
                if times:
                    first = times[0] if first is None else min(first, times[0])
                    last = times[-1] if last is None else max(last, times[-1])
                stored = next(
                    (
                        t["channels"][key]["curve"]
                        for t in sources
                        if t["channels"][key]["id"] == curve_id
                    ),
                    None,
                )
                unchanged = stored is not None and (
                    stored.times,
                    stored.values,
                    stored.flags,
                    stored.data,
                    [
                        b - a
                        for a, b in zip(stored.starts, stored.starts[1:] + [len(times)])
                    ],
                ) == (times, values, *runs)
                if not unchanged:
                    cls._write_curve(records[curve_id], times, values, *runs, hlen)

            # The union's span: every merged stack's declared times, every key.
            span = [
                t for take in chosen for t in cls._stack_times(records[take["stack"]])
            ]
            span += [t for t in (first, last) if t is not None]
            if span:
                lo, hi = min(span), max(span)
                cls._set_times(
                    records[kept["stack"]],
                    {
                        b"LocalStart": lo,
                        b"LocalStop": hi,
                        b"ReferenceStart": lo,
                        b"ReferenceStop": hi,
                    },
                    hlen,
                )

            stacks = {
                take["stack"]: take["name"]
                for take in model["takes"]
                if take is not kept and take["name"] not in parked
            }
            dropped = cls._drop_stacks(sections, stacks) if stacks else {}
            cls._rename_take(
                sections, records[kept["stack"]], kept["name"], final, span, hlen
            )
            for layer in kept["layers"]:  # a layer named after its take (Blender)
                if layer in records:
                    cls._rename_object(
                        records[layer], kept["name"], final, b"AnimLayer"
                    )

            report.update(
                take=final,
                takes=[take["name"] for take in chosen],
                dropped=[
                    t["name"]
                    for t in model["takes"]
                    if t not in chosen and t["name"] not in parked
                ],
                objects=dropped.get("objects", {}),
                connections=dropped.get("connections", 0),
            )
            target = dst or src
            report["path"] = target
            part = cls._write_part(target, version, roots, footer_id, magic)
        cls._promote(part, target)
        return report

    #: Model classes that draw nothing and bind nothing by themselves -- a
    #: group or locator (``Null``), a joint (``LimbNode``, and the older
    #: ``Limb`` / ``Root``) and a curve (``Line`` is how Maya writes a NURBS
    #: curve). Only these may go: a mesh, a camera, a light -- or a class this
    #: writer does not know -- is content, and keeps every ancestor holding it.
    INERT_MODEL_CLASSES = (
        b"Null",
        b"LimbNode",
        b"Limb",
        b"Root",
        b"Line",
        b"NurbsCurve",
    )
    #: What a removed Model takes with it once every one of their owners has
    #: gone: its attribute, its curve geometry with that geometry's deformers,
    #: and the animation driving any of them. No other object type is ever
    #: removed, and a Model wired to one is kept.
    OWNED_OBJECTS = (
        b"NodeAttribute",
        b"Geometry",
        b"Deformer",
        b"AnimationCurveNode",
        b"AnimationCurve",
    )

    @classmethod
    def drop_apparatus(
        cls,
        src: str,
        dst: Optional[str] = None,
        *,
        section: Mapping[str, str],
        separator: str = "|",
    ) -> Dict[str, Any]:
        """Remove the rig apparatus *section* names, and everything only it owns.

        The consumer half of :class:`~pythontk.RigMachinery`, run on the file
        instead of on a scene: *section* is what the producer's census named
        (``RigMachinery.classify``, made unambiguous), matched against the
        Models by leaf name, and :meth:`RigMachinery.select` decides with THIS
        file's own facts as the net -- a named node whose subtree holds a mesh
        or other content, an influence of a skin that survives, a carrier of
        in-band data (a scene record, a curve proxy) or an object this writer
        does not own is refused, by name. Each Model that goes takes its whole
        subtree, its attribute, its curve geometry and that geometry's
        deformers, and every animation curve node and curve only it drove;
        every connection naming a removed object, its bind-pose entries (and
        a bind pose they leave empty) and the removed objects' ``Definitions``
        counts go with it. Everything else is copied byte for byte.

        Why it exists: a carrier ships every node of a scene, so a baked rig
        arrives twice -- its motion on the joints, and the controls, IK and
        helper groups that used to make it, each still animated. FBX2glTF bakes
        every node at every frame, so they cost conversion time as well as
        bytes. Measured on a production assembly: ~600 of ~2480 nodes, and half
        its animation data, drove nothing a skin references and held no mesh.

        Parameters:
            src: The FBX to read.
            dst: Where to write; ``None`` rewrites *src* in place.
            section: ``{producer path: kind}`` -- the apparatus, as
                ``RigMachinery.classify`` / ``unambiguous`` returned it.
            separator: The hierarchy separator of *section*'s paths.

        Returns:
            ``{"models", "kinds", "refused", "objects", "connections"}`` -- how
            many Models were removed, ``{kind: count}`` over them, the named
            nodes kept (sorted), ``{record name: count}`` of every object
            removed (Models included) and how many connections went. When
            nothing qualifies the file is not written at all.

        Raises:
            ValueError: *src* is not a binary FBX this writer can re-emit.
        """
        from pythontk.core_utils.engines.rig_graph.rig_machinery import (
            RigMachinery,
        )

        report: Dict[str, Any] = {
            "models": 0,
            "kinds": {},
            "refused": [],
            "objects": {},
            "connections": 0,
        }
        if not section:
            return report
        with (
            open(src, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            version, roots, footer_id, magic = cls._load(buf)
            sections = {record.name: record for record in roots}
            objects = sections.get(b"Objects")
            if objects is None:
                return report
            connections = sections.get(b"Connections")
            graph = cls._model_graph(objects, connections)
            doomed, refused = RigMachinery.select(
                section,
                cls._families(graph, section, separator),
                protected=cls._load_bearing(graph, section, separator),
                separator=separator,
            )
            names = graph["names"]
            children = graph["children"]
            skinned = cls._skinned(graph)
            dropped = {oid for oid, name in names.items() if name in doomed}
            # A Model only goes with its whole subtree -- one that left a
            # survivor behind would re-root it at the scene origin -- and an
            # influence only with every Model its skin deforms, or that skin's
            # cluster would dangle. What stays re-checks its own parent on the
            # next pass.
            kept = set()
            while True:
                keep = {
                    oid
                    for oid in dropped
                    if any(child not in dropped for child in children.get(oid, ()))
                    or any(
                        not graph["owners"].get(g, set()) <= dropped
                        for g in skinned.get(oid, ())
                    )
                }
                if not keep:
                    break
                dropped -= keep
                kept |= keep
            report["refused"] = sorted(set(refused) | {names[oid] for oid in kept})
            if not dropped:
                return report

            removed = cls._owned_closure(graph, dropped)
            before = len(connections.children) if connections is not None else 0
            # A bind pose whose every member went describes nothing, and
            # Maya's importer, handed one, binds the skins that stayed at the
            # wrong pose: 6 of 7 production wire looms 97-327 cm off, exact
            # once the 7 emptied poses were gone.
            counts = cls._remove_objects(
                sections,
                graph,
                removed,
                cls._header_fmt(version >= 7500)[1],
                also=lambda record: record.name == b"Pose"
                and cls._drop_pose_nodes(record, removed),
            )
            if connections is not None:
                report["connections"] = before - len(connections.children)

            report["models"] = len(dropped)
            report["kinds"] = RigMachinery.tally(
                {oid: doomed[names[oid]] for oid in dropped}
            )
            report["objects"] = {k.decode(): v for k, v in counts.items()}
            target = dst or src
            part = cls._write_part(target, version, roots, footer_id, magic)
        cls._promote(part, target)
        return report

    @classmethod
    def _model_graph(
        cls, objects: _Record, connections: Optional[_Record]
    ) -> Dict[str, Any]:
        """The file's object table and Model hierarchy, as plain lookups.

        ``kinds`` (id -> record name), ``classes`` (id -> the object's class,
        ``b"Mesh"`` / ``b"Cluster"`` ...), ``id_of`` (record identity -> id),
        ``names`` / ``records`` for Models, ``parent`` / ``children`` of the
        Model hierarchy, ``owners`` (id -> every object it is wired UNDER),
        ``links`` (every connection as ``(child, parent)``) and ``rows`` (each
        connection record with its decoded head).
        """
        kinds: Dict[int, bytes] = {}
        classes: Dict[int, bytes] = {}
        id_of: Dict[int, int] = {}
        names: Dict[int, str] = {}
        records: Dict[int, _Record] = {}
        for record in objects.children:
            head = cls._scalars(record)
            if not head or not isinstance(head[0], int):
                continue
            oid = head[0]
            kinds[oid] = record.name
            id_of[id(record)] = oid
            if len(head) > 2 and isinstance(head[2], bytes):
                classes[oid] = head[2]
            if record.name == b"Model":
                label = head[1] if len(head) > 1 and isinstance(head[1], bytes) else b""
                names[oid] = FbxFile._display_name(label) or ""
                records[oid] = record
        rows = [
            (record, cls._scalars(record))
            for record in (connections.children if connections else [])
        ]
        links: List[Tuple[int, int]] = []
        parent: Dict[int, int] = {}
        children: Dict[int, List[int]] = {}
        owners: Dict[int, set] = {}
        for _record, row in rows:
            if (
                len(row) < 3
                or not isinstance(row[1], int)
                or not isinstance(row[2], int)
            ):
                continue
            links.append((row[1], row[2]))
            owners.setdefault(row[1], set()).add(row[2])
            if row[1] in names and row[2] in names:
                parent[row[1]] = row[2]
                children.setdefault(row[2], []).append(row[1])
        return {
            "kinds": kinds,
            "classes": classes,
            "id_of": id_of,
            "names": names,
            "records": records,
            "parent": parent,
            "children": children,
            "owners": owners,
            "links": links,
            "rows": rows,
        }

    @staticmethod
    def _skinned(graph: Dict[str, Any]) -> Dict[int, List[int]]:
        """``{influence Model: [geometry its skins deform]}``.

        A skin binds through its sub-deformers: each ``Cluster`` is wired
        under its ``Skin``, the Skin under the geometry it deforms, and the
        influence Model under the Cluster.
        """
        kinds, classes, owners = graph["kinds"], graph["classes"], graph["owners"]
        skinned: Dict[int, List[int]] = {}
        for child, parent in graph["links"]:
            if child not in graph["names"] or classes.get(parent) != b"Cluster":
                continue
            for skin in owners.get(parent, ()):
                if kinds.get(skin) == b"Deformer":
                    skinned.setdefault(child, []).extend(owners.get(skin, ()))
        return skinned

    @staticmethod
    def _families(
        graph: Dict[str, Any], section: Mapping[str, str], separator: str
    ) -> Dict[str, List[str]]:
        """``{name: [that Model and its descendants' names]}`` for every Model
        *section* could name -- the ``subtrees`` :meth:`RigMachinery.select`
        judges. A name several Models share collects all their subtrees, so a
        refusal of one is a refusal of the name."""
        leaves = {str(path).rsplit(separator, 1)[-1] for path in section}
        names, children = graph["names"], graph["children"]
        families: Dict[str, List[str]] = {}
        for oid, name in names.items():
            if name not in leaves and name.rsplit(".", 1)[0] not in leaves:
                continue
            family = families.setdefault(name, [])
            stack = [oid]
            while stack:
                node = stack.pop()
                family.append(names[node])
                stack.extend(children.get(node, ()))
        return families

    @classmethod
    def _load_bearing(
        cls, graph: Dict[str, Any], section: Mapping[str, str], separator: str
    ) -> set:
        """The Model names this file says must survive, whatever *section* says.

        Content (a Model class outside :attr:`INERT_MODEL_CLASSES`); every
        influence of a skin that stays -- one deforming content geometry, or a
        curve whose own Model is not being dropped (its sub-deformer would
        dangle); a carrier of in-band data (a scene-record channel, the curve
        proxy marker), whose loss would silently cost the deliverable its
        records; and a Model wired to an object this writer does not own.
        """
        from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
        from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert

        kinds, classes, names = graph["kinds"], graph["classes"], graph["names"]
        owners = graph["owners"]
        leaves = {str(path).rsplit(separator, 1)[-1] for path in section}
        channels = {spec.key.encode("utf-8") for spec in SceneRecords.all()}
        channels.add(MeshConvert.CURVE_PROXY_MARKER.encode("utf-8"))
        partners = set(cls.OWNED_OBJECTS) | {b"Model"}

        protected = set()
        for oid, name in names.items():
            if classes.get(oid, b"") not in cls.INERT_MODEL_CLASSES:
                protected.add(name)
            elif cls._user_properties(graph["records"][oid]) & channels:
                protected.add(name)
        for child, parent in graph["links"]:
            for mine, other in ((child, parent), (parent, child)):
                if mine in names and other and kinds.get(other) not in partners:
                    protected.add(names[mine])

        def survives(geometry: int) -> bool:
            if classes.get(geometry) not in cls.INERT_MODEL_CLASSES:
                return True  # a mesh (or anything but a curve) is content
            return any(
                names[model] not in leaves
                for model in owners.get(geometry, ())
                if model in names
            )

        for influence, geometry in cls._skinned(graph).items():
            if any(survives(g) for g in geometry):
                protected.add(names[influence])
        return protected

    @classmethod
    def _owned_closure(cls, graph: Dict[str, Any], models: set) -> set:
        """*models* plus every object only they (and what goes with them) own.

        Ownership follows the wiring UP: an attribute, geometry or curve node
        is owned by the object it is connected under, a deformer by its
        geometry, a curve by its curve node. A curve node's animation layer is
        not an owner -- it holds curve nodes of every object -- so a curve node
        goes once everything it animates has.
        """
        kinds = graph["kinds"]
        owners: Dict[int, set] = {}
        for child, parent in graph["links"]:
            if not parent or kinds.get(child) not in cls.OWNED_OBJECTS:
                continue
            if kinds.get(parent) == b"AnimationLayer":
                continue
            owners.setdefault(child, set()).add(parent)
        removed = set(models)
        grew = True
        while grew:
            grew = False
            for oid, parents in owners.items():
                if oid not in removed and parents <= removed:
                    removed.add(oid)
                    grew = True
        return removed

    @classmethod
    def _user_properties(cls, record: _Record) -> set:
        """The names of *record*'s user-flagged ``Properties70`` entries."""
        found = set()
        for child in record.children:
            if child.name != b"Properties70":
                continue
            for prop in child.children:
                head = cls._scalars(prop)
                if len(head) >= 4 and isinstance(head[3], bytes) and b"U" in head[3]:
                    found.add(head[0])
        return found

    @classmethod
    def _drop_pose_nodes(cls, pose: _Record, removed: set) -> bool:
        """Remove *pose*'s entries for *removed* nodes and recount it; True
        when that took every entry it had (a pose already empty is not this
        drop's to remove)."""
        entries = [child for child in pose.children if child.name == b"PoseNode"]
        gone = []
        for entry in entries:
            for child in entry.children:
                if child.name != b"Node":
                    continue
                head = cls._scalars(child)
                if head and head[0] in removed:
                    gone.append(entry)
                break
        if not gone:
            return False
        pose.children = [child for child in pose.children if child not in gone]
        cls._shift_count(pose, -len(gone), b"NbPoseNodes")
        return len(gone) == len(entries)

    @classmethod
    def rewrite(cls, src: str, dst: str) -> None:
        """Re-serialise *src* to *dst* unchanged -- the writer's own round trip.

        With nothing replaced every offset lands where it was, so the output
        is byte-identical to the input; that is the invariant a test pins,
        and what makes any later difference attributable to the edit alone.
        """
        with (
            open(src, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            version, roots, footer_id, magic = cls._load(buf)
            with open(dst, "wb") as out:
                cls._write(out, version, roots, footer_id, magic)
