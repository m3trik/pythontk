# !/usr/bin/python
# coding=utf-8
"""Rewrite the payload of a binary FBX -- no DCC, no FBX SDK.

:class:`FbxMedia` is the writer :mod:`fbx_file` deliberately is not, scoped to
the edits a hand-off pipeline needs on its scratch copy: replacing the
``Video`` objects' ``Content`` payloads -- the textures
``FBXExportEmbeddedTextures`` copies into the file at full authoring
resolution (:meth:`FbxMedia.downsize`), and the grayscale ones FBX2glTF would
pack as white (:meth:`FbxMedia.expand_grayscale`) -- dropping animation
takes the converter would bake for nothing (:meth:`FbxMedia.drop_takes`), and
dropping the apparatus of a baked rig (:meth:`FbxMedia.drop_apparatus`).
Everything else is copied byte for byte. Only the record headers are
re-serialised, because binary FBX stores
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

import io
import logging
import mmap
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from pythontk.file_utils._file_utils import FileUtils
from pythontk.file_utils.mesh_convert.fbx_file import FBX_MAGIC

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

    @staticmethod
    def _header_fmt(wide: bool) -> Tuple[str, int]:
        return ("<QQQ", 24) if wide else ("<III", 12)

    @classmethod
    def _parse(cls, buf, pos: int, wide: bool) -> Tuple[Optional[_Record], int]:
        """Parse the record at *pos*; ``(None, end)`` for a NULL record."""
        fmt, hlen = cls._header_fmt(wide)
        end, count, plen = struct.unpack_from(fmt, buf, pos)
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
            ValueError: Not a binary FBX, or a footer this writer cannot
                re-emit faithfully (the caller then leaves the file alone).
        """
        if bytes(buf[: len(FBX_MAGIC)]) != FBX_MAGIC:
            raise ValueError("not a binary FBX")
        version = struct.unpack_from("<I", buf, len(FBX_MAGIC))[0]
        wide = version >= 7500
        pos = len(FBX_MAGIC) + 4
        roots: List[_Record] = []
        while True:
            record, pos = cls._parse(buf, pos, wide)
            if record is None:
                break
            roots.append(record)
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

    @staticmethod
    def _video_records(roots: Iterable[_Record]) -> List[Tuple[str, _Record]]:
        """``(basename, Content record)`` for every embedded ``Video`` object."""
        found: List[Tuple[str, _Record]] = []
        for root in roots:
            if root.name != b"Objects":
                continue
            for obj in root.children:
                if obj.name != b"Video":
                    continue
                filename, content = "", None
                for child in obj.children:
                    if child.name == b"Filename" and len(child.props) > 5:
                        filename = bytes(child.props[5:]).decode("utf-8", "replace")
                    elif child.name == b"Content" and child.prop_count:
                        content = child
                if content is not None and bytes(content.props[:1]) == b"R":
                    found.append(
                        (os.path.basename(filename.replace("\\", "/")), content)
                    )
        return found

    @staticmethod
    def _raw(record: _Record) -> bytes:
        """The bytes of a single ``R`` property."""
        return bytes(record.props[5:])

    @staticmethod
    def _pack_raw(data: bytes) -> bytes:
        return b"R" + struct.pack("<I", len(data)) + data

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
            videos = cls._video_records(roots)
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
        os.replace(part, target)
        return report


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
        GLB afterwards -- longest edge to *max_size*, aspect kept, Lanczos --
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
        from PIL import Image

        exempt = {os.path.basename(str(name)) for name in exempt}

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
            return image.resize(target, Image.LANCZOS)

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

    #: The objects an animation take owns, owner first: a stack holds layers,
    #: a layer holds curve nodes, a curve node holds curves.
    TAKE_CHAIN = (
        b"AnimationStack",
        b"AnimationLayer",
        b"AnimationCurveNode",
        b"AnimationCurve",
    )

    @classmethod
    def drop_takes(
        cls, src: str, dst: Optional[str] = None, *, names: Iterable[str]
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
                file does not carry are ignored.

        Returns:
            ``{"takes", "objects", "connections"}`` -- the take names dropped
            (file order), ``{record name: count}`` removed, and the number of
            connections removed. When no named take is present the file is not
            written at all.

        Raises:
            ValueError: *src* is not a binary FBX this writer can re-emit.
        """
        wanted = {str(name) for name in names}
        report: Dict[str, Any] = {"takes": [], "objects": {}, "connections": 0}
        with (
            open(src, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            version, roots, footer_id, magic = cls._load(buf)
            sections = {record.name: record for record in roots}
            objects = sections.get(b"Objects")
            if objects is None:
                return report

            kind_of: Dict[int, bytes] = {}
            id_of_record: Dict[int, int] = {}
            stacks: Dict[int, str] = {}
            for record in objects.children:
                if record.name not in cls.TAKE_CHAIN:
                    continue
                head = cls._scalars(record)
                if not head or not isinstance(head[0], int):
                    continue
                kind_of[head[0]] = record.name
                id_of_record[id(record)] = head[0]
                if record.name == b"AnimationStack" and len(head) > 1:
                    display = cls._display(head[1])
                    if display in wanted:
                        stacks[head[0]] = display
            if not stacks:
                return report

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
                    owners = [
                        p for p in parents.get(oid, ()) if kind_of.get(p) == owner
                    ]
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

            if connections is not None:
                connections.children = [
                    record
                    for record, row in rows
                    if not (len(row) >= 3 and (row[1] in dropped or row[2] in dropped))
                ]
                report["connections"] = len(rows) - len(connections.children)

            takes = sections.get(b"Takes")
            if takes is not None:
                takes.children = [
                    child
                    for child in takes.children
                    if not (
                        child.name == b"Take" and cls._first_string(child) in wanted
                    )
                ]
                left = [
                    cls._first_string(c) for c in takes.children if c.name == b"Take"
                ]
                for child in takes.children:
                    if child.name == b"Current" and cls._first_string(child) in wanted:
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
                        cls._subtract_count(object_type, count)
                        total += count
                if total:
                    cls._subtract_count(definitions, total)

            report["takes"] = list(stacks.values())
            report["objects"] = {k.decode(): v for k, v in removed.items()}
            target = dst or src
            part = cls._write_part(target, version, roots, footer_id, magic)
        os.replace(part, target)
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
        every connection naming a removed object, its bind-pose entries and
        the removed objects' ``Definitions`` counts go with it. Everything else
        is copied byte for byte.

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
            counts: Dict[bytes, int] = {}
            survivors = []
            for record in objects.children:
                oid = graph["id_of"].get(id(record))
                if oid in removed:
                    counts[record.name] = counts.get(record.name, 0) + 1
                else:
                    survivors.append(record)
                    if record.name == b"Pose":
                        cls._drop_pose_nodes(record, removed)
            objects.children = survivors

            if connections is not None:
                before = len(connections.children)
                connections.children = [
                    record
                    for record, row in graph["rows"]
                    if not (len(row) >= 3 and (row[1] in removed or row[2] in removed))
                ]
                report["connections"] = before - len(connections.children)

            definitions = sections.get(b"Definitions")
            if definitions is not None:
                total = 0
                for object_type in definitions.children:
                    if object_type.name != b"ObjectType":
                        continue
                    head = cls._scalars(object_type)
                    count = counts.get(head[0], 0) if head else 0
                    if count:
                        cls._subtract_count(object_type, count)
                        total += count
                if total:
                    cls._subtract_count(definitions, total)

            report["models"] = len(dropped)
            report["kinds"] = RigMachinery.tally(
                {oid: doomed[names[oid]] for oid in dropped}
            )
            report["objects"] = {k.decode(): v for k, v in counts.items()}
            target = dst or src
            part = cls._write_part(target, version, roots, footer_id, magic)
        os.replace(part, target)
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
                names[oid] = cls._display(label)
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
        from pythontk.core_utils.scene_records import SceneRecords
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
    def _drop_pose_nodes(cls, pose: _Record, removed: set) -> None:
        """Remove *pose*'s entries for *removed* nodes and recount it."""
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
            return
        pose.children = [child for child in pose.children if child not in gone]
        for child in pose.children:
            if child.name != b"NbPoseNodes":
                continue
            head = cls._scalars(child)
            if head and isinstance(head[0], int):
                tag = bytes(child.payload[:1])
                left = max(0, head[0] - len(gone))
                child.replaced = (
                    b"L" + struct.pack("<q", left)
                    if tag == b"L"
                    else b"I" + struct.pack("<i", left)
                )

    @staticmethod
    def _display(raw: bytes) -> str:
        """The human half of an ``name\\x00\\x01Class`` object name."""
        return bytes(raw).split(b"\x00\x01", 1)[0].decode("utf-8", "replace")

    @classmethod
    def _first_string(cls, record: _Record) -> Optional[str]:
        head = cls._scalars(record)
        return (
            head[0].decode("utf-8", "replace")
            if head and isinstance(head[0], bytes)
            else None
        )

    @staticmethod
    def _pack_string(data: bytes) -> bytes:
        return b"S" + struct.pack("<I", len(data)) + data

    @classmethod
    def _subtract_count(cls, record: _Record, count: int) -> None:
        """Lower *record*'s ``Count`` child by *count* (never below zero)."""
        for child in record.children:
            if child.name != b"Count":
                continue
            head = cls._scalars(child)
            if head and isinstance(head[0], int):
                child.replaced = b"I" + struct.pack("<i", max(0, head[0] - count))

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
