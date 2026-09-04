# !/usr/bin/python
# coding=utf-8
"""Rewrite the embedded media of a binary FBX -- no DCC, no FBX SDK.

:class:`FbxMedia` is the writer :mod:`fbx_file` deliberately is not, scoped to
the one edit a hand-off pipeline needs: replacing the ``Video`` objects'
``Content`` payloads -- the textures ``FBXExportEmbeddedTextures`` copies into
the file at full authoring resolution. Everything else is copied byte for
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

import io
import logging
import mmap
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pythontk.file_utils.mesh_convert.fbx_file import FBX_MAGIC
from pythontk.file_utils.temp_artifacts import TempArtifacts

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
            workers: Decode/encode threads; ``None`` picks from the core count.

        Returns:
            ``{"images", "resized", "before", "after"}`` -- embedded image
            count, how many were rewritten, and the embedded bytes before and
            after. When nothing qualifies the file is not rewritten at all.

        Raises:
            ValueError: *src* is not a binary FBX this writer can re-emit.
        """
        from PIL import Image

        exempt = {os.path.basename(str(name)) for name in exempt}
        with (
            open(src, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            version, roots, footer_id, magic = cls._load(buf)
            videos = cls._video_records(roots)
            report = {
                "images": len(videos),
                "resized": 0,
                "before": sum(len(cls._raw(r)) for _n, r in videos),
                "after": 0,
            }

            def shrink(item: Tuple[str, _Record]) -> Optional[bytes]:
                name, record = item
                data = cls._raw(record)
                if not max_size or name in exempt:
                    return None
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        fmt = image.format
                        if (
                            fmt not in cls.REWRITABLE_FORMATS
                            or max(image.size) <= max_size
                        ):
                            return None
                        image.load()
                        if image.mode in ("P", "1"):
                            image = image.convert(
                                "RGBA" if "transparency" in image.info else "RGB"
                            )
                        if image.mode not in cls.RESIZABLE_MODES:
                            logger.debug(
                                "FbxMedia: %s left at %s (mode %s)",
                                name,
                                image.size,
                                image.mode,
                            )
                            return None
                        scale = max_size / float(max(image.size))
                        target = tuple(
                            max(1, round(edge * scale)) for edge in image.size
                        )
                        resized = image.resize(target, Image.LANCZOS)
                        out = io.BytesIO()
                        if fmt == "PNG":
                            resized.save(
                                out, format="PNG", compress_level=png_compress_level
                            )
                        else:
                            if resized.mode == "RGBA":
                                resized = resized.convert("RGB")
                            resized.save(out, format="JPEG", quality=jpeg_quality)
                except Exception as error:  # noqa: BLE001 -- keep the authored bytes
                    logger.warning("FbxMedia: %s left as found: %s", name, error)
                    return None
                encoded = out.getvalue()
                return encoded if len(encoded) < len(data) else None

            count = max(
                1, min(workers or min(8, os.cpu_count() or 1), len(videos) or 1)
            )
            with ThreadPoolExecutor(
                max_workers=count, thread_name_prefix="ptk-fbx-media"
            ) as pool:
                results = list(pool.map(shrink, videos))
            for (_name, record), encoded in zip(videos, results):
                if encoded is not None:
                    record.replaced = cls._pack_raw(encoded)
                    report["resized"] += 1
            report["after"] = sum(len(r.payload) - 5 for _n, r in videos)
            if not report["resized"]:
                return report

            target = dst or src
            # Beside the target, so the replace below stays atomic -- but
            # TRACKED, not a raw allocation: a raise inside _write (disk full
            # is realistic for a multi-hundred-MB payload) used to strand a
            # partial file with nothing to sweep it. The except arm clears it
            # now; the store's age-gated sweep clears it when the process dies
            # before any finally can run.
            scratch = TempArtifacts(
                "fbx_downsize", policy="scoped", dir=os.path.dirname(target) or "."
            )
            part = scratch.path(extension=".part")
            try:
                with open(part, "wb") as out:
                    cls._write(out, version, roots, footer_id, magic)
            except BaseException:
                scratch.cleanup()
                raise
        # After the mmap closes: Windows refuses to replace a mapped file, and
        # `target` is `src` for an in-place run.
        os.replace(part, target)
        return report

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
