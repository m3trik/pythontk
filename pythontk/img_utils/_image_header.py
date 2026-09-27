# !/usr/bin/python
# coding=utf-8
"""Image-file headers read without decoding the image: dimensions (PNG, JPEG,
TGA, DDS, EXR, Radiance HDR) and the float formats' structural integrity.
"""

from __future__ import annotations

import struct
from typing import List, Tuple, Optional


class _ImgHeaderInternal:
    """Header parsing behind :class:`ImgUtils`' size and integrity checks.

    Reached through :meth:`ImgUtils.get_image_size` and
    :meth:`ImgUtils.validate_image_integrity` (an ``ImgUtils`` base, so the
    helpers resolve as ``ImgUtils._<name>``); nothing here is called directly.
    """

    @staticmethod
    def _radiance_resolution(data: bytes) -> Tuple[str, int, List[Tuple[bytes, int]]]:
        """Walk a Radiance header to its resolution line -- the one parse behind
        both :meth:`_validate_radiance_hdr` and :meth:`_radiance_size`.

        The header is text up to the first blank line (``\\n\\n``, per the
        format: a ``\\r\\n`` header is not one), and the next line names both
        axes, slow (scanline) axis first: ``-Y 512 +X 1024``.

        Returns:
            ``(problem, offset, axes)``: *problem* is ``""`` or why the header is
            unusable (``"incomplete header"`` / ``"missing resolution line"``);
            *offset* is where the pixel data starts; *axes* is
            ``[(b"Y", 512), (b"X", 1024)]`` in file order, or ``[]`` for a
            resolution string this does not read (not an error).
        """
        blank = data.find(b"\n\n")
        if blank < 0:
            return "incomplete header", 0, []
        start = blank + 2
        eol = data.find(b"\n", start)
        if eol < 0:
            return "missing resolution line", 0, []
        tokens = data[start:eol].split()
        axes = []
        if len(tokens) == 4:
            for label, value in ((tokens[0], tokens[1]), (tokens[2], tokens[3])):
                if len(label) != 2 or label[:1] not in (b"+", b"-"):
                    return "", eol + 1, []
                try:
                    axes.append((label[1:], int(value)))
                except ValueError:
                    return "", eol + 1, []
        return "", eol + 1, axes

    @staticmethod
    def _validate_radiance_hdr(fp: str) -> Tuple[bool, str]:
        """Walk a Radiance HDR's scanlines to detect truncation."""
        with open(fp, "rb") as f:
            data = f.read()
        if not data.startswith(b"#?"):  # #?RADIANCE / #?RGBE
            return True, ""  # not the expected format; don't block
        problem, off, axes = _ImgHeaderInternal._radiance_resolution(data)
        if problem:
            return False, problem
        if len(axes) != 2:
            return True, ""  # nonstandard resolution string; skip the strict check
        # File order: scanline count, then scanline length -- whichever axes.
        (_, height), (_, width) = axes
        n = len(data)

        # New-style adaptive RLE: each scanline is 0x02 0x02 <hi> <lo> then four
        # run-length-encoded channels. Old/flat RGBE has no markers.
        if (
            8 <= width <= 0x7FFF
            and off + 4 <= n
            and data[off] == 2
            and data[off + 1] == 2
        ):
            rows = 0
            while rows < height and off + 4 <= n:
                if data[off] != 2 or data[off + 1] != 2:
                    break
                if ((data[off + 2] << 8) | data[off + 3]) != width:
                    break
                off += 4
                truncated = False
                for _channel in range(4):
                    x = 0
                    while x < width:
                        if off >= n:
                            truncated = True
                            break
                        run = data[off]
                        off += 1
                        if run > 128:  # a run of (run-128) identical bytes
                            off += 1
                            x += run - 128
                        else:  # (run) literal bytes
                            off += run
                            x += run
                    if truncated or off > n:
                        truncated = True
                        break
                if truncated:
                    break
                rows += 1
            if rows < height:
                return False, f"truncated: {rows}/{height} scanlines"
            return True, ""

        # Flat RGBE fallback: 4 bytes/pixel, no length markers.
        expected = width * height * 4
        if (n - off) < expected:
            return False, f"truncated: {n - off}/{expected} pixel bytes"
        return True, ""

    @staticmethod
    def _validate_exr(fp: str, size: int) -> Tuple[bool, str]:
        """Check an OpenEXR magic number + a sane minimum size."""
        with open(fp, "rb") as f:
            magic = f.read(4)
        if magic != b"\x76\x2f\x31\x01":
            return False, "not an OpenEXR file (bad magic)"
        if size < 64:  # header alone is larger than this
            return False, "EXR too small to be valid"
        return True, ""

    @staticmethod
    def _read_cstring(f, limit: int = 256) -> bytes:
        """Read a NUL-terminated string from *f* (the NUL consumed, not returned).

        ``b""`` at end of file, or when no NUL arrives within *limit* bytes --
        a corrupt header must end the walk, not run it to the end of the file.
        """
        out = bytearray()
        while len(out) < limit:
            char = f.read(1)
            if not char or char == b"\0":
                return bytes(out)
            out += char
        return b""

    @staticmethod
    def _exr_size(f) -> Optional[Tuple[int, int]]:
        """``(width, height)`` from an open OpenEXR file's ``dataWindow``.

        Walks the header's ``name\\0 type\\0 size value`` records from byte 8 and
        seeks past every attribute but the one it wants, so a header carrying a
        large attribute (a preview, render metadata) is not read. A multi-part
        file's first part answers: its header sits at the same offset.
        """
        f.seek(8)  # magic + version flags
        for _ in range(1024):  # tens of attributes in practice; bound a bad file
            name = _ImgHeaderInternal._read_cstring(f)
            if not name:  # the empty name that ends the header, or EOF
                return None
            _ImgHeaderInternal._read_cstring(f)  # type name
            raw = f.read(4)
            if len(raw) < 4:
                return None
            (size,) = struct.unpack("<i", raw)
            if name == b"dataWindow":
                box = f.read(16)
                if size != 16 or len(box) < 16:
                    return None
                x0, y0, x1, y1 = struct.unpack("<iiii", box)
                width, height = x1 - x0 + 1, y1 - y0 + 1
                return (width, height) if width > 0 and height > 0 else None
            if size < 0:
                return None
            f.seek(size, 1)
        return None

    @staticmethod
    def _radiance_size(f) -> Optional[Tuple[int, int]]:
        """``(width, height)`` from an open Radiance HDR's resolution line.

        Parsed by :meth:`_radiance_resolution`, so a header the integrity check
        rejects never sizes. Each value belongs to the axis it follows -- a
        rotated file lists X first -- so the pair is read by axis, not position.
        """
        f.seek(0)
        problem, _offset, axes = _ImgHeaderInternal._radiance_resolution(f.read(65536))
        if problem:
            return None
        axes = dict(axes)
        width, height = axes.get(b"X"), axes.get(b"Y")
        if not width or not height or width < 0 or height < 0:
            return None
        return width, height

    @staticmethod
    def _image_size_from_header(image_path: str) -> Optional[Tuple[int, int]]:
        """``(width, height)`` from an image header using only the stdlib.

        Covers JPEG, PNG, DDS, TGA, OpenEXR and Radiance HDR. Reads the
        dimensions out of the file header — no PIL, numpy, or cv2 — so it works
        in dependency-light interpreters (e.g. Metashape's bundled Python, or
        Blender's, where the fallback decodes the whole image), and it is the
        only way to size the two HDR formats, which PIL does not read at all.
        ``None`` for an unrecognized or truncated file.
        """
        try:
            with open(image_path, "rb") as f:
                head = f.read(24)
                if head[:4] == b"\x76\x2f\x31\x01":  # OpenEXR magic
                    return _ImgHeaderInternal._exr_size(f)
                if head[:2] == b"#?":  # Radiance: #?RADIANCE / #?RGBE
                    return _ImgHeaderInternal._radiance_size(f)
                # PNG: 8-byte signature, then IHDR chunk (width,height big-endian u32).
                if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
                    w, h = struct.unpack(">II", head[16:24])
                    return int(w), int(h)
                # DDS: "DDS " + DDS_HEADER (size 124), height then width, LE u32.
                if head[:4] == b"DDS " and len(head) >= 20:
                    if struct.unpack("<I", head[4:8])[0] == 124:
                        h, w = struct.unpack("<II", head[12:20])
                        return int(w), int(h)
                    return None
                # TGA has no signature: trust the extension, then the header's own
                # sanity (a known image type and pixel depth, a non-zero size).
                if str(image_path).lower().endswith((".tga", ".targa")):
                    if (
                        len(head) >= 18
                        and head[2] in (1, 2, 3, 9, 10, 11)
                        and head[16] in (8, 15, 16, 24, 32)
                    ):
                        w, h = struct.unpack("<HH", head[12:16])
                        if w and h:
                            return int(w), int(h)
                    return None
                # JPEG: SOI 0xFFD8, then scan segments for a Start-Of-Frame marker.
                if head[:2] == b"\xff\xd8":
                    f.seek(2)
                    while True:
                        b = f.read(1)
                        if not b:
                            return None
                        if b != b"\xff":
                            continue
                        marker = f.read(1)
                        while marker == b"\xff":  # skip fill bytes
                            marker = f.read(1)
                        if not marker:
                            return None
                        m = marker[0]
                        if 0xD0 <= m <= 0xD9:  # RSTn / SOI / EOI: no length
                            continue
                        lb = f.read(2)
                        if len(lb) < 2:
                            return None
                        seglen = struct.unpack(">H", lb)[0]
                        if seglen < 2:  # invalid: length includes its own 2 bytes
                            return None  # guards against a backward-seek infinite loop
                        # SOF0..SOF15 carry the frame size (excl. DHT/JPG/DAC: C4/C8/CC).
                        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                            f.read(1)  # sample precision
                            hw = f.read(4)
                            if len(hw) < 4:
                                return None
                            h, w = struct.unpack(">HH", hw)
                            return int(w), int(h)
                        f.seek(seglen - 2, 1)  # skip to next segment
        except Exception:
            return None
        return None
