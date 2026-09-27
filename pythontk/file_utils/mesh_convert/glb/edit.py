# !/usr/bin/python
# coding=utf-8
"""The GLB container every ``mesh_convert`` pass edits: :class:`GlbEdit`.

One parser and one writer for the GLB container: the header and JSON chunk
read once, the BIN pulled in lazily, the file written once on close. The
buffer bookkeeping every pass shares lives here too -- appending bufferViews,
collecting what no longer has a reader, and renumbering accessors -- so the
:class:`MeshConvert` passes and the ``glb`` siblings (reader, clips, fades,
key reduction, tangents) all edit through the same few functions.
"""

import base64
import hashlib
import json
import logging
import os
import struct
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

#: What every GLB repair on :class:`MeshConvert` accepts as its target: a path,
#: or an already-open :class:`GlbEdit` to join. The forward reference is never
#: resolved at runtime -- it exists so the signatures say which of the two they
#: take.
GlbTarget = Union[str, os.PathLike, "GlbEdit"]


class GlbEdit:
    """A GLB parsed once and held open: read the JSON, edit, write once.

    Every repair on :class:`MeshConvert` edits the JSON chunk and nothing
    else, but each used to open, parse and rewrite the file for itself --
    and the preview path runs three of them back to back on the same GLB.
    A single push therefore read a file the size of its geometry three
    times over to change a handful of fields. This is the shared handle
    that collapses that to one read and one write; see
    :meth:`open` for how a caller joins several repairs
    onto one session.

    Two properties carry most of the saving:

    - :attr:`rest` -- everything after the JSON chunk, which in practice is
      the geometry -- is read **lazily**. The common outcome is a repair
      that finds nothing to change, and that path now never reads past the
      first few kilobytes of a file that is routinely hundreds of
      megabytes.
    - :attr:`bin_data` is a ``memoryview`` into :attr:`rest`, not a slice
      of it. A slice copies, which put peak memory at twice the file size
      to gain nothing: every consumer here only ever reads it.
    """

    #: Byte offset of the JSON chunk's payload -- the 12-byte file header
    #: plus the 8-byte chunk header. Fixed by the GLB spec, and the seek
    #: target for the in-place write in :meth:`write`.
    JSON_OFFSET = 12 + 8

    def __init__(self, path: str, version_bytes: bytes, gltf: dict, json_len: int):
        self.path = path
        self.version_bytes = version_bytes
        self.gltf = gltf
        #: Length of the JSON chunk as it currently sits on disk. Updated
        #: by a full rewrite so a second write in the same session still
        #: knows where the chunk ends.
        self.json_len = json_len
        #: Set by any editor that changed :attr:`gltf`. Without it the
        #: close writes nothing, so a repair that matches nothing costs no
        #: I/O at all -- which is the usual case for the alpha repair that
        #: runs after every single conversion.
        self.dirty = False
        #: Set by :meth:`replace_rest`. Forces the writer down the
        #: full-rewrite path: after a BIN repack the JSON usually SHRINKS,
        #: and the in-place fast path would write the new JSON while
        #: leaving the OLD BIN on disk -- offsets pointing into bytes that
        #: no longer exist.
        self.rest_dirty = False
        self._rest: Optional[bytes] = None
        self._bin: Optional[memoryview] = None
        #: ``(image index, channel)`` -> that channel's ``(min, max)``.
        #: Keyed by channel so the alpha probes and the ORM probe share one
        #: decode of an atlas rather than one cache each.
        self._channel_extrema: Dict[Tuple[int, str], Optional[Tuple[int, int]]] = {}
        #: Texture path -> texture index, shared by every channel writer on
        #: this session so a map used as both base colour and emissive is
        #: embedded once rather than once per channel.
        self.embedded: Dict[str, int] = {}
        #: Image ENTRIES this session embedded, relocated out of the JSON
        #: chunk and into the BIN by
        #: :meth:`relocate_embedded_images` when the session
        #: closes. Held as the dicts themselves rather than indices
        #: because a pass that drops images (``prune_glb_textures``)
        #: rebuilds the list and shifts every index after the hole; the
        #: entry survives that, and its absence afterwards is exactly the
        #: signal that it was pruned and must not be relocated.
        self.pending_images: List[dict] = []
        #: sha256(image payload) -> image index, for images the file ALREADY
        #: carried when this session opened. Built lazily by
        #: :meth:`image_by_content`.
        self._image_digests: Optional[Dict[str, int]] = None

    @property
    def rest(self) -> bytes:
        """Every byte after the JSON chunk, read on first use and cached."""
        if self._rest is None:
            with open(self.path, "rb") as f:
                f.seek(self.JSON_OFFSET + self.json_len)
                self._rest = f.read()
        return self._rest

    def _trailing_chunks(self) -> bytes:
        """Whatever follows the BIN chunk -- or the JSON, when there is no BIN.

        GLB is a chunked container, and the spec tells a client that meets a
        chunk type it does not know to IGNORE it, not to discard it. Nothing
        here decodes such a chunk, so nothing here could rebuild one either;
        :meth:`replace_rest` carries this slice over verbatim instead.
        """
        rest = self.rest
        if len(rest) >= 8 and rest[4:8] == b"BIN\x00":
            # Chunk lengths are 4-byte aligned by spec, so the declared
            # length already covers the BIN's own padding.
            return bytes(rest[8 + struct.unpack("<I", rest[:4])[0] :])
        return bytes(rest)  # no BIN at all: the whole tail is other chunks

    def replace_rest(self, new_bin: bytes) -> None:
        """Swap the BIN chunk's payload for *new_bin* (repack support).

        Rebuilds the chunk header, drops the derived caches, and marks the
        session so the writer rewrites the whole container -- see
        :attr:`rest_dirty` for why the in-place path must not run.

        Only the BIN is replaced; any chunk beyond it rides along untouched
        (see :meth:`_trailing_chunks`). This rebuilds the entire tail from
        one payload, so without that carry-over a repack would delete bytes
        it never read. BIN is emitted first, which is where the spec wants
        it, so the result stays valid even for a file that had no BIN.
        """
        tail = self._trailing_chunks()
        new_bin += b"\x00" * ((4 - (len(new_bin) % 4)) % 4)
        self._rest = struct.pack("<I4s", len(new_bin), b"BIN\x00") + new_bin + tail
        self._bin = None
        self._image_digests = None
        self._channel_extrema = {}
        self.rest_dirty = True
        self.dirty = True

    @property
    def bin_data(self) -> Optional[memoryview]:
        """The BIN chunk's payload as a read-only view, or ``None``."""
        if self._bin is None:
            rest = self.rest
            if len(rest) >= 8 and rest[4:8] == b"BIN\x00":
                length = struct.unpack("<I", rest[:4])[0]
                self._bin = memoryview(rest)[8 : 8 + length]
            else:  # probed and absent; remembered so we don't re-probe
                self._bin = memoryview(b"")
        return self._bin or None

    @property
    def materials(self) -> list:
        return self.gltf.get("materials", []) or []

    @property
    def textures(self) -> list:
        return self.gltf.get("textures", []) or []

    @property
    def images(self) -> list:
        return self.gltf.get("images", []) or []

    @property
    def buffer_views(self) -> list:
        return self.gltf.get("bufferViews", []) or []

    def _image_payload(self, image: dict) -> Optional[bytes]:
        """An image's encoded bytes, or ``None`` when they are not in the file.

        Reads :attr:`bin_data` only for an image that actually uses a
        ``bufferView``, so a file whose images are all data URIs or external
        keeps this class's "never read past the JSON chunk" property.
        """
        view_index = image.get("bufferView")
        if view_index is not None:
            blob = self.bin_data
            views = self.buffer_views
            if blob is None or not 0 <= view_index < len(views):
                return None
            view = views[view_index]
            start = view.get("byteOffset", 0)
            return bytes(blob[start : start + view["byteLength"]])
        uri = image.get("uri") or ""
        if uri.startswith("data:") and "," in uri:
            try:
                return base64.b64decode(uri.split(",", 1)[1])
            except ValueError:  # binascii.Error subclasses ValueError
                return None
        return None

    @property
    def image_digests(self) -> Dict[str, int]:
        """``sha256(payload) -> image index`` for the images this file holds.

        The embed cache upstream is keyed by source PATH, so it can only
        dedupe embeds this session made. It cannot see that the converter
        already embedded the very same texture as a ``bufferView`` -- which
        is the normal case, because an FBX exported with embedded media
        carries its whole PBR set into the GLB, and the sidecar then
        re-applies the channels FBX translation drops. Measured on a
        production room: base colour, diffuse and emissive were each
        present twice, 23 MB of duplicate payload costing 31 MB on disk
        once base64 inflated the second copy -- a quarter of the file.

        Content-addressed rather than name-addressed because the two copies
        arrive by different routes and agree on nothing else: the
        bufferView image is named by the FBX exporter, the sidecar's by its
        source path. Writers register what they append, so two source files
        with identical bytes also collapse to one embed.
        """
        if self._image_digests is None:
            self._image_digests = {}
            for index, image in enumerate(self.images):
                payload = self._image_payload(image)
                if payload:
                    self._image_digests.setdefault(
                        hashlib.sha256(payload).hexdigest(), index
                    )
        return self._image_digests

    def texture_for_image(self, image_index: int) -> int:
        """Index of a texture sampling *image_index*, appending one if needed.

        An existing texture is reused with ITS sampler rather than a fresh
        one forced to repeat: that texture is how the file itself already
        samples this image, so inheriting it is the answer least likely to
        change how anything renders.
        """
        for index, texture in enumerate(self.textures):
            if texture.get("source") == image_index:
                return index
        textures = self.gltf.setdefault("textures", [])
        samplers = self.gltf.setdefault("samplers", [])
        if not samplers:
            samplers.append({"wrapS": 10497, "wrapT": 10497})
        textures.append({"source": image_index, "sampler": 0})
        return len(textures) - 1

    def image_for_texture(self, texture_index: Any) -> Optional[int]:
        """Image index a texture samples, or ``None``.

        The inverse of :meth:`texture_for_image`, and the one place that
        knows an extension binding shadows the plain ``source``: after
        :meth:`MeshConvert.optimize_glb_textures` a texture carries
        ``EXT_texture_webp`` or ``KHR_texture_basisu``, beside a plain
        ``source`` only when the KTX2 pass wrote a core-readable fallback
        twin -- either way the extension is what a capable loader reads.
        Bounds-checked
        like :meth:`base_color_image` -- a negative index is malformed,
        not a reference to the last texture.
        """
        textures = self.textures
        if not isinstance(texture_index, int):
            return None
        if not 0 <= texture_index < len(textures):
            return None
        texture = textures[texture_index]
        extensions = texture.get("extensions") or {}
        source = texture.get("source")
        for binding in self.TEXTURE_CONTAINER_EXTENSIONS:
            shadow = (extensions.get(binding) or {}).get("source")
            if shadow is not None:
                source = shadow
                break
        if not isinstance(source, int) or not 0 <= source < len(self.images):
            return None
        return source

    def base_color_image(self, mat: dict) -> Optional[int]:
        """Index of *mat*'s base-colour image, or ``None`` if it has none.

        The shared front half of both alpha repairs: material -> pbr ->
        baseColorTexture -> texture -> image, bounds-checked at each hop.
        Written out twice it had already drifted -- one copy bounds-checked
        the image index and the other left it to the probe to notice.

        The texture -> image half (including both bounds checks, and the
        ``EXT_texture_webp`` shadowing) is :meth:`image_for_texture`'s job;
        this adds only the material -> texture hop. Written out in full it
        had already drifted once, so there is one copy of the walk.
        """
        pbr = mat.get("pbrMetallicRoughness") or {}
        bct = pbr.get("baseColorTexture")
        if not bct:
            return None
        return self.image_for_texture(bct.get("index"))

    def image_label(self, img_idx: int) -> str:
        """How a finding or a fix record names an image.

        Its glTF name, else its uri, else its index -- a converter that
        carried none of the first two still has to produce something a
        reader can match against the file.
        """
        images = self.images
        entry = images[img_idx] if 0 <= img_idx < len(images) else {}
        return entry.get("name") or entry.get("uri") or f"image[{img_idx}]"

    def image_bytes(self, img_entry: dict) -> Optional[bytes]:
        """Raw bytes for a glTF image entry, or ``None`` if unavailable.

        Resolves all three ways an image can be stored: inline as a
        ``data:`` URI, as a file beside the GLB, or as a slice of the BIN
        chunk. Only the last touches :attr:`bin_data`, so a GLB whose
        images are all external never reads the geometry.
        """
        uri = img_entry.get("uri")
        if uri:
            if uri.startswith("data:"):
                try:
                    _, b64 = uri.split(",", 1)
                    return base64.b64decode(b64)
                except Exception:  # noqa: BLE001 — malformed URI, not fatal
                    return None
            sibling = os.path.join(os.path.dirname(self.path), uri)
            if os.path.isfile(sibling):
                with open(sibling, "rb") as f:
                    return f.read()
            return None

        bv_idx = img_entry.get("bufferView")
        buffer_views = self.buffer_views
        # Two-sided, as in `base_color_image`: a negative index would slice
        # the wrong bufferView rather than be rejected.
        if not isinstance(bv_idx, int) or not 0 <= bv_idx < len(buffer_views):
            return None
        bin_data = self.bin_data
        if bin_data is None:
            return None
        bv = buffer_views[bv_idx]
        offset = bv.get("byteOffset", 0)
        length = bv.get("byteLength", 0)
        return bin_data[offset : offset + length] or None

    def alpha_extrema(self, img_idx: int) -> Optional[Tuple[int, int]]:
        """``(min, max)`` of an image's alpha channel, or ``None``.

        ``None`` covers every case a caller must not act on: an index out
        of range, bytes that could not be resolved, a decoder that refused
        the file, and an image with no alpha channel at all -- they are
        deliberately not distinguished, because the answer to each is the
        same "leave this material alone".
        """
        return self.channel_extrema(img_idx, "A")

    def channel_extrema(
        self, img_idx: int, channel: str = "A"
    ) -> Optional[Tuple[int, int]]:
        """``(min, max)`` of one ``R``/``G``/``B``/``A`` channel, or ``None``.

        ``None`` is every case a caller must not act on: an index out of
        range, bytes that could not be resolved, a decoder that refused the
        file, and -- for ``A`` alone -- an image with no alpha channel at
        all. Deliberately not distinguished: the answer to each is the same
        "leave this material alone".

        Cached on the session per ``(image, channel)``, so an atlas shared
        by twenty materials is decoded once across *every* probe here (both
        alpha repairs and the ORM check) rather than once inside each.

        Pillow is imported here rather than at module scope; the public
        entry points check for it up front and raise something actionable.
        """
        key = (img_idx, channel)
        if key in self._channel_extrema:
            return self._channel_extrema[key]

        from io import BytesIO

        from PIL import Image

        result: Optional[Tuple[int, int]] = None
        images = self.images
        if img_idx < len(images):
            raw = self.image_bytes(images[img_idx])
            if raw:
                try:
                    with Image.open(BytesIO(raw)) as im:
                        im.load()
                        if channel == "A":
                            # An image with no alpha is not "alpha 255
                            # everywhere" -- the callers repair BLEND
                            # materials, and treating opaque-by-absence as
                            # a finding would flag every RGB texture.
                            has_alpha = im.mode in ("RGBA", "LA", "PA") or (
                                im.mode == "P" and "transparency" in im.info
                            )
                            if has_alpha:
                                result = im.convert("RGBA").getchannel("A").getextrema()
                        else:
                            result = im.convert("RGB").getchannel(channel).getextrema()
                except Exception as exc:  # noqa: BLE001 — varied decoder errors
                    logger.debug(
                        "GLB channel probe: skipped image %s (%s) [%s]",
                        img_idx,
                        exc,
                        channel,
                    )
        self._channel_extrema[key] = result
        return result

    #: Texture-container extensions the texture pass binds, and therefore owns
    #: the declarations for (:meth:`MeshConvert.optimize_glb_textures`). Anything else in ``extensionsUsed`` (material
    #: extensions, ``KHR_texture_transform`` from the lightmap pass) is another
    #: writer's claim and is never touched by the reconciliation below.
    #:
    #: Also the complete list of places a texture can name its image BESIDE the
    #: core ``source``, which is why `GlbEdit.image_for_texture` (resolving a
    #: binding) and `MeshConvert.dedupe_glb_images` (rewriting one) both read it: three
    #: hand-written copies of these two names is exactly how one of them ends
    #: up not knowing about the next container added here.
    #:
    #: Ordered by RESOLUTION PRECEDENCE, basisu first, for `image_for_texture`,
    #: which takes the first binding it finds. The KTX2 encode pops any webp
    #: binding before writing its own, so the two provably never coexist today
    #: and the order is moot -- but that guarantee lives in a distant method,
    #: and a set whose iteration order is load-bearing somewhere should not
    #: depend on it. `_reconcile_texture_extensions` handles each name
    #: independently and is order-free.
    TEXTURE_CONTAINER_EXTENSIONS = ("KHR_texture_basisu", "EXT_texture_webp")

    #: glTF ``componentType`` -> (struct format char, byte size). The one
    #: table, shared with :class:`GlbReader` -- which imports this module, so
    #: the layout facts live on this side of that edge rather than in a second
    #: copy that can drift.
    ACCESSOR_COMPONENT_TYPES: Dict[int, Tuple[str, int]] = {
        5120: ("b", 1),
        5121: ("B", 1),
        5122: ("h", 2),
        5123: ("H", 2),
        5125: ("I", 4),
        5126: ("f", 4),
    }

    #: glTF accessor ``type`` -> component count.
    ACCESSOR_TYPE_COUNT: Dict[str, int] = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }

    @classmethod
    @contextmanager
    def open(cls, glb: GlbTarget):
        """Yield an open :class:`GlbEdit` for *glb*, writing once on close.

        *glb* is a path, **or an already-open session** -- in which case it is
        yielded as-is and its owner keeps responsibility for the write. That
        second form is what lets these repairs compose: each one takes either,
        so running three against a path costs three read/write cycles, while
        wrapping the same three in one ``open`` costs one::

            with GlbEdit.open(path) as glb:
                MeshConvert.set_glb_base_color(glb, base_color)
                MeshConvert.set_glb_emissive(glb, emissive)

        Nothing is written when the body raises -- a half-applied edit must not
        reach disk -- nor when no editor set :attr:`GlbEdit.dirty`.
        """
        if isinstance(glb, cls):
            yield glb
            return
        edit = cls.read(os.fspath(glb))
        yield edit
        if edit.dirty:
            # Once, here, rather than per writer: composed repairs each embed
            # into the JSON chunk and the BIN is rebuilt a single time for all
            # of them. Runs on the OWNER's close only -- a session handed in by
            # a caller is relocated when that caller closes it, after its own
            # writers have finished embedding.
            cls.relocate_embedded_images(edit)
            cls.write(edit)

    @classmethod
    def read(cls, glb_path: str) -> "GlbEdit":
        """Parse a GLB's container and JSON chunk into an open edit session.

        The single owner of GLB container parsing for this class — the JSON
        chunk is what every repair here edits, and re-deriving the offsets per
        function is how one of them ends up with a subtly different idea of
        where the BIN chunk starts.

        Only the JSON chunk is read: everything after it is pulled in on demand
        by :attr:`GlbEdit.rest`, because most repairs decide they have nothing
        to do from the JSON alone and the remainder is the whole geometry.
        """
        if not os.path.isfile(glb_path):
            raise FileNotFoundError(glb_path)

        with open(glb_path, "rb") as f:
            # Read the fixed 12-byte header and the 8-byte chunk header in one
            # go each and length-check them: a file truncated mid-write reaches
            # struct.unpack with a short buffer, and struct.error derives from
            # Exception — it slips past callers' (RuntimeError, ValueError,
            # OSError) per-file handlers and aborts a whole batch.
            header = f.read(12)
            if len(header) < 12 or header[:4] != b"glTF":
                raise ValueError(f"Not a GLB file: {glb_path}")
            version_bytes = header[4:8]  # total length is recomputed on write
            chunk0_header = f.read(8)
            if len(chunk0_header) < 8:
                raise ValueError(f"Malformed GLB: truncated chunk header ({glb_path})")
            chunk0_len = struct.unpack("<I", chunk0_header[:4])[0]
            chunk0_type = chunk0_header[4:]
            if chunk0_type != b"JSON":
                raise ValueError(f"Malformed GLB: first chunk not JSON ({glb_path})")
            payload = f.read(chunk0_len)
            # Same reasoning as the header checks, and newly reachable now that
            # the read stops at the chunk boundary instead of consuming the
            # file: a chunk header promising more JSON than the file holds must
            # surface as the ValueError callers already handle.
            if len(payload) < chunk0_len:
                raise ValueError(f"Malformed GLB: truncated JSON chunk ({glb_path})")
            gltf = json.loads(payload.decode("utf-8"))

        return cls(glb_path, version_bytes, gltf, chunk0_len)

    @staticmethod
    def write(edit: "GlbEdit") -> None:
        """Persist *edit*'s JSON chunk, rewriting the whole file only if it must.

        The JSON chunk is padded to a 4-byte boundary with spaces and the total
        length recomputed — both required by the GLB spec, and both easy to get
        wrong in a way that only some loaders reject.

        When the re-serialized JSON still fits the chunk it came from it is
        padded back out to *exactly* that length and written in place, so every
        offset after it — and the entire BIN chunk — is left untouched. That is
        the common case rather than a lucky one: the alpha repair changes a
        single float, and this writer serializes compactly while most producers
        do not. The alternative is pulling a few hundred megabytes of geometry
        through memory to edit a few bytes of JSON.

        Padding out to the original length rather than shrinking to fit is what
        keeps that safe: the chunk header, the total-length field and every
        byte beyond stay correct only if the chunk keeps its size.
        """
        new_json = json.dumps(edit.gltf, separators=(",", ":")).encode("utf-8")

        if len(new_json) <= edit.json_len and not edit.rest_dirty:
            new_json += b" " * (edit.json_len - len(new_json))
            with open(edit.path, "r+b") as f:
                f.seek(edit.JSON_OFFSET)
                f.write(new_json)
            return

        new_json += b" " * ((4 - (len(new_json) % 4)) % 4)
        # Resolved before the truncating open, not inside it: `rest` is lazy,
        # and reading it back out of a file we have just emptied would hand the
        # writer nothing.
        rest = edit.rest
        with open(edit.path, "wb") as f:
            f.write(b"glTF")
            f.write(edit.version_bytes)
            f.write(struct.pack("<I", 12 + 8 + len(new_json) + len(rest)))
            f.write(struct.pack("<I", len(new_json)))
            f.write(b"JSON")
            f.write(new_json)
            f.write(rest)
        edit.json_len = len(new_json)

    @staticmethod
    def append_bin_views(edit: "GlbEdit", payloads: Sequence[bytes]) -> List[int]:
        """Append *payloads* to the BIN as new bufferViews; return their indices.

        The one place bytes are added to a GLB's buffer, shared by the image
        relocator and the visibility writer. Append-only by construction: the
        existing BIN is copied verbatim and the new payloads land past its end,
        so every prior bufferView keeps its index, its ``byteOffset`` and its
        bytes, and no accessor is touched. Each payload is padded to a 4-byte
        boundary, which is what an accessor reading it requires.

        Returns ``[]`` without touching the file when there is nothing to
        append, or when buffer 0 is EXTERNAL (declares a ``uri``): such a GLB
        has no BIN to append to, and writing one would strand the appended
        views on bytes the file does not carry while overwriting that buffer's
        ``byteLength``. Callers treat the empty list as "not possible here" and
        leave their payload wherever it already is.
        """
        if not payloads:
            return []
        gltf = edit.gltf
        buffers = gltf.setdefault("buffers", [])
        if buffers and buffers[0].get("uri"):
            return []

        # The existing BIN joins in as a memoryview rather than a `bytes` copy:
        # on a production GLB that copy is the entire geometry, and `join`
        # reads the view directly, so peak memory is one BIN, not two.
        blob = edit.bin_data
        chunks: List[Any] = [] if blob is None else [blob]
        offset = 0 if blob is None else len(blob)
        pad = (4 - (offset % 4)) % 4
        if pad:  # the appended views must start 4-byte aligned
            chunks.append(b"\x00" * pad)
            offset += pad

        views = gltf.setdefault("bufferViews", [])
        added: List[int] = []
        for raw in payloads:
            views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw)})
            added.append(len(views) - 1)
            chunks.append(raw)
            offset += len(raw)
            tail = (4 - (len(raw) % 4)) % 4
            if tail:
                chunks.append(b"\x00" * tail)
                offset += tail

        new_bin = b"".join(chunks)
        if not buffers:  # a GLB that carried no BIN at all now has one
            buffers.append({})
        buffers[0]["byteLength"] = len(new_bin)
        edit.replace_rest(new_bin)
        return added

    @staticmethod
    def compact_bin(edit: "GlbEdit") -> int:
        """Reclaim every bufferView nothing references, and repack the BIN.

        The garbage collector behind any pass that ORPHANS payload. It does not
        decide what is dead -- it collects what no longer has a reader -- so a
        pass only has to rewire, and whatever it stopped pointing at is
        reclaimed here. Membership is read off the file itself: every
        ``bufferView`` key anywhere in the JSON, which is the only sweep that
        is correct for both kinds of reference. An accessor-only sweep is the
        trap this exists to avoid -- images name their view DIRECTLY rather
        than through an accessor, so counting accessor references alone reads
        every image in the file as garbage (measured while sizing this pass on
        a production assembly: 75.78 MB of phantom saving).

        Survivors keep their order and their dicts; only ``byteOffset`` moves,
        each view padded to the 4-byte boundary an accessor requires, and every
        ``bufferView`` reference in the JSON is renumbered to match.

        Returns:
            Bytes of BIN payload reclaimed. 0 -- and no rewrite -- when nothing
            is orphaned, when buffer 0 is EXTERNAL (declares a ``uri``), where
            there is no BIN here to repack, or when a surviving view reaches
            past the end of the BIN, where repacking would turn a file that
            arrived truncated into one that merely lies about its offsets.
        """
        gltf = edit.gltf
        views = gltf.get("bufferViews") or []
        if not views:
            return 0
        buffers = gltf.get("buffers") or []
        if buffers and buffers[0].get("uri"):
            return 0
        if edit.bin_data is None:
            # Views but no BIN to read them from: the file is already
            # inconsistent, and repacking would write a zero-length buffer
            # under views that still declare a length. Leave it as found.
            return 0

        def _walk(node, out):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "bufferView" and isinstance(value, int):
                        out.add(value)
                    else:
                        _walk(value, out)
            elif isinstance(node, list):
                for item in node:
                    _walk(item, out)
            return out

        # Everything BUT the view array itself: a view names a buffer, never
        # another view, and walking it would re-add every index.
        live = _walk({k: v for k, v in gltf.items() if k != "bufferViews"}, set())
        if all(index in live for index in range(len(views))):
            return 0

        blob = edit.bin_data
        # Every surviving view must be fully backed BEFORE anything is written.
        # Slicing past the end of the BIN yields a short string silently, and
        # the view keeps the `byteLength` it declared -- so a file that arrived
        # truncated would be rewritten into one whose views point at bytes that
        # are not there, which no longer reads as damaged, just wrong. This pass
        # runs on every conversion (see `MeshConvert.fbx_to_glb`), so declining is the only
        # safe answer: leave the file exactly as found and reclaim nothing.
        for index in sorted(live):
            if not 0 <= index < len(views):
                continue
            view = views[index] or {}
            end = int(view.get("byteOffset") or 0) + int(view.get("byteLength") or 0)
            if end > len(blob):
                logger.warning(
                    "BIN compaction declined: bufferView %d ends at %d but the "
                    "buffer is %d bytes. Repacking would silently shorten it.",
                    index,
                    end,
                    len(blob),
                )
                return 0

        chunks: List[bytes] = []
        kept: List[Dict[str, Any]] = []
        view_map: Dict[int, int] = {}
        offset = 0
        reclaimed = 0
        for old, view in enumerate(views):
            length = int(view.get("byteLength") or 0)
            if old not in live:
                reclaimed += length
                continue
            start = int(view.get("byteOffset") or 0)
            # Fully backed: the sweep above proved it, and `bin_data is None`
            # returned early.
            data = bytes(blob[start : start + length])
            view = dict(view)
            view["byteOffset"] = offset
            chunks.append(data)
            offset += len(data)
            pad = (4 - (len(data) % 4)) % 4
            if pad:
                chunks.append(b"\x00" * pad)
                offset += pad
            view_map[old] = len(kept)
            kept.append(view)

        gltf["bufferViews"] = kept

        def _remap(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "bufferView" and isinstance(value, int):
                        node[key] = view_map.get(value, value)
                    else:
                        _remap(value)
            elif isinstance(node, list):
                for item in node:
                    _remap(item)

        # The comprehension is a new dict over the SAME value objects, so the
        # renumbering lands on the file's own nodes.
        _remap({k: v for k, v in gltf.items() if k != "bufferViews"})

        new_bin = b"".join(chunks)
        if not buffers:
            gltf["buffers"] = buffers = [{}]
        buffers[0]["byteLength"] = len(new_bin)
        edit.replace_rest(new_bin)
        return reclaimed

    @classmethod
    def relocate_embedded_images(cls, edit: "GlbEdit") -> int:
        """Move this session's embedded images from the JSON chunk into the BIN.

        Every channel writer embeds as a ``data:`` URI, which keeps its edit
        inside the JSON chunk -- no buffer offsets to recompute, which is the
        part of GLB surgery that silently corrupts a file. That was priced for
        a local preview, but the same writers build deliverables: measured on
        TURRETS_WIRES.glb, the packed ORM's base64 put the JSON chunk at 45% of
        an 8.9 MB file, 1.0 MB of it pure base64 premium, all of it parsed
        before a loader can draw. This pays the JSON back down once, on close,
        after every writer has had its say.

        Safe because it only ever **appends** (see :meth:`append_bin_views`,
        which does the appending). That is the whole difference from
        ``optimize_glb_textures``, which rewrites payloads in place and so must
        recompute the offsets this pass leaves alone.

        Only images *this session embedded* move. A ``data:`` URI the file
        arrived with is the caller's, and an external ``uri`` cannot be read
        from here at all -- rewriting either would be a side effect on input
        rather than a fix to output.

        Returns:
            Number of images relocated (0 leaves the file untouched).
        """
        pending, edit.pending_images = edit.pending_images, []
        if not pending:
            return 0
        gltf = edit.gltf
        images = gltf.get("images") or []
        # Identity, not index: a pruning pass rebuilds `images` and shifts
        # every index after the hole, but a dropped entry is simply absent.
        live_ids = {id(image) for image in images}
        live = [
            image
            for image in pending
            if id(image) in live_ids and str(image.get("uri", "")).startswith("data:")
        ]
        if not live:
            return 0
        # Leave the payloads in the JSON when there is no BIN to append to:
        # base64 is a size cost, corrupting the buffer table is not.
        views = cls.append_bin_views(
            edit, [base64.b64decode(image["uri"].split(",", 1)[1]) for image in live]
        )
        if not views:
            return 0
        for image, view in zip(live, views):
            image["bufferView"] = view
            image.pop("uri", None)
        return len(live)

    @staticmethod
    def bin_view(gltf: Dict[str, Any], accessor: Dict[str, Any]) -> Optional[dict]:
        """The bufferView *accessor* reads, when its bytes are this GLB's BIN.

        A GLB embeds exactly one buffer -- buffer 0, with no ``uri`` -- so a
        view on another buffer, on a buffer 0 that names an external file, or
        one a compression extension decodes (``EXT_meshopt_compression``) does
        not describe the bytes at its offsets in the BIN chunk. Read there
        anyway it decodes to plausible garbage rather than an error. ``None``
        for those, and for an accessor with no (valid) ``bufferView``; the one
        definition every accessor reader and writer here checks against.
        """
        views = gltf.get("bufferViews") or []
        view_index = accessor.get("bufferView")
        if not isinstance(view_index, int) or not 0 <= view_index < len(views):
            return None
        view = views[view_index] or {}
        buffers = gltf.get("buffers") or []
        if (
            view.get("buffer", 0) != 0
            or view.get("extensions")
            or not buffers
            or (buffers[0] or {}).get("uri")
        ):
            return None
        return view

    @classmethod
    def accessor_elements(cls, edit: "GlbEdit", index: int) -> Optional[List[bytes]]:
        """One accessor's elements as raw byte strings, or None if unreadable.

        Bytes, not decoded numbers: the question every caller here asks is
        whether the elements are IDENTICAL, and comparing the payload answers
        it exactly, without a float round-trip that could call two different
        bit patterns equal (or two equal ones different).

        None whenever the layout is anything but tightly packed and
        self-contained -- sparse, interleaved (``byteStride``), no bufferView,
        or bytes that are not this file's BIN (:meth:`bin_view`). Those are
        legal glTF that this pass has no business rewriting, and refusing them
        is what keeps it safe to run on any file.
        """
        accessors = edit.gltf.get("accessors") or []
        if not 0 <= index < len(accessors):
            return None
        accessor = accessors[index] or {}
        if accessor.get("sparse"):
            return None
        view = cls.bin_view(edit.gltf, accessor)
        if view is None or view.get("byteStride"):
            return None
        spec = cls.ACCESSOR_COMPONENT_TYPES.get(accessor.get("componentType"))
        count = cls.ACCESSOR_TYPE_COUNT.get(accessor.get("type"))
        if not spec or not count:
            return None
        size = spec[1]
        stride = size * count
        n = int(accessor.get("count") or 0)
        blob = edit.bin_data
        if blob is None or n <= 0:
            return None
        start = int(view.get("byteOffset") or 0) + int(accessor.get("byteOffset") or 0)
        end = start + n * stride
        if end > int(view.get("byteOffset") or 0) + int(view.get("byteLength") or 0):
            return None
        raw = bytes(blob[start:end])
        if len(raw) != n * stride:  # truncated BIN: slicing would pad silently
            return None
        return [raw[i : i + stride] for i in range(0, len(raw), stride)]

    #: Extensions that hold accessor indices the core walk cannot see. Dropping
    #: an animation's payload renumbers nothing, but it does decide which
    #: accessors are unreferenced, and one of these could still be reading a
    #: buffer this pass is about to release. Same "bail whole rather than
    #: guess" contract as :attr:`MeshConvert._IMAGE_REFERRING_EXTENSIONS`.
    _ACCESSOR_REFERRING_EXTENSIONS = frozenset(
        {
            "EXT_mesh_gpu_instancing",
            "EXT_mesh_features",
            "EXT_instance_features",
            "EXT_structural_metadata",
        }
    )

    @classmethod
    def map_accessor_refs(cls, gltf: Dict[str, Any], visit) -> None:
        """Apply *visit* to every CORE accessor reference in *gltf*.

        ``visit(index)`` returns a replacement index, or ``None`` to leave it.
        One definition of "where accessors are named", shared by the collector
        and the renumberer so the two cannot disagree about a site -- the bug
        that shape prevents is dropping an accessor something still reads.

        Accessors are named by a dozen different keys rather than one (an
        ``indices`` here, an ``inverseBindMatrices`` there, a whole
        ``attributes`` map per primitive), which is why they cannot be swept
        generically the way ``bufferView`` can, and why the caller pairs this
        with a refusal on extensions that add sites of their own.
        """

        def at(container, key):
            value = container.get(key)
            if isinstance(value, int):
                replacement = visit(value)
                if replacement is not None:
                    container[key] = replacement

        for mesh in gltf.get("meshes") or []:
            for prim in (mesh or {}).get("primitives") or []:
                at(prim, "indices")
                maps = [prim.get("attributes") or {}]
                maps.extend(t or {} for t in prim.get("targets") or [])
                for attributes in maps:
                    for name in list(attributes):
                        at(attributes, name)
        for skin in gltf.get("skins") or []:
            at(skin or {}, "inverseBindMatrices")
        for animation in gltf.get("animations") or []:
            for sampler in (animation or {}).get("samplers") or []:
                at(sampler or {}, "input")
                at(sampler or {}, "output")

    @classmethod
    def referenced_accessors(cls, gltf: Dict[str, Any]) -> set:
        """Every accessor index the file still reads, from the core sites."""
        live: set = set()

        def collect(index):
            live.add(index)
            return None  # collect only; never rewrite

        cls.map_accessor_refs(gltf, collect)
        return live

    @classmethod
    def drop_orphaned_accessors(cls, edit: "GlbEdit", candidates: Set[int]) -> int:
        """Delete the *candidates* nothing reads any more; renumber the rest.

        Deleted, not merely unbound. An unbound accessor is valid glTF and
        costs only its JSON, but it keeps its bufferView alive for
        :meth:`compact_bin`, the numbering drifts further from the file a
        reader sees, and a pass that runs on every rebuild accumulates a dead
        accessor per channel per run (measured on the clip fixture: +272 bytes
        a run). The constant-channel collapse left 15,843 of them (1.59 MB of
        JSON) on a production 4K assembly.

        Parameters:
            edit: An open session, already rewired: what is still referenced
                is read off the file rather than passed in.
            candidates: Accessor indices the caller stopped reading; only the
                ones nothing else references are deleted.

        Returns:
            How many were deleted. 0 when none is orphaned, or when the file
            uses an extension that may hold accessor indices of its own
            (:attr:`_ACCESSOR_REFERRING_EXTENSIONS`): renumbering under one of
            those would silently re-point it at the wrong data, so the pass
            declines rather than guess.
        """
        gltf = edit.gltf
        accessors = gltf.get("accessors") or []
        orphaned = {
            index
            for index in candidates
            if isinstance(index, int) and 0 <= index < len(accessors)
        } - cls.referenced_accessors(gltf)
        if not orphaned:
            return 0
        foreign = cls._ACCESSOR_REFERRING_EXTENSIONS & set(
            gltf.get("extensionsUsed") or []
        )
        if foreign:
            logger.info(
                "Accessors: kept %d unreferenced accessor(s) -- %s may hold "
                "accessor indices this pass cannot see.",
                len(orphaned),
                ", ".join(sorted(foreign)),
            )
            return 0
        keep = [i for i in range(len(accessors)) if i not in orphaned]
        remap = {old: new for new, old in enumerate(keep)}
        gltf["accessors"] = [accessors[i] for i in keep]
        cls.map_accessor_refs(gltf, lambda index: remap.get(index))
        edit.dirty = True
        return len(orphaned)

    @classmethod
    def release_animation_payload(
        cls,
        edit: "GlbEdit",
        animations: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    ) -> int:
        """Free what animations REMOVED from the file were reading.

        Dropping a clip from ``animations`` frees nothing on its own: its
        accessors stay in the array, and an accessor keeps its bufferView alive
        whether or not anything still reads the accessor. This drops the ones
        no surviving mesh, skin or clip references, renumbers what survives,
        and :meth:`compact_bin` then collects the payload.

        Takes them all at once so the collector runs ONCE: it rebuilds the
        whole BIN, so calling it per clip would copy a hundreds-of-megabyte
        buffer once per clip to reclaim one clip's worth each time.

        Parameters:
            edit: An open session; the caller must ALREADY have removed the
                animations from ``gltf["animations"]``, since what is still
                referenced is read off the file rather than passed in.
            animations: The removed clip, or clips.

        Returns:
            Bytes of BIN payload reclaimed; 0 when the file uses an extension
            that may hold accessor indices of its own, where the pass declines
            rather than guess -- renumbering under one of those would silently
            re-point it at the wrong data.
        """
        if isinstance(animations, dict):
            animations = [animations]
        mine = set()
        for animation in animations:
            for sampler in (animation or {}).get("samplers") or []:
                for key in ("input", "output"):
                    mine.add((sampler or {}).get(key))
        if not cls.drop_orphaned_accessors(edit, mine):
            return 0
        return cls.compact_bin(edit)
