# !/usr/bin/python
# coding=utf-8
import base64
import copy
import hashlib
import io
import json
import logging
import os
import platform as _platform
import shlex
import shutil
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from pythontk.core_utils.help_mixin import HelpMixin

logger = logging.getLogger(__name__)

#: What every GLB repair on :class:`MeshConvert` accepts as its target: a path,
#: or an already-open :class:`MeshConvert.GlbEdit` to join. The forward
#: reference is never resolved at runtime -- it exists so the signatures say
#: which of the two they take.
GlbTarget = Union[str, os.PathLike, "MeshConvert.GlbEdit"]

# godotengine/FBX2glTF — single-binary FBX -> glTF/GLB converter, the same
# tool Godot 4 uses internally for FBX import. Pinned to v0.13.1.
FBX2GLTF_VERSION = "0.13.1"
FBX2GLTF_PLATFORMS = {
    "windows": {
        "url": f"https://github.com/godotengine/FBX2glTF/releases/download/v{FBX2GLTF_VERSION}/FBX2glTF-windows-x86_64.zip",
        "type": "zip",
        "executable": "FBX2glTF-windows-x86_64",
    },
    "linux": {
        "url": f"https://github.com/godotengine/FBX2glTF/releases/download/v{FBX2GLTF_VERSION}/FBX2glTF-linux-x86_64.zip",
        "type": "zip",
        "executable": "FBX2glTF-linux-x86_64",
    },
    "darwin": {
        "url": f"https://github.com/godotengine/FBX2glTF/releases/download/v{FBX2GLTF_VERSION}/FBX2glTF-macos-x86_64.zip",
        "type": "zip",
        "executable": "FBX2glTF-macos-x86_64",
    },
}


class MeshConvert(HelpMixin):
    """3D mesh format conversion via the godotengine/FBX2glTF CLI.

    Currently supports static-mesh FBX -> GLB (binary glTF 2.0).
    The FBX2glTF binary is fetched on first use into the pythontk-managed
    tools directory under ``~/.pythontk/tools/`` (overridable via
    ``PYTHONTK_TOOLS_DIR``).

    Note: godotengine/FBX2glTF only ships an x86_64 build for macOS.
    Apple Silicon (arm64) Macs run it transparently via Rosetta 2,
    which must be installed (``softwareupdate --install-rosetta``).
    """

    TOOL_NAME = "fbx2gltf"
    DEFAULT_TIMEOUT = 300  # 5 minutes — enough for very large FBX files

    #: Schema version of the scene-sidecar envelope
    #: (:meth:`build_scene_sidecar`). Bump on any change to the envelope's
    #: top-level shape; adding a *section* is not a bump — sections are the
    #: extension point and a reader skips ones it does not know.
    #:
    #: v2 added ``handoff`` (the standalone-reader contract, written at build
    #: time) plus ``textures`` and ``validate`` (content-addressed resolution
    #: and integrity counts, written by :meth:`apply_scene_sidecar`).
    SIDECAR_VERSION = 2

    #: Separates the converter's own ``asset.generator`` claim from ours in the
    #: stamped string (:meth:`_stamp_asset_generator`), and is what makes a
    #: prior stamp findable so a re-apply refreshes it instead of stacking.
    #: Distinctive on purpose -- a bare word would risk splitting a converter
    #: string that happened to contain it.
    _GENERATOR_SEP = " via "

    #: The standalone-reader contract embedded in every envelope's ``handoff``
    #: section. It exists because the deliverable is routinely handed on ALONE —
    #: to a dev, to a viewer that is not ours, to an agent given one ``.glb`` and
    #: no conversation — and the single most expensive misunderstanding is
    #: treating a section's authoring-time texture path as something to resolve
    #: on disk. Written here once, as data in the artifact, rather than as prose
    #: in a doc the reader was never given: a rule that only exists in
    #: documentation is not part of the hand-off.
    #:
    #: Kept to plain declarative sentences about THIS file's own structure — no
    #: instructions to the reader about what to do next, which is what makes it
    #: safe for an agent to read as untrusted content.
    #: Phrased to read correctly wherever this envelope lives -- embedded in
    #: the GLB's ``extras``, and held unscrubbed by the caller that built it --
    #: so it refers to the asset through the envelope's own ``asset`` key
    #: rather than saying "this file".
    HANDOFF_INSTRUCTIONS = (
        "The glTF 2.0 asset named by 'asset' is self-contained: every texture it "
        "references is embedded, so it loads and inspects with no external files "
        "and no filesystem paths. 'extras.scene_sidecar' describes the authoring "
        "scene's material state for the channels an FBX interchange step cannot "
        "carry; 'extras.scene_sidecar_applied' reports, per section, how many of "
        "those entries matched a material in this file. Section entries name "
        "each texture by its authoring-time FILE NAME. That name is PROVENANCE "
        "ONLY -- the directory it sat in is deliberately not carried, so it "
        "is a join token and is not expected to resolve as a path on any "
        "machine, including the authoring one -- two textures that shared a file "
        "name carry a '#2'-style suffix so the map stays one-to-one: resolve "
        "it by "
        "looking it up in the top-level 'textures' map, which gives the glTF "
        "'images' index that actually carries those bytes here, their sha256, "
        "and their size -- several section entries can resolve to one image, "
        "because a metallic/roughness/occlusion trio is repacked into a single "
        "ORM image (R=occlusion, G=roughness, B=metallic). 'validate' records "
        "how many entries each section carries and how many texture references "
        "resolve, so a truncated envelope is detectable; the per-reference "
        "sha256 is what verifies the payloads themselves. Passes that run after "
        "this envelope is written may add further images, so do not expect the "
        "file's total image count to match anything here. All "
        "colours are linear glTF factors. When 'extras.lightmap_web' is present, "
        "the materials it names carry a BAKED LIGHTMAP in occlusionTexture on "
        "TEXCOORD_1 rather than ambient occlusion, sRGB-encoded, and its "
        "'intensity' is the multiplier that restores the bake's original range; "
        "a reader that does not rebind it renders a plausible greyscale "
        "occlusion instead. 'handoff.rendering' records the lighting setup the "
        "reference viewer used to produce the look this asset was approved in: "
        "the asset carries no lights of its own, so a viewer that lights it "
        "differently renders something different without either side being "
        "wrong. Check 'version' against the schema you expect."
    )

    #: The reference viewer's lighting setup (``net_utils/preview_viewer.html``),
    #: published as data so a recipient can reproduce the look the asset was
    #: signed off in instead of inferring it. A baked asset needs this more than
    #: an unbaked one, not less: its lighting is already in its textures, so the
    #: viewer's own rig has to be *withheld* by exactly the right amount, and
    #: every number below is a measured compromise rather than a default (see the
    #: viewer's own comments for what each one is compensating for). Without it a
    #: recipient reasonably adds a normal key light and blows out every baked
    #: surface -- which reads as a bake regression and sends them back to the
    #: baker, where nothing is wrong.
    #:
    #: ``test_preview_server`` pins these against the viewer's literals, so the
    #: published contract cannot drift from what the viewer actually does.
    RENDERING_POLICY: Dict[str, Any] = {
        "renderer": {
            "toneMapping": "ACESFilmic",
            "toneMappingExposure": 1.0,
            "outputColorSpace": "srgb",
        },
        "environment": {
            "source": "three.js RoomEnvironment",
            "prefilter": "PMREM",
            "prefilterBlur": 0.04,
            "intensity": 1.0,
            "note": (
                "The main light. Kept non-zero even for a fully baked asset: "
                "lightmap irradiance carries no normal term, so with the "
                "environment off nothing left in the render samples the normal "
                "and normal maps, roughness and speculars all go inert."
            ),
        },
        "keyLight": {
            "type": "directional",
            "color": "#ffffff",
            "intensity": 0.9,
            "position": [3, 6, 4],
            "disabled_when": "any material in the asset is lightmapped",
            "note": (
                "Scene-wide, so it cannot be spared the baked geometry it would "
                "contradict; it goes off entirely once anything is baked."
            ),
        },
        "lightmappedMaterials": {
            "envMapIntensity": 0.25,
            "lightMapIntensity": (
                "per material, from extras.lightmap_web.materials[<name>].intensity"
            ),
            "note": (
                "Applied per material, not scene-wide: only a material that "
                "carries a bake has its lighting already in it. Un-baked props in "
                "the same asset keep the full environment (1.0)."
            ),
        },
    }

    #: Default worker cap for :meth:`optimize_glb_textures`' encode pass, capped
    #: again by the core count at call time. Deliberately well below a modern
    #: core count -- see the phase-B comment there: the ceiling is host memory
    #: (each worker holds a fully decoded 4K source), not cores, because this
    #: routinely runs inside a DCC already holding the exported scene.
    OPTIMIZE_WORKERS = 8

    #: Slot semantic -> (Basis codec, sRGB transfer) for KTX2 mode. The glTF
    #: structural twin of ``MapOptimizer.resolve_compression``'s registry rule:
    #: ETC1S only where a lossy codec is safe (perceptual sRGB color), UASTC
    #: for normals and linear data, and -- the ``None`` row -- UASTC + sRGB for
    #: an image nothing samples, where a mislabel must at least not band it.
    BASIS_BY_SEMANTIC: Dict[Optional[str], Tuple[str, bool]] = {
        "color": ("ETC1S", True),
        "data": ("UASTC", False),
        "normal": ("UASTC", False),
        None: ("UASTC", True),
    }

    #: Semantic precedence when one image is sampled by several slots: the
    #: quality/correctness-critical use wins the encode.
    _SEMANTIC_RANK: Dict[str, int] = {"color": 0, "data": 1, "normal": 2}

    #: Extensions that reference ``images`` from outside the material tree
    #: (root-level ``specularImages`` here). :meth:`prune_glb_unreferenced_textures`
    #: only follows material -> texture -> image, so a file declaring one of
    #: these is left alone rather than renumbered under it.
    _IMAGE_REFERRING_EXTENSIONS = frozenset({"EXT_lights_image_based"})

    #: Sidecar section -> the writer that applies it to a GLB, in application
    #: order. This is the applier column of the scene-data grid: a new kind of
    #: extended scene setup is one more section in the DCC-side reader
    #: (mayatk/blendertk ``SceneState``) plus one more row here — no method
    #: edits anywhere else. Both current rows exist because FBX loses the
    #: channel for every shader that is not the host's own legacy model
    #: (measured on Maya 2025: an aiStandardSurface arrives with no emissive
    #: at all and a flat white base colour). Base colour runs first only for
    #: tidiness; the two touch disjoint fields.
    SIDECAR_APPLIERS: Dict[str, str] = {
        "base_color": "set_glb_base_color",
        "emissive": "set_glb_emissive",
        "metallic_roughness": "set_glb_metallic_roughness",
    }
    # Image types glTF 2.0 accepts natively. Anything else (TIFF, EXR, TGA —
    # all common in a DCC source tree) is re-encoded to PNG via Pillow when
    # available (see `_reencode_as_png`), and otherwise rejected by name
    # rather than written as an unloadable data URI.
    IMAGE_MIME_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }

    # ------------------------------------------------------------------ #
    # The open-GLB handle every repair below operates on
    # ------------------------------------------------------------------ #

    class GlbEdit:
        """A GLB parsed once and held open: read the JSON, edit, write once.

        Every repair on :class:`MeshConvert` edits the JSON chunk and nothing
        else, but each used to open, parse and rewrite the file for itself --
        and the preview path runs three of them back to back on the same GLB.
        A single push therefore read a file the size of its geometry three
        times over to change a handful of fields. This is the shared handle
        that collapses that to one read and one write; see
        :meth:`MeshConvert.open_glb` for how a caller joins several repairs
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
        #: target for the in-place write in :meth:`MeshConvert._write_glb`.
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
            #: :meth:`MeshConvert._relocate_embedded_images` when the session
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
            :meth:`optimize_glb_textures` a texture carries
            ``EXT_texture_webp`` beside its fallback ``source``, or (KTX2
            mode) ``KHR_texture_basisu`` beside a core-readable fallback
            twin (none in pure-delivery mode) -- either
            way the extension is what a capable loader reads. Bounds-checked
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
            for binding in ("KHR_texture_basisu", "EXT_texture_webp"):
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
                                    result = (
                                        im.convert("RGBA").getchannel("A").getextrema()
                                    )
                            else:
                                result = (
                                    im.convert("RGB").getchannel(channel).getextrema()
                                )
                    except Exception as exc:  # noqa: BLE001 — varied decoder errors
                        logger.debug(
                            "GLB channel probe: skipped image %s (%s) [%s]",
                            img_idx,
                            exc,
                            channel,
                        )
            self._channel_extrema[key] = result
            return result

    @classmethod
    def _platform_exe_name(cls) -> str:
        """Return the FBX2glTF binary name for the current platform."""
        plat = _platform.system().lower()
        info = FBX2GLTF_PLATFORMS.get(plat)
        if not info:
            raise LookupError(f"FBX2glTF: unsupported platform '{plat}'")
        return info["executable"]

    @classmethod
    def resolve_binary(
        cls,
        required: bool = True,
        auto_install: bool = False,
        prompt: bool = True,
    ) -> Optional[str]:
        """Resolve the FBX2glTF executable from PATH or managed installs.

        Parameters:
            required:      Raise FileNotFoundError when missing.
            auto_install:  Download FBX2glTF if not found.
            prompt:        Ask before downloading (TTY only; non-TTY proceeds).

        Returns:
            Absolute path to FBX2glTF executable, or None.
        """
        platform_exe = cls._platform_exe_name()
        # Try platform-specific binary name first (matches release zip),
        # then plain "FBX2glTF" for users who renamed it.
        for candidate in (platform_exe, "FBX2glTF"):
            on_path = shutil.which(candidate)
            if on_path:
                return on_path

        from pythontk.core_utils.app_installer import AppInstaller

        managed = AppInstaller.get_path(
            cls.TOOL_NAME, executable=platform_exe, add_to_path=True
        )
        if managed:
            return managed

        if not auto_install:
            if required:
                raise FileNotFoundError(
                    f"FBX2glTF not found on PATH (looked for {platform_exe!r}). "
                    "Pass auto_install=True to download it."
                )
            return None

        if prompt:
            if not (sys.stdin and sys.stdin.isatty()):
                # No interactive console (CI, GUI host, pythonw.exe, etc.).
                # Refuse to silently download — caller must opt-in via prompt=False.
                if required:
                    raise FileNotFoundError(
                        "FBX2glTF is not installed and no interactive console "
                        "is available to confirm the download. Pass "
                        "prompt=False to install non-interactively."
                    )
                return None
            sys.stdout.write(
                f"\nFBX2glTF v{FBX2GLTF_VERSION} is not installed. "
                f"Download to ~/.pythontk/tools/ now? [y/N] "
            )
            sys.stdout.flush()
            answer = sys.stdin.readline().strip().lower()
            if answer not in ("y", "yes"):
                if required:
                    raise FileNotFoundError("User declined FBX2glTF installation.")
                return None

        try:
            return AppInstaller.ensure(
                cls.TOOL_NAME,
                platforms=FBX2GLTF_PLATFORMS,
                executable=platform_exe,
                version=FBX2GLTF_VERSION,
            )
        except (RuntimeError, OSError, LookupError) as exc:
            if required:
                raise
            logger.warning("FBX2glTF install failed: %s", exc)
            return None

    @classmethod
    def fbx_to_glb(
        cls,
        src: str,
        dst: Optional[str] = None,
        *,
        overwrite: bool = False,
        auto_install: bool = True,
        prompt: bool = True,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        extra_args: Optional[List[str]] = None,
        sidecar: Optional[Dict[str, Any]] = None,
        lightmaps: bool = True,
    ) -> str:
        """Convert an FBX file to a binary glTF 2.0 (GLB) file.

        Parameters:
            src:           Input FBX path.
            dst:           Output GLB path. Defaults to src with .glb extension.
                           ``.glb`` is appended if absent.
            overwrite:     Replace existing destination.
            auto_install:  Download FBX2glTF if missing.
            prompt:        Ask before downloading.
            timeout:       Subprocess timeout in seconds. None disables.
            extra_args:    Extra CLI flags forwarded to FBX2glTF
                           (e.g. ``["--draco"]``, ``["-v"]``).
            sidecar:       A scene-sidecar envelope (:meth:`build_scene_sidecar`)
                           to apply to and embed in the converted GLB — the one
                           parameter that turns a bare conversion into a
                           scene-faithful deliverable. Applied inside the same
                           post-conversion edit session as the alpha repair, so
                           it costs no extra file pass. Callers that need the
                           per-section outcome summary call
                           :meth:`apply_scene_sidecar` separately instead.
            lightmaps:     Wire the host scene's committed lightmaps into the
                           GLB (:meth:`apply_glb_lightmaps`). Default on and
                           self-feeding -- the manifest travels inside the FBX,
                           so a scene with no committed bake is a clean no-op
                           and callers pass nothing.

        Returns:
            Absolute path to the written GLB file.
        """
        src_abs = os.path.abspath(src)
        if not os.path.isfile(src_abs):
            raise FileNotFoundError(f"FBX source not found: {src_abs}")
        if os.path.splitext(src_abs)[1].lower() != ".fbx":
            raise ValueError(f"Expected .fbx input, got: {src_abs}")

        if dst is None:
            dst = os.path.splitext(src_abs)[0] + ".glb"
        elif not dst.lower().endswith(".glb"):
            dst = dst + ".glb"
        dst_abs = os.path.abspath(dst)

        if os.path.exists(dst_abs):
            if not overwrite:
                raise FileExistsError(
                    f"GLB output already exists: {dst_abs}. "
                    "Pass overwrite=True to replace."
                )
            os.remove(dst_abs)

        os.makedirs(os.path.dirname(dst_abs) or ".", exist_ok=True)

        binary = cls.resolve_binary(
            required=True, auto_install=auto_install, prompt=prompt
        )

        # FBX2glTF wants the output base WITHOUT extension; --binary forces .glb.
        # --user-properties copies FBX user properties into per-node glTF
        # ``extras`` (measured against v0.13.1 + Maya 2025: the DataNodes
        # ``data_export`` channels arrive with it and are silently dropped
        # without). A carrier must not drop data another carrier deliberately
        # embedded, and the flag is a no-op on an FBX with no user properties.
        output_base = os.path.splitext(dst_abs)[0]
        cmd = [
            binary,
            "-i",
            src_abs,
            "-o",
            output_base,
            "--binary",
            "--user-properties",
        ]
        if extra_args:
            cmd.extend(extra_args)

        logger.debug("FBX2glTF: %s", shlex.join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"FBX2glTF timed out after {timeout}s converting {src_abs}"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"FBX2glTF failed (exit={result.returncode}):\n"
                f"  cmd: {shlex.join(cmd)}\n"
                f"  stdout: {result.stdout}\n"
                f"  stderr: {result.stderr}"
            )
        if not os.path.isfile(dst_abs):
            raise RuntimeError(
                f"FBX2glTF exited 0 but {dst_abs} was not created.\n"
                f"  stdout: {result.stdout}"
            )

        # One post-conversion edit session for everything that touches the
        # JSON chunk: the alpha repair and (when given) the scene sidecar.
        # The alpha repair keeps its own guard so its failure never costs the
        # sidecar, and neither ever costs the successful conversion.
        try:
            with cls.open_glb(dst_abs) as edit:
                try:
                    fixes = cls.fix_glb_phantom_opaque_alpha(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("fix_glb_phantom_opaque_alpha skipped: %s", exc)
                    fixes = []
                for fx in fixes:
                    logger.info(
                        "fix_glb_phantom_opaque_alpha: %s baseColorFactor[3] %.3f -> %.3f (image: %s)",
                        fx["material"],
                        fx["old_alpha"],
                        fx["new_alpha"],
                        fx["image"],
                    )
                if sidecar:
                    cls.apply_scene_sidecar(edit, sidecar)
                if lightmaps:
                    # Guarded like the alpha repair: a lightmap failure must
                    # never cost the sidecar or the conversion.
                    try:
                        bound = cls.apply_glb_lightmaps(edit)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("GLB lightmaps skipped: %s", exc)
                    else:
                        if bound:
                            logger.info(
                                "Lightmaps wired into %d material binding(s).",
                                len(bound),
                            )
        except Exception as exc:  # noqa: BLE001 — never let post-process kill a successful conversion
            logger.warning("GLB post-process skipped: %s", exc)

        return dst_abs

    # ------------------------------------------------------------------ #
    # Scene sidecar — the envelope carrying what FBX translation drops
    # ------------------------------------------------------------------ #

    @classmethod
    def build_scene_sidecar(
        cls,
        sections: Optional[Dict[str, Any]],
        source: Dict[str, str],
        asset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Wrap *sections* in the versioned scene-sidecar envelope.

        The one place the envelope schema exists — every producer (the WebXR
        preview bridges, the scene exporters' GLB tasks) builds through this
        rather than shaping the dict itself, so the schema cannot fork between
        packages. The top level is the frozen contract a standalone reader (a
        dev tool holding only the deliverable and this data) parses against::

            {
              "version": 2,              # SIDECAR_VERSION
              "source": {"application": "maya", "version": "2025"},
              "asset": "<payload basename>",
              "color_space": "linear",   # every color below, as the appliers
                                         #   also expect (glTF factors)
              "sections": {...},         # the extension point
              "handoff": {               # the standalone-reader contract
                "instructions": "<HANDOFF_INSTRUCTIONS>",
                "reads": {...}           # extras key -> what it holds
              }
            }

        :meth:`apply_scene_sidecar` adds two more top-level keys when it embeds
        the envelope, because only it can know them: ``textures`` (each
        authoring path -> the glTF image index and sha256 actually carrying it)
        and ``validate`` (the counts the envelope was written against).

        Scope note — this is *not* a second channel registry beside the DCC
        packages' ``DataNodes``. Tool-authored semantic metadata (shots, audio
        events, lightmap manifests, ...) rides **inside** the FBX as user
        properties on the ``data_export`` carrier; the sidecar carries only
        repairs for what the FBX *format* mistranslates about the scene's
        literal content, derived scene-read-only at push/export time. A
        section must never duplicate a ``DataNodes`` channel — one home per
        section per deliverable.

        Parameters:
            sections: ``{section: data}`` from a DCC-side ``SceneState``
                reader. ``None`` or empty still builds an envelope — an empty
                ``sections`` is itself information (nothing needed repair).
            source: Producer identity, e.g.
                ``{"application": "maya", "version": "2025"}``.
            asset: Basename of the deliverable this envelope belongs to.
        """
        return {
            "version": cls.SIDECAR_VERSION,
            "source": source,
            "asset": asset,
            "color_space": "linear",
            "sections": dict(sections or {}),
            "handoff": {
                "instructions": cls.HANDOFF_INSTRUCTIONS,
                # Where the rest of the self-description lives. Derived from the
                # registries rather than listed by hand, so a section or a
                # manifest added later cannot silently fall out of the contract.
                "reads": {
                    "extras.scene_sidecar": "this envelope",
                    "extras.scene_sidecar_applied": "per-section apply outcome",
                    f"extras.{cls.LIGHTMAP_WEB_KEY}": (
                        "materials whose occlusion carrier is a baked lightmap"
                    ),
                },
                "sections": sorted(cls.SIDECAR_APPLIERS),
                # How to LIGHT what the sections describe. The rest of this
                # envelope says what the asset is; without this, a recipient can
                # rebuild every material correctly and still not reproduce the
                # look, because the lighting lives in the viewer rather than in
                # the file.
                "rendering": copy.deepcopy(cls.RENDERING_POLICY),
            },
        }

    @classmethod
    def apply_scene_sidecar(
        cls, glb: GlbTarget, sidecar: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Apply a scene-sidecar envelope to a GLB and embed it in its extras.

        Every section is dispatched through :attr:`SIDECAR_APPLIERS` against
        **one** open edit session, then the envelope itself (plus the
        per-section outcome summary) is written into the glTF root ``extras``
        — so the artifact leaves self-describing: what the scene authored
        (``extras["scene_sidecar"]``) and what this pass did about it
        (``extras["scene_sidecar_applied"]``), readable by any glTF tool with
        no side files. :meth:`read_scene_sidecar` is the counterpart.

        A section absent from the envelope is simply skipped, and a section
        failure is logged rather than raised: a deliverable missing one
        section still beats no deliverable. Every failure an applier can
        actually reach (an unreadable texture, an image no decoder wants) is
        handled inside it and reported as a skipped image; the per-section
        catch here is the backstop for anything that is not, and the
        container-level catch reports *every* offered section as failed
        rather than letting sections that "applied" claim a success that
        never reached disk.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            sidecar: The envelope (:meth:`build_scene_sidecar`). ``None`` (or
                anything falsy) is a true no-op; an envelope with empty
                *sections* still embeds (sidecar was on, the scene had nothing
                to carry) — the distinction is a real envelope vs no envelope.

        Returns:
            ``{section: outcome}`` — ``"N of M"``, ``"0 of M matched"`` or
            ``"failed (...)"`` per offered section; empty when none offered.
        """
        if not sidecar:
            return {}
        sections = sidecar.get("sections") or {}
        summary: Dict[str, str] = {}
        try:
            with cls.open_glb(glb) as edit:
                for section, method in cls.SIDECAR_APPLIERS.items():
                    data = sections.get(section)
                    if not data:
                        continue
                    apply = getattr(cls, method)
                    try:
                        applied = apply(edit, data)
                    except (OSError, ValueError) as error:
                        logger.warning("Sidecar %r not applied: %s", section, error)
                        summary[section] = f"failed ({error})"
                        continue
                    if not applied:
                        # The section was read but nothing landed — almost
                        # always a name mismatch, which the applier has just
                        # logged in full.
                        logger.warning(
                            "Sidecar %r matched none of its %s entr(ies) in the GLB.",
                            section,
                            len(data),
                        )
                        summary[section] = f"0 of {len(data)} matched"
                        continue
                    logger.info("Sidecar %r applied to %s.", section, len(applied))
                    summary[section] = f"{len(applied)} of {len(data)}"
                # Sweep what the writers above unbound (FBX2glTF's converted
                # ORM after the repack, most often) BEFORE the texture map
                # below records image indices -- pruning renumbers them.
                cls.prune_glb_unreferenced_textures(edit)
                extras = edit.gltf.setdefault("extras", {})
                # A COPY, with the resolution keys added: the caller's envelope
                # is theirs to keep (unscrubbed, useful on the authoring machine),
                # and only this pass can know which glTF image each authoring
                # path became.
                embedded = dict(sidecar)
                embedded["textures"] = cls._sidecar_texture_map(edit)
                # Deliberately NOT the file's total image/texture counts. Later
                # passes add images -- the lightmap applier runs after this in
                # every production path -- so a total stamped here is stale on
                # arrival, and a reader using it as an integrity check would
                # reject every lightmapped deliverable. What is stable is what
                # this envelope itself claims: how many entries each section
                # carries and how many texture references resolve. Those, plus
                # the per-reference digests, are the check worth making.
                # Ship file names, not the authoring machine's directory tree.
                # Runs BEFORE `validate` is sized and before the digests are
                # stamped, so both describe the envelope as delivered.
                cls._scrub_sidecar_paths(embedded)
                embedded["validate"] = {
                    "sections": {
                        name: len(entries)
                        for name, entries in sections.items()
                        # A malformed envelope (a null section, say) is skipped
                        # by the dispatch above; sizing it must not be the thing
                        # that raises -- the container catch here only handles
                        # OSError/ValueError, so a TypeError would abort a whole
                        # apply that was otherwise fine.
                        if hasattr(entries, "__len__")
                    },
                    "textures": len(embedded["textures"]),
                }
                extras["scene_sidecar"] = embedded
                extras["scene_sidecar_applied"] = dict(summary)
                cls._stamp_asset_generator(edit, sidecar)
                # The materials this envelope did NOT cover still carry the
                # converter's own packing, which is the one measured to arrive
                # solid white. Checked here because the session is already open
                # and the covered names are in hand; a fully-covered export
                # decodes nothing.
                #
                # Only the DESTRUCTIVE finding is raised at export time. The
                # `unvalidated` half is real coverage information but it is not
                # a defect on its own, and an export-time warning an artist
                # cannot act on is how the actionable ones stop being read --
                # it is reported to the recipient instead, by `verify_glb`.
                suspect = cls.suspect_orm_materials(
                    edit, described=sections.get("metallic_roughness") or ()
                )
                harmful = sorted(
                    name
                    for name, found in suspect.items()
                    if found["finding"] == cls.ORM_FINDING_METALLIC_FULL
                )
                if harmful:
                    logger.warning(
                        "Metallic=1 everywhere on %s material(s) not covered by "
                        "the sidecar: %s. glTF reads metallic from the ORM's blue "
                        "channel, so these render with no diffuse response (black "
                        "under a lightmap) -- name them in the scene sidecar's "
                        "metallic_roughness section, or re-export their source "
                        "maps as RGB.",
                        len(harmful),
                        ", ".join(harmful),
                        extra={"preset": "highlight"},
                    )
                # Every reference is meant to carry a content address; one that
                # cannot be digested is an entry a verifying reader has no way
                # to check, so a shortfall is said out loud rather than shipped
                # as a quietly incomplete map.
                stamped = cls._stamp_sidecar_digests(edit)
                if stamped != len(embedded["textures"]):
                    logger.warning(
                        "Sidecar: %d of %d texture reference(s) could not be "
                        "content-addressed (their image payload was unreadable).",
                        len(embedded["textures"]) - stamped,
                        len(embedded["textures"]),
                    )
                edit.dirty = True
        except (OSError, ValueError) as error:
            logger.warning("Sidecar not applied to %s: %s", glb, error)
            return {
                section: f"failed ({error})"
                for section in cls.SIDECAR_APPLIERS
                if sections.get(section)
            }
        return summary

    @classmethod
    def _stamp_asset_generator(
        cls, edit: "MeshConvert.GlbEdit", sidecar: Optional[Dict[str, Any]] = None
    ) -> str:
        """Name this pipeline in the glTF's own ``asset.generator``; return it.

        The one provenance field glTF itself defines, and every viewer and
        inspector already displays -- so it reaches a recipient who opens the
        file in a tool that is not ours and reads nothing else we write. The
        converter's own string is kept and ours appended: FBX2glTF really did
        produce the geometry, and replacing that claim would lose the fact that
        matters most when a mesh arrives wrong.

        Deliberately coarse -- the authoring app + version the envelope already
        names, and this package's version. No host, no user, no paths: a
        generator string travels to whoever gets the file (see
        :meth:`_scrub_sidecar_paths` for the same rule applied to the envelope).

        Idempotent: re-applying an envelope to an already-stamped GLB refreshes
        the stamp rather than appending a second one.
        """
        parts = []
        source = (sidecar or {}).get("source") or {}
        # Joined from what is actually present: an f-string over a missing
        # version interpolates the literal "None" into the MIDDLE of the string,
        # where .strip() cannot reach it ("maya None + pythontk 0.9.24"), and a
        # version naming no application used to be dropped outright.
        named = " ".join(
            str(v) for v in (source.get("application"), source.get("version")) if v
        )
        if named:
            parts.append(named)
        try:  # Deferred: the root package imports this module.
            from pythontk import __version__ as ptk_version

            parts.append(f"pythontk {ptk_version}")
        except ImportError:  # pragma: no cover - a broken install, not a case
            pass
        ours = " + ".join(parts)
        asset = edit.gltf.setdefault("asset", {})
        # Drop a stamp from an earlier run before appending this one, so the
        # string stays "<converter> via <ours>" however many passes touch it.
        prior = (asset.get("generator") or "").split(cls._GENERATOR_SEP)[0].strip()
        # A file whose ONLY generator claim is a previous stamp of ours has no
        # separator to split on, so the whole string would come back as "prior"
        # and the next pass would append ours to itself. Our stamp always names
        # this package, which is what makes it recognisable across versions.
        if "pythontk " in prior:
            prior = ""
        asset["generator"] = f"{prior}{cls._GENERATOR_SEP}{ours}" if prior else ours
        edit.dirty = True
        return asset["generator"]

    @classmethod
    def _scrub_sidecar_paths(cls, embedded: Dict[str, Any]) -> int:
        """Strip authoring directories out of *embedded* in place; return the count.

        The envelope names every texture by the absolute path it had on the
        authoring machine -- in both the ``textures`` map's keys and the section
        entries that resolve through them. That is fine locally and is a
        disclosure in a hand-off: a GLB sent to an external developer spells out
        the client folder tree it was built from.

        The directory is what leaks; the file name is not, and the join does not
        need either. Both sides of the join are the SAME string, so rewriting
        both consistently keeps a plain string lookup working -- no schema
        version bump, and every existing reader keeps resolving. (The content
        address stamped by :meth:`_stamp_sidecar_digests` is the identity that
        actually verifies bytes; this name is only a join token.)

        Two distinct paths CAN share a basename, so a collision is disambiguated
        with a ``#N`` suffix assigned in sorted-path order -- deterministic, and
        it keeps the map injective, which a bare basename would not.

        Only the embedded COPY is scrubbed. The caller's own envelope keeps
        full provenance, which is the half that is useful on the authoring
        machine.
        """
        entries = embedded.get("textures")
        if not isinstance(entries, dict):
            return 0

        mapping: Dict[str, str] = {}
        claimed: Dict[str, str] = {}  # scrubbed name -> the path that took it
        for path in sorted(entries):
            if not isinstance(path, str):
                continue
            name = os.path.basename(path.replace("\\", "/").rstrip("/")) or path
            if claimed.get(name, path) != path:
                stem, dot, ext = name.rpartition(".")
                suffix = 2
                while True:
                    candidate = (
                        "{}#{}{}{}".format(stem, suffix, dot, ext)
                        if dot
                        else "{}#{}".format(name, suffix)
                    )
                    if claimed.get(candidate, path) == path:
                        break
                    suffix += 1
                name = candidate
            claimed[name] = path
            if name != path:
                mapping[path] = name

        if not mapping:
            return 0

        def rewrite(node):
            """Replace any string that IS one of the mapped paths, anywhere."""
            if isinstance(node, dict):
                return {
                    mapping.get(k, k) if isinstance(k, str) else k: rewrite(v)
                    for k, v in node.items()
                }
            if isinstance(node, list):
                return [rewrite(v) for v in node]
            if isinstance(node, str):
                return mapping.get(node, node)
            return node

        for key, value in list(embedded.items()):
            embedded[key] = rewrite(value)
        return len(mapping)

    @classmethod
    def _sidecar_texture_map(
        cls, edit: "MeshConvert.GlbEdit"
    ) -> Dict[str, Dict[str, Any]]:
        """``{authoring path: {"image": index}}`` for everything this session embedded.

        The join a standalone reader needs and a path cannot give it: a section
        entry names a texture by the path it had on the authoring machine, which
        resolves nowhere else, so the envelope carries the map from that name to
        the glTF ``images`` index actually holding those bytes.
        :meth:`_stamp_sidecar_digests` fills in the content address.

        Derived wholly from :attr:`GlbEdit.embedded` (source key -> texture
        index), which is why it needs no knowledge of any section's shape and a
        section added later is covered for free. Two wrinkles that shape it:

        * The ORM writer's cache key is the three source paths joined by ``|``,
          because the trio collapses into ONE packed image. Splitting it back
          out is what lets each of ``metallic``/``roughness``/``occlusion``
          resolve -- all three to the same index, which is the truth.
        * ``None`` appears in that join for a slot with no map (``"a|None|b"``),
          and is not a path. A real file named ``None`` is indistinguishable
          here, which costs nothing: it would resolve to the image it is
          actually part of.
        """
        resolved: Dict[str, Dict[str, Any]] = {}
        for cache_key, texture_index in edit.embedded.items():
            image_index = edit.image_for_texture(texture_index)
            if image_index is None:
                continue
            for path in str(cache_key).split("|"):
                if path and path != "None":
                    # setdefault, not assignment: one path CAN legitimately
                    # resolve to two images -- embedded whole for one channel
                    # and folded into an ORM for another -- and a single-valued
                    # map can only say one. First wins, which is well defined
                    # because SIDECAR_APPLIERS fixes the order and the whole
                    # embed is the more direct answer than a channel of a pack.
                    # Assignment made the winner depend on applier order, which
                    # is the kind of thing that changes silently.
                    resolved.setdefault(path, {"image": image_index})
        return resolved

    @classmethod
    def _stamp_sidecar_digests(cls, edit: "MeshConvert.GlbEdit") -> int:
        """Refresh the embedded envelope's content addresses; return how many.

        Split from :meth:`_sidecar_texture_map` because the two answer questions
        with different lifetimes. *Which* image carries a path is decided once,
        when the sidecar is applied, and never changes -- image indices are
        stable. *What those bytes are* changes afterwards:
        :meth:`optimize_glb_textures` resizes and re-encodes every image it
        touches, so a digest taken at apply time describes bytes the delivered
        file no longer contains. So the digest is stamped by whoever last wrote
        the payloads, and this is idempotent so both callers can.

        Reads the envelope out of ``extras`` rather than taking it as an
        argument: the optimize pass has no sidecar in hand and must not need
        one -- it just refreshes whatever the file already declares.
        """
        sidecar = (edit.gltf.get("extras") or {}).get("scene_sidecar")
        entries = sidecar.get("textures") if isinstance(sidecar, dict) else None
        if not isinstance(entries, dict):
            return 0
        images = edit.images
        stamped = 0
        for ref in entries.values():
            if not isinstance(ref, dict):
                continue
            index = ref.get("image")
            if not isinstance(index, int) or not 0 <= index < len(images):
                continue
            payload = edit._image_payload(images[index])
            if not payload:
                continue
            ref["sha256"] = hashlib.sha256(payload).hexdigest()
            ref["bytes"] = len(payload)
            mime = images[index].get("mimeType")
            if mime:
                ref["mimeType"] = mime
            stamped += 1
        if stamped:
            edit.dirty = True
        return stamped

    @classmethod
    def sidecar_foreign_packings(
        cls,
        sidecar: Optional[Dict[str, Any]],
        target: str = "ORM",
        workflow: Optional[str] = None,
    ) -> Dict[str, str]:
        """``{path: map type}`` for envelope textures authored for another engine.

        The pre-flight half of the compatibility story: given a built scene
        sidecar, which of the maps it is about to carry into a glTF deliverable
        are packed for a *different* engine family? Answers before any
        conversion runs, so an exporter can gate on it (see mayatk's /
        blendertk's ``check_material_compatibility``) rather than discovering it
        in the log afterwards.

        Lives here because :meth:`build_scene_sidecar` owns the envelope schema
        and mayatk/blendertk cannot import each other -- written per DCC, the
        walk over ``sections`` would drift the moment a section was added. The
        *judgement* is not duplicated either: it delegates to
        :meth:`MapFactory.foreign_packings`, the single predicate.

        Every string value in every section is offered, so a section added
        later is covered without editing this. That is safe because the
        predicate only reports paths it can resolve to a declared map type.

        Parameters:
            sidecar: A :meth:`build_scene_sidecar` envelope, or ``None``.
            target: The map type the deliverable wants (``"ORM"`` for glTF).
            workflow: A registry workflow name instead of a map-type *target* --
                the form an exporter's texture-template selection arrives in.
                Takes precedence over *target*; see
                :meth:`MapFactory.foreign_packings` for the two judgements.

        Returns:
            ``{path: map type}``; empty for ``None``, an empty envelope, or a
            source set that is already appropriate.
        """
        from pythontk.core_utils.engines.textures.map_factory import MapFactory

        paths: List[str] = []
        for entry in ((sidecar or {}).get("sections") or {}).values():
            if not isinstance(entry, dict):
                continue
            for spec in entry.values():
                if isinstance(spec, str):
                    paths.append(spec)
                elif isinstance(spec, dict):
                    paths.extend(v for v in spec.values() if isinstance(v, str))
        return MapFactory.foreign_packings(paths, target=target, workflow=workflow)

    @classmethod
    def read_scene_sidecar(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """The scene-sidecar envelope embedded in a GLB, or ``None``.

        The consumer half of :meth:`apply_scene_sidecar` — what a downstream
        tool (or a test) calls to get the scene description back out of a
        deliverable with no side files. Reads only the JSON chunk; the
        geometry is never touched.
        """
        with cls.open_glb(glb) as edit:
            return (edit.gltf.get("extras") or {}).get("scene_sidecar")

    @classmethod
    def verify_glb(cls, glb: GlbTarget) -> Dict[str, Any]:
        """Check a delivered GLB against the envelope it carries.

        The reader a *recipient* runs. Everything the envelope promises is
        checkable from the file alone -- that is what ``textures`` (content
        addresses) and ``validate`` (the counts the envelope claims for itself)
        are for -- but until this existed nothing in the ecosystem read either
        back, so a truncated envelope or a payload swapped after the digests
        were stamped arrived indistinguishable from a good one.

        Read-only and side-file-free by design: a recipient on another machine,
        in another DCC, or holding nothing but the ``.glb`` can run it, and it
        can never be what damages the asset it is inspecting.

        Decodes one channel of each ORM image the envelope did not describe
        (see :meth:`suspect_orm_materials`); everything else reads the JSON
        chunk and hashes payloads. That cost is deliberate here and nowhere
        else -- this is an explicit recipient-side call, not an export step.

        Returns:
            A report dict, always carrying ``ok`` (bool), ``problems`` (the
            findings that made ``ok`` False) and ``notes`` (observations that
            did not), plus what was inspected: ``envelope``
            (present/version/source/asset), ``textures`` (``checked``,
            ``verified``, ``mismatched``, ``unresolved``), ``sections``
            (declared vs applied), ``orm`` (per
            :meth:`suspect_orm_materials`, when anything was found),
            ``lightmap`` and ``generator``. ``problems`` and ``ok`` are kept
            strictly in step: an empty ``problems`` always means ``ok``. A GLB
            with no envelope is reported, not raised -- an asset from another
            producer is a legitimate thing to point this at.
        """
        report: Dict[str, Any] = {"ok": True, "problems": [], "notes": []}

        def fail(message: str) -> None:
            report["ok"] = False
            report["problems"].append(message)

        with cls.open_glb(glb) as edit:
            extras = edit.gltf.get("extras") or {}
            envelope = extras.get("scene_sidecar")
            report["generator"] = (edit.gltf.get("asset") or {}).get("generator")
            report["lightmap"] = bool(extras.get(cls.LIGHTMAP_WEB_KEY))
            if not isinstance(envelope, dict):
                report["envelope"] = None
                fail("no scene-sidecar envelope: nothing here declares itself")
                return report

            version = envelope.get("version")
            report["envelope"] = {
                "version": version,
                "source": envelope.get("source"),
                "asset": envelope.get("asset"),
            }
            # Newer is the case worth saying out loud: this reader would skip
            # whatever the newer schema added and report a clean bill on a file
            # it only partly understood.
            if version != cls.SIDECAR_VERSION:
                fail(
                    f"envelope schema v{version} against this reader's "
                    f"v{cls.SIDECAR_VERSION}"
                )

            declared = envelope.get("sections") or {}
            if not isinstance(declared, dict):
                # Said out loud rather than normalised to {}: quietly treating a
                # block this reader cannot parse as an empty one would report a
                # clean bill on an envelope it never read -- the exact false
                # green this method exists to end.
                fail("envelope 'sections' is not an object")
                declared = {}
            applied = extras.get("scene_sidecar_applied")
            report["sections"] = {
                "declared": {
                    name: len(entries)
                    for name, entries in declared.items()
                    if hasattr(entries, "__len__")
                },
                "applied": applied if isinstance(applied, dict) else {},
            }
            for name, outcome in report["sections"]["applied"].items():
                # The apply pass records its own bad news in the artifact; a
                # verifying reader has to repeat it rather than assume the
                # recipient read the log of a run on someone else's machine.
                # Anchored, not `in`: the summary spells a partial apply "10 of
                # 20", and a bare "0 of" substring test calls that a failure.
                text = str(outcome)
                if text.startswith("failed") or text.startswith("0 of"):
                    fail(f"section {name!r} did not land: {outcome}")

            images = edit.images
            refs = envelope.get("textures") or {}
            if not isinstance(refs, dict):
                fail("envelope 'textures' is not an object")
                refs = {}
            checked = verified = 0
            mismatched: List[str] = []
            unresolved: List[str] = []
            for path, ref in refs.items():
                if not isinstance(ref, dict):
                    continue
                checked += 1
                index, digest = ref.get("image"), ref.get("sha256")
                payload = (
                    edit._image_payload(images[index])
                    if isinstance(index, int) and 0 <= index < len(images)
                    else None
                )
                if not payload or not digest:
                    unresolved.append(path)
                elif hashlib.sha256(payload).hexdigest() != digest:
                    mismatched.append(path)
                else:
                    verified += 1
            report["textures"] = {
                "checked": checked,
                "verified": verified,
                "mismatched": mismatched,
                "unresolved": unresolved,
            }
            if mismatched:
                fail(
                    f"{len(mismatched)} texture reference(s) do not match their "
                    "recorded sha256 -- the payload changed after it was stamped"
                )
            if unresolved:
                fail(
                    f"{len(unresolved)} texture reference(s) resolve to no image "
                    "payload in this file"
                )

            # `validate` is the envelope's claim about ITSELF, so a disagreement
            # means the envelope was truncated or rewritten, not that a texture
            # is missing -- worth separating from the digest failures above.
            claims = envelope.get("validate")
            if isinstance(claims, dict):
                if claims.get("sections") != report["sections"]["declared"]:
                    fail("envelope 'validate' section counts disagree with 'sections'")
                if claims.get("textures") != checked:
                    fail(
                        f"envelope 'validate' claims {claims.get('textures')} texture "
                        f"reference(s), found {checked}"
                    )
            else:
                fail("envelope carries no 'validate' block to check against")

            # `described` is the envelope's own section, so a deliverable that
            # documented every ORM it carries reports nothing here -- and one
            # that carries a binding nothing described says so, which is the
            # only way a mask map packed for another engine (read channel for
            # channel as ORM, and perfectly ordinary-looking data) surfaces at
            # all.
            suspect = cls.suspect_orm_materials(
                edit, described=declared.get("metallic_roughness") or {}
            )
            if suspect:
                report["orm"] = suspect
                by_finding: Dict[str, List[str]] = {}
                for material, found in sorted(suspect.items()):
                    by_finding.setdefault(found["finding"], []).append(material)
                harmful = by_finding.get(cls.ORM_FINDING_METALLIC_FULL)
                unvalidated = by_finding.get(cls.ORM_FINDING_UNVALIDATED)
                if harmful:
                    fail(
                        "metallic=1 everywhere on: "
                        + ", ".join(harmful)
                        + " (no diffuse response; black under a lightmap)"
                    )
                if unvalidated:
                    # A NOTE, not a problem: an ORM the producer never had to
                    # repair is perfectly legitimate, and putting this in
                    # `problems` would make the obvious `if report["problems"]`
                    # read as a defect on a sound deliverable. It is said at all
                    # because nothing here can vouch for its channel layout.
                    report["notes"].append(
                        "ORM binding not described by the envelope on: "
                        + ", ".join(unvalidated)
                        + " -- its channel layout is unverified (a mask map "
                        "packed for another engine reads as ORM here)"
                    )
        return report

    # ------------------------------------------------------------------ #
    # Lightmaps: carry a host DCC's committed bake into the GLB deliverable
    # ------------------------------------------------------------------ #

    #: FBX user-property key the host DCCs publish their lightmap manifest under
    #: (mayatk/blendertk ``LightmapBaker.LIGHTMAP_METADATA``); arrives in the GLB
    #: as node extras via FBX2glTF's ``--user-properties``.
    LIGHTMAP_METADATA_KEY = "lightmap_metadata"
    #: Highest ``lightmap_metadata`` schema this applier knows how to read.
    LIGHTMAP_METADATA_VERSION = 1
    #: Root-extras key the web viewer reads (``preview_viewer.html``).
    LIGHTMAP_WEB_KEY = "lightmap_web"
    #: Per-object bake marker, riding the same FBX user-property channel as the
    #: manifest (mayatk/blendertk ``LightmapBaker.LIGHTMAP_INFO_ATTR``). Carries
    #: its own copy of the locate hint, once per lightmapped object.
    LIGHTMAP_INFO_KEY = "lightmapInfo"
    #: The publisher's absolute authoring directory for the baked maps -- a
    #: BUILD-TIME hint for :meth:`apply_glb_lightmaps` to find the EXRs, with no
    #: reader once they are embedded. Stripped on the way out (see
    #: :meth:`_reconcile_node_markers`); it is machine-specific, so shipping it leaks
    #: the authoring drive layout and tells the recipient nothing they can use.
    LOCATE_HINT_KEY = "dir"

    @classmethod
    def _lightmap_manifest(cls, gltf: dict) -> Optional[Dict[str, Any]]:
        """Parse the ``lightmap_metadata`` manifest out of a parsed glTF, or ``None``.

        Split from :meth:`read_glb_lightmap_manifest` so the applier can read it from
        the session it is ALREADY holding -- a second ``open_glb`` on the path would
        re-read and re-parse the file, which is exactly what :class:`GlbEdit` exists to
        avoid.
        """
        for node in gltf.get("nodes", []) or []:
            extras = node.get("extras") or {}
            # The same TWO on-disk shapes :meth:`_reconcile_node_markers` walks,
            # and for the same reason: FBX2glTF nests user properties under
            # extras.fromFBX.userProperties (the Maya route), a native glTF
            # export writes them as TOP-LEVEL node extras. Probing only the
            # nested shape makes the whole applier a silent no-op on a natively
            # exported deliverable -- and this is public API, pointed at
            # whatever GLB it is handed, so the shape is not knowable from the
            # call site. Nested first: a file carrying both came through the FBX
            # hop, and that copy is the one the converter transcribed.
            for props in (
                (extras.get("fromFBX") or {}).get("userProperties") or {},
                extras,
            ):
                entry = props.get(cls.LIGHTMAP_METADATA_KEY)
                if entry is None:
                    continue
                # FBX2glTF wraps each property as {"type": ..., "value": ...}.
                raw = entry.get("value") if isinstance(entry, dict) else entry
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, ValueError) as error:
                    logger.warning(
                        "Unparsable %s on node %r: %s",
                        cls.LIGHTMAP_METADATA_KEY,
                        node.get("name"),
                        error,
                    )
                    return None
                return data if isinstance(data, dict) else None
        return None

    @classmethod
    def without_locate_hints(cls, data_export: Dict[str, Any]) -> Dict[str, Any]:
        """Copy of a ``data_export`` snapshot with build-time locate hints removed.

        The DCC-side counterpart to :meth:`_reconcile_node_markers`, which does the
        same job on a parsed glTF. Both mayatk and blendertk write an export
        sidecar straight from a ``DataNodes.dump(decode=True)`` snapshot, and
        that snapshot carries :attr:`LOCATE_HINT_KEY` -- an absolute authoring
        directory that resolves nowhere but the machine that baked the maps, and
        so discloses that machine's drive layout to whoever receives the file.

        Lives here rather than in either DCC package because it is the same rule
        as the glTF-side scrub applied to a different container: keeping the key
        name, the channel name and the removal in one place is what stops a
        second hint key from being added to two of the three and missed in the
        third.

        Never mutates: the caller's snapshot is a dump it may still be using,
        and a scrub performed for *serialization* has no business editing it.
        A snapshot that actually carries a hint therefore comes back as a new
        dict; one that carries none is returned as-is, since copying a payload
        with nothing to strip would churn every clean export for no benefit.
        Either way the result is meant to be read, not edited in place.
        """
        section = data_export.get(cls.LIGHTMAP_METADATA_KEY)
        # A snapshot taken WITHOUT decode=True leaves the manifest a JSON string;
        # no production caller does that, and re-serializing one here would
        # silently change the shape the sidecar records, so leave it alone.
        if not isinstance(section, dict) or cls.LOCATE_HINT_KEY not in section:
            # Nothing to strip: hand the snapshot straight back rather than
            # churning a copy of a payload that is already clean (pinned by
            # test_without_locate_hints_passes_clean_snapshots_through). Note
            # this is the one path whose result is the caller's own object --
            # safe because the contract is only that this never MUTATES the
            # snapshot, and treating the return as read-only satisfies both.
            return data_export
        scrubbed = dict(data_export)
        scrubbed[cls.LIGHTMAP_METADATA_KEY] = {
            k: v for k, v in section.items() if k != cls.LOCATE_HINT_KEY
        }
        return scrubbed

    @classmethod
    def _reconcile_node_markers(cls, gltf: dict, final: dict = None) -> int:
        """Correct the per-node lightmap markers and drop their build-time hints.

        Call once the maps are embedded: the hint's only reader is the applier's
        own EXR lookup, so after that it is dead weight that ships an absolute
        authoring path to whoever receives the file. Both carriers are scrubbed --
        the scene-wide manifest and every per-object marker (which holds its own
        copy, so a room with 46 lightmapped objects shipped 46 more).

        Surgical by design: only this one key goes, leaving ``map``/``uv_set``/
        ``intensity``/``scaleOffset`` intact, so a consumer reading the markers
        keeps everything it can actually act on. A basename alone stays resolvable
        against ``search_dirs`` or the GLB's own directory.

        Those KEPT values are also corrected here, from *final* — the map name
        and intensity this run actually committed, keyed by object name. They are
        written by the DCC bake pass BEFORE the web encode exists, so they name an
        ``.exr`` that ships nowhere and an intensity of 1.0 that predates
        normalisation, while ``extras.lightmap_web`` carries the embedded PNG and
        the scalar restoring the bake range. Measured on a client hand-off: 13.65625
        against 1.0, i.e. a consumer trusting a marker rendered the bake ~13.7x too
        dark, and the markers are what a reader finds FIRST (they sit next to the
        mesh). Correcting beats deleting: the applier reads these markers to locate
        the EXR, and the surviving keys are documented as a consumer contract — the
        defect is that they were stale, not that they exist.

        Both jobs ride the ONE walk because both rewrite the same wrapped-JSON
        markers; splitting them would duplicate the unwrap/re-serialize dance that
        every reader of this structure has to get right.

        Args:
            gltf: Parsed glTF, mutated in place.
            final: Optional ``{object_name: {"map": str, "intensity": float}}`` — the
                values this run committed. Omitted, the markers are only stripped.

        Returns:
            int: number of carriers changed (0 when there was nothing to do).
        """
        stripped = 0
        for node in gltf.get("nodes", []) or []:
            extras = node.get("extras") or {}
            # TWO on-disk shapes, both real. FBX2glTF nests user properties under
            # extras.fromFBX.userProperties (the Maya route); blendertk's native
            # glTF export writes them as TOP-LEVEL node extras, verified on a real
            # deliverable. Walking only the first silently skipped every
            # Blender-authored GLB — and this is public API the preview server
            # points at whatever GLB it is given, so the shape is not knowable
            # from the call site.
            for props in (
                (extras.get("fromFBX") or {}).get("userProperties") or {},
                extras,
            ):
                for key in (cls.LIGHTMAP_METADATA_KEY, cls.LIGHTMAP_INFO_KEY):
                    entry = props.get(key)
                    if entry is None:
                        continue
                    wrapped = isinstance(entry, dict) and "value" in entry
                    raw = entry.get("value") if wrapped else entry
                    # Only the JSON-string form can hide a hint; anything else is
                    # either already a dict or not ours to rewrite.
                    try:
                        data = json.loads(raw) if isinstance(raw, str) else raw
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(data, dict):
                        continue
                    # Correct the stale values first, then drop the hint. Keyed by the
                    # NODE name because that is what the records carry; a marker on a
                    # node this run did not bind is left exactly as found rather than
                    # guessed at.
                    corrected = False
                    # Guarded on a truthy name: several nodes can be nameless, and
                    # None == None would then apply one binding's values to every
                    # nameless marker in the file.
                    node_name = node.get("name")
                    committed = (final or {}).get(node_name) if node_name else None
                    if committed:
                        for field in ("map", "intensity"):
                            if (
                                field in committed
                                and data.get(field) != committed[field]
                            ):
                                data[field] = committed[field]
                                corrected = True
                    had_hint = cls.LOCATE_HINT_KEY in data
                    if not had_hint and not corrected:
                        continue
                    data.pop(cls.LOCATE_HINT_KEY, None)
                    # Re-serialize in the shape it arrived in, so a reader that does
                    # not know about this scrub sees no structural change.
                    new = json.dumps(data) if isinstance(raw, str) else data
                    if wrapped:
                        entry["value"] = new
                    else:
                        props[key] = new
                    stripped += 1
        return stripped

    @classmethod
    def read_glb_lightmap_manifest(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """The ``lightmap_metadata`` manifest riding a GLB's node extras, or ``None``.

        The manifest travels **in-band**: the host DCC publishes it as a string user
        property on its ``data_export`` carrier node, Maya's FBX exporter writes it as
        an FBX user property, and FBX2glTF's ``--user-properties`` (always passed by
        :meth:`fbx_to_glb`) transcribes it into that node's glTF extras as
        ``extras.fromFBX.userProperties.<key>`` -- probe-verified against v0.13.1.
        So no consumer has to pass anything; the deliverable feeds its own repair.

        A GLB that never made the FBX hop carries the same key as a TOP-LEVEL node
        extra instead (a native glTF export's custom properties), and both shapes
        are read -- the marker walk knows both, and a probe that knew only one
        would make this whole path a silent no-op on half the deliverables.
        Returns ``None`` for a GLB carrying neither, which is a clean no-op for
        every caller.
        """
        with cls.open_glb(glb) as edit:
            return cls._lightmap_manifest(edit.gltf)

    @classmethod
    def apply_glb_lightmaps(
        cls,
        glb: GlbTarget,
        search_dirs: Sequence[str] = (),
        carrier: str = "occlusion",
        percentile: Optional[float] = None,
        replace_authored: bool = True,
    ) -> List[Dict[str, Any]]:
        """Wire a host DCC's committed lightmaps into a GLB for the web viewer.

        The GLB consumer half of the scene-state contract: the bake tool commits to
        the *scene* (per-shape markers -> the ``lightmap_metadata`` manifest on the
        FBX ``data_export`` carrier), and this reads that manifest back **out of the
        GLB itself** (:meth:`read_glb_lightmap_manifest`), encodes each referenced HDR
        EXR for the web (:meth:`ImgUtils.encode_hdr_for_web`), embeds it, and binds it
        as the material's ``occlusionTexture`` on ``TEXCOORD_1`` with the
        ``lightmap_web`` root-extras manifest the viewer rebinds from. glTF has no
        lightmap slot; the occlusion carrier is the convention the viewer (and
        blendertk's native exporter) already share -- a naive viewer degrades to grey
        AO rather than to nothing.

        A GLB with no manifest is a clean no-op, which is what makes it safe to run
        unconditionally after every conversion. Name matching is exact first, then
        namespace-tolerant: a name that misses is retried with the ``NS:`` prefix
        stripped from both sides and binds only when that leaf is unambiguous among
        the GLB's nodes (manifests and exports can disagree about namespaces
        without either being wrong -- an older publisher stripped them, some
        exporters flatten them). Every remaining miss is loud: an ambiguous leaf,
        a name matching no node at all, a primitive without ``TEXCOORD_1`` (the
        FBX was exported without the second UV set), an EXR that cannot be found,
        and two objects claiming one material with different maps (atlas packing
        prevents this; reaching it means per-object maps on a shared material --
        the symptom is one object wearing another's lighting) are each warned and
        skipped, never guessed at.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.
            search_dirs: Extra directories to resolve the manifest's EXR basenames
                against. Tried after the manifest's own ``dir`` hint, before the
                GLB's directory.
            carrier: ``"occlusion"`` (default) or ``"emissive"`` -- which material
                slot carries the map (mirror of blendertk's ``CARRIERS``).
            percentile: Encode divisor percentile
                (default :attr:`ImgUtils.HDR_WEB_PERCENTILE`).
            replace_authored: Whether a map already sitting in the carrier slot
                gives way to the lightmap. ``True`` (default) displaces it with
                a warning -- a bake IS the deliverable, and its occlusion term
                supersedes a separate AO map. ``False`` keeps the authored map
                and skips the lightmap for that material, warned. Either way, a
                slot holding the material's own ``metallicRoughnessTexture`` is
                NOT authored -- that is the packed-ORM occlusion binding
                (:meth:`set_glb_metallic_roughness`, or FBX2glTF's converted
                packing), whose R channel the bake already contains -- so it is
                displaced silently regardless.

        **Per-instance atlas rects travel as glTF-standard ``KHR_texture_transform``.**
        A manifest record with a non-identity ``scaleOffset`` (an instance's patch of a
        shared atlas over the mesh's shared [0,1] unwrap) cannot bind on the shared
        material -- every sibling would sample the same patch -- so its node gets its
        own MATERIAL clone carrying the rect as a texture transform, and, when the node
        shares its glTF mesh with siblings (FBX2glTF preserves instancing as one mesh
        referenced by many nodes -- probe-measured), its own MESH entry too. Both
        clones are pure JSON referencing the same accessors/bufferViews: zero geometry
        or texture duplication, and any compliant viewer (three.js, model-viewer,
        Babylon, the production WebXR app) renders the rect with no custom code.

        Returns:
            One record per (object, material) binding: ``{"material", "object",
            "map", "intensity", "scaleOffset"}`` -- several objects sharing one atlas
            material each get a record. Empty when there was no manifest or nothing
            matched.
        """
        import copy

        from pythontk.img_utils._img_utils import ImgUtils

        slot = "occlusionTexture" if carrier != "emissive" else "emissiveTexture"
        identity = [1.0, 1.0, 0.0, 0.0]
        records: List[Dict[str, Any]] = []

        def _authored_carrier(material: dict) -> Optional[dict]:
            """The AUTHORED map in the carrier slot, or ``None``.

            A slot reference sharing its texture index with the material's own
            ``metallicRoughnessTexture`` is the packed-ORM occlusion binding,
            not an authored map -- its R channel is AO the bake already
            contains (with real bounce), so displacing it is never a loss and
            never worth a warning.
            """
            existing = material.get(slot)
            if not existing:
                return None
            if slot == "occlusionTexture":
                mr = (material.get("pbrMetallicRoughness") or {}).get(
                    "metallicRoughnessTexture"
                )
                if mr and existing.get("index") == mr.get("index"):
                    return None
            return existing

        # ONE session for the read and the write: the manifest is read from the very
        # GLB being edited, so re-opening the path to find it would re-read and
        # re-parse the file that GlbEdit exists to read exactly once.
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            manifest = cls._lightmap_manifest(gltf)
            if not manifest:
                return []
            try:
                version = int(manifest.get("version", -1))
            except (TypeError, ValueError):
                version = -1
            if not 0 < version <= cls.LIGHTMAP_METADATA_VERSION:
                # Refuse rather than guess: a newer schema is free to change what the
                # fields MEAN, and binding a map through a misread manifest looks like
                # a bad bake rather than a version problem.
                logger.warning(
                    "lightmap_metadata v%r is newer than this reader (v%s); "
                    "lightmaps not wired.",
                    manifest.get("version"),
                    cls.LIGHTMAP_METADATA_VERSION,
                )
                return []
            entries = manifest.get("objects") or []
            if not entries:
                return []

            dirs = [d for d in [manifest.get("dir"), *search_dirs] if d]
            dirs.append(os.path.dirname(os.path.abspath(edit.path)))

            nodes_by_name: Dict[str, List[dict]] = {}
            mesh_users: Dict[int, int] = {}  # mesh index -> node reference count
            for node in gltf.get("nodes", []) or []:
                if "mesh" in node:
                    nodes_by_name.setdefault(node.get("name", ""), []).append(node)
                    mesh_users[node["mesh"]] = mesh_users.get(node["mesh"], 0) + 1
            # Namespace-tolerant fallback index (Maya ``NS:leaf``). The manifest
            # and the export can disagree about namespaces without either being
            # wrong -- an older publisher stripped them, some exporters flatten
            # them -- and an exact-only match then silently unbinds every
            # namespaced object (a delivered room whose referenced racks all
            # rendered black). Match exact first; fall back to comparing with
            # the namespace stripped from BOTH sides, but only when that leaf
            # is unambiguous among the GLB's nodes -- a guessed bind on a
            # duplicate leaf would put one object's lighting on another.
            leaf_index: Dict[str, List[str]] = {}
            for full in nodes_by_name:
                leaf_index.setdefault(full.rsplit(":", 1)[-1], []).append(full)

            # exr abspath -> encode scalar; ``None`` records a failed encode so a map
            # shared by several objects is not retried (and re-logged) per object.
            scalars: Dict[str, Optional[float]] = {}
            claimed: Dict[int, str] = {}  # material index -> exr abspath
            # Source material indices whose authored carrier map was already
            # reported (displaced, or kept under replace_authored=False) -- so the
            # warning fires once however many instances or primitives share it.
            dropped_authored: Set[int] = set()
            web_materials: Dict[str, Dict[str, Any]] = {}
            # Keyed by the RESOLVED node rather than by the manifest entry:
            # node lookup is namespace-tolerant (a manifest "room" can bind a
            # GLB node "NS:room"), so keying the marker corrections off the
            # manifest name would miss on exactly the scenes that tolerance
            # exists for -- silently, since the markers would simply keep their
            # stale values.
            marker_updates: Dict[str, Dict[str, Any]] = {}
            used_transform = False
            for entry in entries:
                name, basename = entry.get("name"), entry.get("map")
                rect = [float(v) for v in (entry.get("scaleOffset") or identity)]
                has_rect = rect != identity
                nodes = nodes_by_name.get(name or "")
                if not nodes and name:
                    leaves = leaf_index.get(name.rsplit(":", 1)[-1]) or []
                    if len(leaves) == 1:
                        nodes = nodes_by_name[leaves[0]]
                    elif len(leaves) > 1:
                        logger.warning(
                            "Lightmap for %r: leaf name matches several GLB "
                            "nodes (%s) -- ambiguous, not bound.",
                            name,
                            ", ".join(sorted(leaves)),
                        )
                        continue
                if not nodes:
                    logger.warning(
                        "Lightmap for %r: no mesh node by that name in the GLB "
                        "(nodes: %s).",
                        name,
                        ", ".join(sorted(nodes_by_name)) or "<none>",
                    )
                    continue
                src = next(
                    (
                        p
                        for d in dirs
                        for p in [os.path.join(d, basename or "")]
                        if basename and os.path.isfile(p)
                    ),
                    None,
                )
                if src is None:
                    logger.warning(
                        "Lightmap for %r: %r not found in %s.", name, basename, dirs
                    )
                    continue
                src = os.path.abspath(src)

                png_name = os.path.splitext(basename)[0] + ".png"

                def _scalar(src=src, basename=basename, png_name=png_name):
                    """Encode + embed *src* on FIRST use (never for an entry whose
                    primitives all fail the guards -- an orphan texture otherwise)."""
                    if src not in scalars:
                        try:
                            png, encoded = ImgUtils.encode_hdr_for_web(src, percentile)
                        except (ImportError, ValueError) as error:
                            logger.warning(
                                "Lightmap %r not encoded: %s", basename, error
                            )
                            scalars[src] = None
                        else:
                            scalars[src] = encoded
                            cls._embed_image_bytes(
                                edit, src, png, name=png_name, clamp=True
                            )
                    return scalars[src]

                for node in nodes:
                    if has_rect and mesh_users.get(node["mesh"], 0) > 1:
                        # This node shares its mesh with siblings but needs its own
                        # material binding: give it its own mesh ENTRY. Pure JSON --
                        # the clone references the same accessors/bufferViews, so no
                        # geometry is duplicated. The LAST remaining user keeps the
                        # original entry (no clone needed once it is sole owner).
                        mesh_clone = copy.deepcopy(gltf["meshes"][node["mesh"]])
                        mesh_clone["name"] = (
                            f"{mesh_clone.get('name') or 'mesh'}~{name}"
                        )
                        mesh_users[node["mesh"]] -= 1
                        gltf["meshes"].append(mesh_clone)
                        node["mesh"] = len(gltf["meshes"]) - 1
                        mesh_users[node["mesh"]] = 1
                    for prim in gltf["meshes"][node["mesh"]].get("primitives", []):
                        if "TEXCOORD_1" not in (prim.get("attributes") or {}):
                            logger.warning(
                                "Lightmap for %r: primitive has no TEXCOORD_1 -- "
                                "the FBX was exported without the lightmap UV set; "
                                "skipped.",
                                name,
                            )
                            continue
                        mi = prim.get("material")
                        if mi is None:
                            continue

                        if has_rect:
                            # The rect is per-INSTANCE and the material is shared by
                            # every instance -- so the rect rides a material CLONE as
                            # a glTF-standard KHR_texture_transform, and any
                            # compliant viewer applies it with no custom code. The
                            # clone is JSON only (same shader inputs, same embedded
                            # texture index).
                            base = gltf["materials"][mi]
                            base_name = base.get("name") or f"mat{mi}"
                            # The authored gate runs BEFORE the encode: _scalar()
                            # embeds on first use, so skipping after it would leave
                            # an orphan texture in the file when every instance of
                            # the material is kept authored. Warn ONCE per source
                            # material, not once per clone: a room whose 46 pieces
                            # share one material would otherwise emit 46 identical
                            # lines and bury every other warning in the log. The
                            # instance names are the noise here -- the material and
                            # the count are the finding.
                            if _authored_carrier(base) is not None:
                                if not replace_authored:
                                    if mi not in dropped_authored:
                                        dropped_authored.add(mi)
                                        logger.warning(
                                            "Material %r keeps its authored %s; its "
                                            "instance lightmaps are not bound "
                                            "(replace_authored=False).",
                                            base_name,
                                            slot,
                                        )
                                    continue
                                if mi not in dropped_authored:
                                    dropped_authored.add(mi)
                                    logger.warning(
                                        "Material %r: its authored %s is dropped on "
                                        "every lightmap clone made from it (the viewer "
                                        "rebinds the slot to lightMap).",
                                        base_name,
                                        slot,
                                    )
                            scalar = _scalar()
                            if scalar is None:  # encode failed, already logged
                                continue
                            clone = copy.deepcopy(base)
                            clone["name"] = f"{base_name}~lm{len(gltf['materials'])}"
                            g_rect = ImgUtils.flip_rect_v(rect)
                            clone[slot] = {
                                "index": edit.embedded[src],
                                "texCoord": 1,
                                "extensions": {
                                    "KHR_texture_transform": {
                                        "offset": [g_rect[2], g_rect[3]],
                                        "scale": [g_rect[0], g_rect[1]],
                                    }
                                },
                            }
                            gltf["materials"].append(clone)
                            prim["material"] = len(gltf["materials"]) - 1
                            used_transform = True
                            web_materials[clone["name"]] = {
                                "map": png_name,
                                "intensity": round(scalar, 6),
                            }
                            if node.get("name"):
                                marker_updates[node["name"]] = web_materials[
                                    clone["name"]
                                ]
                            records.append(
                                {
                                    "material": clone["name"],
                                    "object": name,
                                    "map": basename,
                                    "intensity": scalar,
                                    "scaleOffset": rect,
                                }
                            )
                            continue

                        if claimed.get(mi, src) != src:
                            logger.warning(
                                "Material %r already carries %r; %r's map %r has "
                                "nowhere to go (per-object maps on a shared "
                                "material -- atlas packing prevents this).",
                                (gltf["materials"][mi].get("name") or mi),
                                os.path.basename(claimed[mi]),
                                name,
                                basename,
                            )
                            continue
                        material = gltf["materials"][mi]
                        if mi not in claimed and _authored_carrier(material):
                            # An AUTHORED map sits on the carrier slot (a real AO
                            # map, say -- the packed-ORM binding is already ruled
                            # out). The default displaces it, loudly: the bake IS
                            # the deliverable, but silently discarding authored
                            # data is not this function's call to make quietly.
                            # replace_authored=False keeps it instead, and the
                            # lightmap for this material is not bound. The skip
                            # never sets ``claimed``, so it once-guards through
                            # ``dropped_authored`` -- per material, or a shared
                            # material would repeat the line for every primitive
                            # of every object wearing it. (The displace branch
                            # once-guards naturally: binding sets ``claimed``.)
                            if not replace_authored:
                                if mi not in dropped_authored:
                                    dropped_authored.add(mi)
                                    logger.warning(
                                        "Material %r keeps its authored %s; its "
                                        "lightmaps are not bound "
                                        "(replace_authored=False).",
                                        material.get("name") or mi,
                                        slot,
                                    )
                                continue
                            logger.warning(
                                "Material %r: replacing its authored %s with the "
                                "lightmap (the viewer rebinds it to the lightMap "
                                "slot).",
                                material.get("name") or mi,
                                slot,
                            )
                        scalar = _scalar()
                        if scalar is None:  # encode failed, already logged
                            continue
                        material[slot] = {
                            "index": edit.embedded[src],
                            "texCoord": 1,
                        }
                        claimed[mi] = src
                        mat_name = material.get("name")
                        if not mat_name:
                            logger.warning(
                                "Material %s is anonymous; it carries the lightmap "
                                "but cannot be keyed in the viewer manifest.",
                                mi,
                            )
                            continue
                        web_materials[mat_name] = {
                            "map": png_name,
                            "intensity": round(scalar, 6),
                        }
                        if node.get("name"):
                            marker_updates[node["name"]] = web_materials[mat_name]
                        records.append(
                            {
                                "material": mat_name,
                                "object": name,
                                "map": basename,
                                "intensity": scalar,
                                "scaleOffset": list(identity),
                            }
                        )

            if used_transform:
                ext_used = edit.gltf.setdefault("extensionsUsed", [])
                if "KHR_texture_transform" not in ext_used:
                    ext_used.append("KHR_texture_transform")
            if web_materials:
                # The exact shape the viewer parses (root extras is its 2nd probe).
                edit.gltf.setdefault("extras", {})[cls.LIGHTMAP_WEB_KEY] = {
                    "version": 1,
                    "carrier": carrier,
                    "uv": 1,
                    "encoding": "srgb",
                    "materials": web_materials,
                }
                # The maps are in the file now, so the authoring-path hints that
                # found them have no reader left -- drop them here, where that is
                # provably true, rather than at export (where the applier still
                # needs them) or never (where they ship). Deliberately gated on a
                # successful embed: a run that bound nothing leaves them intact so
                # a retry -- after fixing a name mismatch, say -- can still locate
                # the EXRs.
                # Same walk corrects the superseded copies: every marker this run
                # bound still names the .exr at the pre-normalisation intensity.
                # Fed the PUBLISHED values (the same dicts that went into
                # lightmap_web), so the copies come out identical rather than
                # merely close: a record's "map" is the SOURCE .exr basename —
                # which is what a caller wants to know — and its "intensity" is
                # unrounded where the published one is round(., 6).
                cls._reconcile_node_markers(edit.gltf, marker_updates)
                edit.dirty = True
        return records

    # ------------------------------------------------------------------ #
    # Post-conversion material sanity check
    # ------------------------------------------------------------------ #

    @classmethod
    def check_glb_materials(cls, glb: GlbTarget) -> List[Dict[str, str]]:
        """Inspect a GLB for materials flagged transparent that should be opaque.

        Catches the Maya/Stingray/OpenPBR/Standard-Surface failure mode where
        a color texture happens to carry an alpha channel (often PNG palette
        transparency) without any actual transparency intent. Maya's FBX
        exporter writes a TransparencyFactor; FBX2glTF then sets
        ``alphaMode: BLEND`` and the renderer disables depth-write —
        producing the "inverted face" / wrong-render-order artifact.

        A material is flagged when its ``alphaMode`` is BLEND or MASK *and*
        its base-color texture's alpha channel is uniformly 255. Genuine
        transparency (varying alpha) is not reported.

        Parameters:
            glb: Path to a binary glTF (.glb) file, or an open
                :class:`GlbEdit` session to inspect. Read-only either way.

        Returns:
            List of findings. Each finding is a dict with keys:
                material   — material name (or '<material[i]>')
                alpha_mode — "BLEND" or "MASK"
                image      — image name / uri / fallback id
                reason     — short human-readable explanation
        """
        try:
            # Presence check only — the session does the decoding. Imported as
            # `from PIL import Image` rather than `import PIL` because the dead
            # original PIL also ships the package name and not the module.
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "check_glb_materials requires Pillow (PIL). Install it with "
                "`pip install pillow`."
            ) from exc

        # Reason text per alpha mode — BLEND and MASK fail in different ways.
        REASONS = {
            "BLEND": (
                "alphaMode=BLEND but base-color alpha is uniformly opaque (255). "
                "Renderers disable depth-write for BLEND, causing render-order "
                "artifacts (faces drawing in the wrong order)."
            ),
            "MASK": (
                "alphaMode=MASK but base-color alpha is uniformly opaque (255). "
                "Every fragment passes the cutoff so alpha-testing is a no-op; "
                "the material should be OPAQUE."
            ),
        }

        findings: List[Dict[str, str]] = []
        with cls.open_glb(glb) as edit:
            for mi, mat in enumerate(edit.materials):
                alpha_mode = mat.get("alphaMode", "OPAQUE")
                if alpha_mode not in REASONS:  # OPAQUE or unknown — skip
                    continue

                # Real transparency can come from the scalar baseColorFactor[3];
                # don't flag those as "accidentally transparent".
                pbr = mat.get("pbrMetallicRoughness") or {}
                bc_factor = pbr.get("baseColorFactor")
                if bc_factor and len(bc_factor) >= 4 and bc_factor[3] < 1.0:
                    continue

                img_idx = edit.base_color_image(mat)
                if img_idx is None:
                    continue

                # Decoded once per source image even if many materials share
                # it, and once across every repair sharing this session.
                if edit.alpha_extrema(img_idx) != (255, 255):
                    continue

                findings.append(
                    {
                        "material": mat.get("name") or f"<material[{mi}]>",
                        "alpha_mode": alpha_mode,
                        "image": edit.image_label(img_idx),
                        "reason": REASONS[alpha_mode],
                    }
                )

        return findings

    @classmethod
    def fix_glb_phantom_opaque_alpha(cls, glb: GlbTarget) -> List[Dict]:
        """Repair the Maya phong → FBX → FBX2glTF transparency translation bug.

        When a Maya phong/lambert/blinn shader has its ``.transparency`` fed
        by a file node's ``.outTransparency``, Maya's FBX exporter writes
        ``TransparencyFactor=1.0`` (the texture is meant to modulate
        per-pixel). FBX2glTF then computes
        ``baseColorFactor[3] = 1 - 1 = 0`` — multiplying every fragment's
        alpha by zero and rendering the mesh fully invisible regardless of
        texture content.

        A material is fixed when ALL of:
            - ``alphaMode`` is BLEND or MASK
            - ``baseColorFactor[3]`` is ~0
            - ``baseColorTexture`` exists and references an image with
              *varying* alpha (a real cutout mask, not uniformly 0 or 255)

        On match, ``baseColorFactor[3]`` is reset to 1.0 so per-pixel alpha
        from the texture controls visibility as intended.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.

        Returns:
            List of fix records. Empty when nothing matched. Each record:
                material   — material name
                old_alpha  — original baseColorFactor[3]
                new_alpha  — 1.0
                image      — the baseColorTexture image identifier
        """
        try:
            # Presence check only — the session does the decoding. Imported as
            # `from PIL import Image` rather than `import PIL` because the dead
            # original PIL also ships the package name and not the module.
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "fix_glb_phantom_opaque_alpha requires Pillow (PIL). "
                "Install it with `pip install pillow`."
            ) from exc

        EPSILON = 1e-4
        fixes: List[Dict] = []
        with cls.open_glb(glb) as edit:
            for mi, mat in enumerate(edit.materials):
                if mat.get("alphaMode") not in ("BLEND", "MASK"):
                    continue
                pbr = mat.get("pbrMetallicRoughness") or {}
                bcf = pbr.get("baseColorFactor")
                if not bcf or len(bcf) < 4 or bcf[3] > EPSILON:
                    continue
                img_idx = edit.base_color_image(mat)
                if img_idx is None:
                    continue
                # Every check above reads the JSON alone; this is the first
                # line that can pull the BIN chunk in, and it is reached only
                # by a material that already looks wrong. A GLB with nothing to
                # fix is therefore never read past its JSON chunk.
                extrema = edit.alpha_extrema(img_idx)
                # Skip uniform alpha (genuinely-transparent or genuinely-opaque
                # textures) — only varying alpha indicates a real cutout mask
                # whose per-pixel control was cancelled by baseColorFactor[3]=0.
                if extrema is None or extrema[0] == extrema[1]:
                    continue

                old_alpha = bcf[3]
                bcf[3] = 1.0
                pbr["baseColorFactor"] = bcf
                mat["pbrMetallicRoughness"] = pbr

                fixes.append(
                    {
                        "material": mat.get("name") or f"<material[{mi}]>",
                        "old_alpha": old_alpha,
                        "new_alpha": 1.0,
                        "image": edit.image_label(img_idx),
                    }
                )

            if fixes:
                edit.dirty = True

        return fixes

    @classmethod
    @contextmanager
    def open_glb(cls, glb: GlbTarget):
        """Yield an open :class:`GlbEdit` for *glb*, writing once on close.

        *glb* is a path, **or an already-open session** -- in which case it is
        yielded as-is and its owner keeps responsibility for the write. That
        second form is what lets these repairs compose: each one takes either,
        so running three against a path costs three read/write cycles, while
        wrapping the same three in one ``open_glb`` costs one::

            with MeshConvert.open_glb(path) as glb:
                MeshConvert.set_glb_base_color(glb, base_color)
                MeshConvert.set_glb_emissive(glb, emissive)

        Nothing is written when the body raises -- a half-applied edit must not
        reach disk -- nor when no editor set :attr:`GlbEdit.dirty`.
        """
        if isinstance(glb, cls.GlbEdit):
            yield glb
            return
        edit = cls._read_glb(os.fspath(glb))
        yield edit
        if edit.dirty:
            # Once, here, rather than per writer: composed repairs each embed
            # into the JSON chunk and the BIN is rebuilt a single time for all
            # of them. Runs on the OWNER's close only -- a session handed in by
            # a caller is relocated when that caller closes it, after its own
            # writers have finished embedding.
            cls._relocate_embedded_images(edit)
            cls._write_glb(edit)

    @classmethod
    def _read_glb(cls, glb_path: str) -> "MeshConvert.GlbEdit":
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

        return cls.GlbEdit(glb_path, version_bytes, gltf, chunk0_len)

    @staticmethod
    def _write_glb(edit: "MeshConvert.GlbEdit") -> None:
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
    def _relocate_embedded_images(edit: "MeshConvert.GlbEdit") -> int:
        """Move this session's embedded images from the JSON chunk into the BIN.

        Every channel writer embeds as a ``data:`` URI, which keeps its edit
        inside the JSON chunk -- no buffer offsets to recompute, which is the
        part of GLB surgery that silently corrupts a file. That was priced for
        a local preview, but the same writers build deliverables: measured on
        TURRETS_WIRES.glb, the packed ORM's base64 put the JSON chunk at 45% of
        an 8.9 MB file, 1.0 MB of it pure base64 premium, all of it parsed
        before a loader can draw. This pays the JSON back down once, on close,
        after every writer has had its say.

        Safe because it only ever **appends**: the existing BIN is copied
        verbatim and the new payloads land past its end, so every prior
        bufferView keeps its index, its ``byteOffset`` and its bytes, and no
        accessor is touched. That is the whole difference from
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
        # Buffer 0 is the BIN chunk only when it declares no ``uri``. A GLB
        # whose first buffer is EXTERNAL has no BIN to append to, and writing
        # one would strand the appended views on bytes the file does not carry
        # while overwriting that buffer's byteLength. Leave the payloads in the
        # JSON: base64 is a size cost, corrupting the buffer table is not.
        buffers = gltf.setdefault("buffers", [])
        if buffers and buffers[0].get("uri"):
            return 0

        # The existing BIN joins in as a memoryview rather than a `bytes` copy:
        # on a production GLB that copy is the entire geometry, and `join`
        # reads the view directly, so peak memory is one BIN, not two.
        blob = edit.bin_data
        chunks = [] if blob is None else [blob]
        offset = 0 if blob is None else len(blob)
        pad = (4 - (offset % 4)) % 4
        if pad:  # the appended views must start 4-byte aligned
            chunks.append(b"\x00" * pad)
            offset += pad

        views = gltf.setdefault("bufferViews", [])
        for image in live:
            raw = base64.b64decode(image["uri"].split(",", 1)[1])
            views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw)})
            image["bufferView"] = len(views) - 1
            image.pop("uri", None)
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
        return len(live)

    @classmethod
    def _image_semantics(cls, edit) -> Dict[int, str]:
        """Map image index -> the strongest slot semantic sampling it.

        The structural classification KTX2 mode encodes by: "color" (base
        color / emissive), "data" (metallic-roughness / occlusion), "normal".
        An image referenced from several slots takes the highest
        :attr:`_SEMANTIC_RANK`; one referenced by nothing is absent (the
        caller's lookup default handles it). Slot walks go through
        ``image_for_texture`` so prior WebP/KTX2 bindings resolve to the same
        image a loader would sample.
        """
        semantics: Dict[int, str] = {}
        rank = cls._SEMANTIC_RANK

        def note(ref: Optional[dict], semantic: str) -> None:
            if not ref:
                return
            src = edit.image_for_texture(ref.get("index"))
            if src is None:
                return
            current = semantics.get(src)
            if current is None or rank[semantic] > rank[current]:
                semantics[src] = semantic

        for mat in edit.gltf.get("materials") or []:
            pbr = mat.get("pbrMetallicRoughness") or {}
            note(pbr.get("baseColorTexture"), "color")
            note(mat.get("emissiveTexture"), "color")
            note(pbr.get("metallicRoughnessTexture"), "data")
            note(mat.get("occlusionTexture"), "data")
            note(mat.get("normalTexture"), "normal")
        return semantics

    @classmethod
    def optimize_glb_textures(
        cls,
        glb: GlbTarget,
        max_size: int = 2048,
        image_format: str = "WEBP",
        quality: int = 85,
        workers: Optional[int] = None,
        ktx2_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Downsize and re-encode a GLB's embedded images for web delivery.

        The texture budget IS the file: measured on a production room, the GLB
        was 94.7 MB of which 87.8 MB (93%) was uncompressed source PNG -- a
        24 MB normal map, a 20 MB character texture -- against 2.6 MB of
        geometry. A headset streams that, then holds it decoded in GPU memory.
        blendertk's native web export learned this first
        (``LightmapWebExport``: *"the texture budget is the whole file
        size"*); this is the same policy for the FBX->GLB path the WebXR
        preview and the exporters use, as a separate opt-in pass so plain
        conversions stay byte-stable.

        Every embedded image (bufferView-backed or data URI) is decoded,
        resized so its longest edge is *max_size*, and re-encoded -- WebP by
        default: alpha-capable, universally decoded by WebXR-class browsers,
        and roughly an order of magnitude smaller than PNG at visually equal
        quality. Lightmaps are exempt from the resize (the bake sized them
        deliberately) and re-encode LOSSLESS (lossy WebP's 4:2:0 chroma
        blotches magenta/green on near-black texels); exemption is both by
        the names the ``lightmap_web`` manifest lists and structurally, by
        texCoord-1 occlusion/emissive binding, so a digest-deduped image
        whose name lies is still protected. A re-encode that comes out
        larger keeps the original bytes.

        The BIN chunk is repacked -- image payloads replaced, former data-URI
        images relocated into it (dropping base64's 33%), every other
        bufferView's bytes copied verbatim with offsets recomputed. Textures
        sampling a converted image gain the standard ``EXT_texture_webp``
        binding while keeping their plain ``source`` as the fallback the
        extension spec describes.

        ``image_format="KTX2"`` is the GPU-memory half: images are encoded to
        KTX2/Basis Universal (requires the ``toktx`` encoder -- see
        ``pythontk.Ktx2Encoder``; the pass raises upfront when it is missing
        rather than silently shipping WebP) and textures rebind through
        ``KHR_texture_basisu``. Where WebP only shrinks the wire, Basis stays
        block-compressed in GPU memory -- measured on a delivered preview GLB,
        6 MB of WebP decoded to ~740 MB of RGBA+mips, which is the actual
        headset ceiling. Specifics of this mode, chosen deliberately:

        * **Per-slot codecs** -- images sampled as normal / metallic-roughness /
          occlusion encode UASTC + linear; base color / emissive encode ETC1S +
          sRGB; an image shared across slots takes the stricter treatment, and
          one bound to a texture but to no material slot gets UASTC + sRGB. An
          image *no texture samples* is left as found -- nothing could rebind
          it, and a non-core ``image/ktx2`` that no declaration enables is
          invalid glTF. Same rule as ``MapOptimizer.resolve_compression``,
          keyed structurally (by glTF slot) instead of by filename.
        * **Lightmaps stay on the lossless-WebP path** even in KTX2 mode:
          ETC1S would blotch them for the same reason lossy WebP does, UASTC
          would re-quantise a deliberately-authored bake, and their carrier
          slot's colorspace handling is viewer-rebound -- fidelity wins over
          GPU residency for exactly these images.
        * **A core-readable fallback rides along by default**
          (*ktx2_fallback*): each converted image also embeds a resized
          PNG/JPEG twin bound as the texture's plain ``source`` -- the
          escape hatch the ``KHR_texture_basisu`` spec defines -- so the
          extension stays in ``extensionsUsed`` and the GLB still opens in
          any stock glTF importer (Blender, Unreal, Unity) instead of being
          a terminal delivery artifact only a basisu viewer can read.
          UASTC-class images (normals, metallic-roughness/occlusion) fall
          back to PNG, ETC1S color to JPEG at *quality* (PNG when it
          carries alpha), so the premium over the KTX2 payload stays modest.
          ``ktx2_fallback=False`` is the pure-delivery mode: no fallback,
          the extension lands in ``extensionsRequired``, and the
          deliverable needs a ``KHR_texture_basisu``-capable viewer
          (three.js ``KTX2Loader``; the bundled preview page wires it) --
          the right trade when every byte is budget, as the WebXR preview
          push chooses. A fallback whose own encode fails is logged and
          dropped, and that image's binding re-tips the extension into
          ``extensionsRequired`` -- never an unreadable texture with a
          declaration claiming otherwise.
        * **Dimensions snap down to power-of-two** -- ``KHR_texture_basisu``
          requires multiple-of-4 dimensions and full mip pyramids (generated at
          encode time; a GPU-compressed texture cannot mip itself), and POT is
          what the WebGL/WebGPU backends want for that anyway. No-op for the
          usual POT sources.
        * **A larger encode is kept** (unlike the transport formats): UASTC can
          exceed a source PNG on the wire and still be the right answer,
          because the win being bought is GPU-resident format, not bytes.

        Parameters:
            glb: Path to a .glb, modified in place, or an open session.
            max_size: Longest edge kept after resize. 0/None skips resizing.
            image_format: ``"WEBP"`` (default), ``"KTX2"``, or any PIL-writable
                format.
            quality: Lossy quality for WEBP/JPEG, and the ETC1S quality dial in
                KTX2 mode (UASTC's tier is fixed by the encoder). Also the
                JPEG quality of KTX2-mode fallback images.
            workers: Concurrent encode threads. Defaults to
                :attr:`OPTIMIZE_WORKERS` capped by the core count; 1 forces the
                serial path.
            ktx2_fallback: KTX2 mode only. ``True`` (default) embeds a
                core-readable PNG/JPEG twin per converted image and binds it
                as the texture's plain ``source``, keeping the GLB importable
                everywhere (extension in ``extensionsUsed``). ``False`` ships
                KTX2 alone and hard-requires a basisu-capable viewer
                (``extensionsRequired``).

        Returns:
            Summary dict: ``images`` (converted count), ``bytes_before`` /
            ``bytes_after`` (image payload totals). Empty when Pillow is
            unavailable or there is nothing to do.
        """
        try:
            from PIL import Image
        except ImportError:
            logger.warning("optimize_glb_textures: Pillow unavailable; skipped.")
            return {}

        image_format = image_format.upper()
        is_ktx2 = image_format == "KTX2"
        mime = f"image/{image_format.lower()}"
        encoder = None
        if is_ktx2:
            # Resolve before any work, and never fall back silently: a caller
            # who asked for GPU-resident compression and silently got WebP
            # would ship a deliverable that *looks* optimized while missing
            # the entire point of the request.
            from pythontk.img_utils._img_utils import ImgUtils

            encoder = ImgUtils.resolve_ktx2_encoder(required=True)
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            images = gltf.get("images") or []
            if not images:
                return {}

            # The bake sized the lightmaps deliberately; never resize them.
            exempt: Set[str] = set()
            raw_manifest = (gltf.get("extras") or {}).get("lightmap_web")
            try:
                manifest = (
                    json.loads(raw_manifest)
                    if isinstance(raw_manifest, str)
                    else (raw_manifest or {})
                )
                for entry in (manifest.get("materials") or {}).values():
                    if entry.get("map"):
                        exempt.add(os.path.basename(entry["map"]))
            except (ValueError, AttributeError):
                pass
            # Structural exemption alongside the name set: an image bound as a
            # texCoord-1 occlusion/emissive map IS a lightmap however it is
            # named -- the digest dedupe can hand a lightmap payload the name
            # of whichever image embedded those bytes first, and a name-only
            # check then resizes/lossy-encodes it anyway.
            exempt_indices: Set[int] = set()
            for mat in gltf.get("materials") or []:
                for slot in ("occlusionTexture", "emissiveTexture"):
                    ref = mat.get(slot) or {}
                    if ref.get("texCoord") != 1:
                        continue
                    src = edit.image_for_texture(ref.get("index"))
                    if src is not None:
                        exempt_indices.add(src)

            # KTX2 mode encodes per SLOT semantic (codec + colorspace), so the
            # classification has to happen while the glTF structure is in hand.
            semantic_by_image = cls._image_semantics(edit) if is_ktx2 else {}
            # The images some texture actually samples -- resolved through the
            # shadow-aware walk, so a re-run over an already-optimized GLB sees
            # the EFFECTIVE binding rather than a stale plain ``source``. In
            # KTX2 mode this gates the encode itself (below).
            sampled: Set[Optional[int]] = (
                {
                    edit.image_for_texture(t_index)
                    for t_index in range(len(gltf.get("textures") or []))
                }
                if is_ktx2
                else set()
            )

            # Pass 1, phase A (serial, cheap): classify every image and collect
            # ONE job per distinct (payload, exemption, semantic) triple. The
            # exemption is part of the key because the same bytes named both as
            # a source texture and as a lightmap must not share the resized
            # encoding; the semantic for the same reason -- bytes sampled as a
            # normal map in one material and as base color in another need a
            # UASTC and an ETC1S encode respectively. (Outside KTX2 mode the
            # semantic is a constant None and the key degenerates to the pair.)
            before = after = 0
            jobs: Dict[Tuple[str, bool, Optional[str]], bytes] = {}
            labels: Dict[Tuple[str, bool, Optional[str]], Union[str, int]] = {}
            key_by_index: Dict[int, Tuple[str, bool, Optional[str]]] = {}
            for index, image in enumerate(images):
                payload = edit._image_payload(image)
                if not payload:
                    continue
                before += len(payload)
                is_exempt = (
                    index in exempt_indices or (image.get("name") or "") in exempt
                )
                if is_ktx2 and not is_exempt and index not in sampled:
                    # No texture samples this image, so nothing can rebind it
                    # through KHR_texture_basisu -- and that declaration is
                    # gated on an actual rebind *deliberately*: it lands in
                    # extensionsREQUIRED, which would hard-require a
                    # basisu-capable viewer for a binding no texture has.
                    # Encoding anyway left the other half of that pair
                    # ungated: the mime rewrite below is driven by
                    # ``replacements``, so a GLB whose textures resolve to
                    # none of them shipped ``image/ktx2`` with no extension
                    # enabling it -- and glTF 2.0 core permits image/jpeg and
                    # image/png only. Keeping the bytes as found is valid
                    # either way, and an image no texture reads is dead
                    # payload whichever format it is in. (Exempt lightmaps
                    # take the WebP path, whose declaration is unconditional,
                    # so they are safe unsampled.)
                    after += len(payload)
                    continue
                semantic = semantic_by_image.get(index) if is_ktx2 else None
                key = (hashlib.sha256(payload).hexdigest(), is_exempt, semantic)
                key_by_index[index] = key
                jobs.setdefault(key, payload)
                labels.setdefault(key, image.get("name") or index)

            def _encode(key: Tuple[str, bool, Optional[str]]) -> Optional[bytes]:
                """Decode, resize and re-encode one job; ``None`` keeps the original."""
                payload, is_exempt, semantic = jobs[key], key[1], key[2]
                try:
                    pil = Image.open(io.BytesIO(payload))
                    pil.load()
                except Exception as error:  # noqa: BLE001 — a bad image keeps its bytes
                    logger.warning(
                        "optimize_glb_textures: unreadable image %r: %s",
                        labels[key],
                        error,
                    )
                    return None
                target = pil.size
                if max_size and max(target) > max_size and not is_exempt:
                    scale = max_size / float(max(target))
                    target = (
                        max(1, round(target[0] * scale)),
                        max(1, round(target[1] * scale)),
                    )
                if is_ktx2 and not is_exempt:
                    # KHR_texture_basisu requires multiple-of-4 dimensions and
                    # a full mip pyramid (generated at encode time); POT
                    # satisfies both at every level and is what the GL/WebGPU
                    # backends want to mip. Snapped DOWN -- an optimize pass
                    # must never grow an asset -- and folded into the max_size
                    # target above so the pixels resample ONCE, not through a
                    # resize-then-snap double pass. No-op for POT sources.
                    target = tuple(
                        max(4, 1 << (max(4, edge).bit_length() - 1)) for edge in target
                    )
                if target != pil.size:
                    pil = pil.resize(target, Image.LANCZOS)
                if is_ktx2 and not is_exempt:
                    codec, srgb = cls.BASIS_BY_SEMANTIC.get(
                        semantic, cls.BASIS_BY_SEMANTIC[None]
                    )
                    from pythontk.file_utils.temp_artifacts import TempArtifacts

                    try:
                        with TempArtifacts("glb_ktx2", policy="scoped") as tmp:
                            out = tmp.path(extension=".ktx2")
                            encoder.encode(
                                pil,
                                out,
                                codec=codec,
                                srgb=srgb,
                                quality=quality if codec == "ETC1S" else None,
                            )
                            with open(out, "rb") as fh:
                                encoded = fh.read()
                    except Exception as error:  # noqa: BLE001 — keep the bytes
                        logger.warning(
                            "optimize_glb_textures: KTX2 encode failed for %r: %s",
                            labels[key],
                            error,
                        )
                        return None
                    # Deliberately NO keep-the-original size rule here: the win
                    # is the GPU-resident format, not the wire, and UASTC
                    # exceeding a source PNG is expected rather than a failure.
                    fb_bytes = fb_mime = None
                    if ktx2_fallback:
                        # The core-readable twin bound as the texture's plain
                        # ``source`` (see the docstring bullet). Same resized
                        # pixels as the KTX2 encode, so the fallback shows what
                        # the basisu path shows. Container by codec class:
                        # ETC1S color -> JPEG at *quality* (PNG when it carries
                        # alpha -- JPEG cannot); UASTC normals/data -> PNG,
                        # where lossy chroma would corrupt the very channels
                        # UASTC was chosen to protect.
                        fb_pil = pil
                        has_alpha = (
                            "A" in fb_pil.getbands()
                            or fb_pil.info.get("transparency") is not None
                        )
                        if codec == "ETC1S" and not has_alpha:
                            fb_format, fb_mime = "JPEG", "image/jpeg"
                            if fb_pil.mode not in ("RGB", "L"):
                                fb_pil = fb_pil.convert("RGB")
                            fb_kwargs = {"quality": quality}
                        else:
                            fb_format, fb_mime = "PNG", "image/png"
                            fb_kwargs = {}
                        fb_buffer = io.BytesIO()
                        try:
                            fb_pil.save(fb_buffer, format=fb_format, **fb_kwargs)
                            fb_bytes = fb_buffer.getvalue()
                        except Exception as error:  # noqa: BLE001 — ship KTX2-only
                            logger.warning(
                                "optimize_glb_textures: fallback %s encode "
                                "failed for %r (ships KTX2-only, viewer must "
                                "support KHR_texture_basisu): %s",
                                fb_format,
                                labels[key],
                                error,
                            )
                            fb_bytes = fb_mime = None
                    return (encoded, fb_bytes, fb_mime)
                # Exempt (lightmap) images in KTX2 mode take the lossless-WebP
                # path -- the mode's docstring bullet says why.
                pil_format = "WEBP" if (is_ktx2 and is_exempt) else image_format
                if mime == "image/png":
                    save_kwargs = {}
                elif is_exempt and pil_format == "WEBP":
                    # Lightmaps must round-trip pixel-exact. Lossy WebP is
                    # YUV 4:2:0 -- chroma at half resolution, quantized --
                    # which on near-black lightmap texels shows as magenta/
                    # green blotching and smears color across atlas rect
                    # borders. Lossless WebP still beats the source PNG.
                    save_kwargs = {"lossless": True, "quality": 100}
                else:
                    save_kwargs = {"quality": quality}
                buffer = io.BytesIO()
                try:
                    pil.save(buffer, format=pil_format, **save_kwargs)
                except Exception as error:  # noqa: BLE001
                    logger.warning(
                        "optimize_glb_textures: %s re-encode failed for %r: %s",
                        pil_format,
                        labels[key],
                        error,
                    )
                    return None
                encoded = buffer.getvalue()
                # Keep the original when the re-encode came out larger.
                return None if len(encoded) >= len(payload) else encoded

            # Phase B: run those jobs concurrently. Pillow does the decode,
            # resize and encode in C with the GIL released, and the encode alone
            # is ~60% of this pass, so threads scale it close to linearly --
            # measured on a production room GLB (27 images, 239 MB of source
            # PNG): 31.8s serial. Threads, not processes: the payloads are
            # already in this process's memory, and pickling hundreds of MB out
            # to workers would cost more than the encode saves.
            #
            # Capped well below the core count on purpose. Each worker holds a
            # fully decoded source (a 4096 RGBA is 67 MB) plus its resize and
            # encode buffers, and this routinely runs inside a DCC that is
            # already holding the scene the export came from -- so the ceiling
            # is host memory, not cores.
            count = max(
                1,
                min(
                    workers or min(cls.OPTIMIZE_WORKERS, os.cpu_count() or 1),
                    len(jobs),
                ),
            )
            if count > 1:
                with ThreadPoolExecutor(
                    max_workers=count, thread_name_prefix="ptk-glb-optimize"
                ) as pool:
                    encoded_by_key = dict(zip(jobs, pool.map(_encode, list(jobs))))
            else:
                encoded_by_key = {key: _encode(key) for key in jobs}

            # Phase C (serial): fan the per-job results back out to the image
            # indices that share them. ``replacements`` keys the image INDEX to
            # its new payload; a KTX2 job returns a tuple carrying the
            # core-readable fallback alongside (``fallbacks`` keys the SAME
            # source index — the twin gets its own appended image at repack).
            replacements: Dict[int, bytes] = {}
            fallbacks: Dict[int, Tuple[bytes, str]] = {}
            for index, key in key_by_index.items():
                encoded = encoded_by_key.get(key)
                if encoded is None:
                    after += len(jobs[key])
                    continue
                if isinstance(encoded, tuple):
                    encoded, fb_bytes, fb_mime = encoded
                    if fb_bytes:
                        fallbacks[index] = (fb_bytes, fb_mime)
                        after += len(fb_bytes)
                replacements[index] = encoded
                after += len(encoded)

            if not replacements:
                return {}

            # Pass 2: repack the BIN. Existing views keep their INDEX (that is
            # what accessors and images reference); only offsets/lengths move.
            views = gltf.get("bufferViews") or []
            # view -> EVERY image that reads it, not just one. FBX2glTF really
            # does point two images at a single bufferView (measured: 4 such
            # pairs on a production room GLB), and co-owners can differ in the
            # one thing that decides their encoding -- a lightmap is exempt from
            # the resize and encodes lossless, its co-owner is not. Recording a
            # single owner per view silently handed the loser the winner's
            # bytes: with the lightmap winning, its co-owner kept full
            # resolution; with the order reversed the LIGHTMAP got resized and
            # lossy-encoded, which is precisely the corruption the structural
            # exemption above exists to prevent.
            image_view_owners: Dict[int, List[int]] = {}
            for idx, img in enumerate(images):
                if "bufferView" in img:
                    image_view_owners.setdefault(img["bufferView"], []).append(idx)

            blob = edit.bin_data
            chunks: List[bytes] = []
            offset = 0
            #: ``(image index, bytes)`` for images that cannot read an existing
            #: view -- carried as bytes because a co-owner may need the
            #: ORIGINAL payload (it had no replacement of its own).
            relocate: List[Tuple[int, bytes]] = []
            for view_index, view in enumerate(views):
                owners = image_view_owners.get(view_index, [])
                # The overwhelmingly common case -- one image owns the view and
                # was re-encoded -- takes the new bytes without ever reading the
                # old ones, which for a 60 MB source PNG is the copy worth
                # skipping. Anything else needs the original, either to keep it
                # or to compare co-owners against.
                if len(owners) == 1 and owners[0] in replacements:
                    data = original = replacements[owners[0]]
                    view.pop("byteStride", None)
                else:
                    start = view.get("byteOffset", 0)
                    original = (
                        bytes(blob[start : start + view["byteLength"]]) if blob else b""
                    )
                    if owners and owners[0] in replacements:
                        data = replacements[owners[0]]
                        view.pop("byteStride", None)
                    else:
                        data = original
                # A co-owner whose final bytes differ from what this view now
                # holds cannot read it; give it its own copy.
                relocate.extend(
                    (idx, replacements.get(idx, original))
                    for idx in owners[1:]
                    if replacements.get(idx, original) != data
                )
                view["byteOffset"] = offset
                view["byteLength"] = len(data)
                padded = data + b"\x00" * ((4 - (len(data) % 4)) % 4)
                chunks.append(padded)
                offset += len(padded)

            # Former data-URI images had no view at all, so they relocate too.
            relocate.extend(
                (index, replacements[index])
                for index, image in enumerate(images)
                if "bufferView" not in image and index in replacements
            )
            # Appended, so no existing view index moves (accessors reference
            # them by index).
            for index, data in relocate:
                views.append(
                    {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
                )
                padded = data + b"\x00" * ((4 - (len(data) % 4)) % 4)
                chunks.append(padded)
                offset += len(padded)
                images[index]["bufferView"] = len(views) - 1
                images[index].pop("uri", None)

            # KTX2 fallbacks are NEW images appended at the end: the KTX2
            # payload keeps the source index (what every prior binding and the
            # digest sidecar reference), and appending moves no existing image
            # or view index. The texture rebind below points plain ``source``
            # here.
            fallback_image_of: Dict[int, int] = {}
            for index, (fb_bytes, fb_mime) in sorted(fallbacks.items()):
                views.append(
                    {"buffer": 0, "byteOffset": offset, "byteLength": len(fb_bytes)}
                )
                padded = fb_bytes + b"\x00" * ((4 - (len(fb_bytes) % 4)) % 4)
                chunks.append(padded)
                offset += len(padded)
                fb_image: Dict[str, Any] = {
                    "mimeType": fb_mime,
                    "bufferView": len(views) - 1,
                }
                name = images[index].get("name")
                if name:
                    fb_image["name"] = f"{name}_fallback"
                images.append(fb_image)
                fallback_image_of[index] = len(images) - 1

            # In KTX2 mode the exempt (lightmap) images took the WebP path, so
            # the mime is per image rather than per run.
            for index in replacements:
                images[index]["mimeType"] = (
                    "image/webp" if (is_ktx2 and key_by_index[index][1]) else mime
                )
            gltf["bufferViews"] = views
            new_bin = b"".join(chunks)
            buffers = gltf.setdefault("buffers", [{}])
            buffers[0]["byteLength"] = len(new_bin)

            webp_images = {
                i for i in replacements if images[i].get("mimeType") == "image/webp"
            }
            ktx2_images = {
                i for i in replacements if images[i].get("mimeType") == "image/ktx2"
            }
            bound_basisu = False
            basisu_without_fallback = False
            if webp_images or ktx2_images:
                for t_index, texture in enumerate(gltf.get("textures") or []):
                    # Resolved through the shadow-aware walk so a re-run of
                    # this pass (or a WebP pass followed by a KTX2 one) rebinds
                    # the texture's EFFECTIVE image, not a stale plain source.
                    src = edit.image_for_texture(t_index)
                    if src in ktx2_images:
                        # A prior WebP binding would now point at KTX2 bytes,
                        # so it goes. Plain ``source`` becomes the appended
                        # core-readable fallback when one exists -- the spec's
                        # own escape hatch, what keeps the extension out of
                        # ``extensionsRequired`` below. Without one (fallback
                        # disabled, or its encode failed) the source is
                        # dropped outright and the extension must be required:
                        # a stale source would hand a non-basisu loader KTX2
                        # bytes labeled as something it can read.
                        extensions = texture.setdefault("extensions", {})
                        extensions.pop("EXT_texture_webp", None)
                        extensions["KHR_texture_basisu"] = {"source": src}
                        fb_index = fallback_image_of.get(src)
                        if fb_index is None:
                            texture.pop("source", None)
                            basisu_without_fallback = True
                        else:
                            texture["source"] = fb_index
                        bound_basisu = True
                    elif src in webp_images:
                        # Standard binding; plain ``source`` stays as fallback.
                        texture.setdefault("extensions", {})["EXT_texture_webp"] = {
                            "source": src
                        }
                        texture.setdefault("source", src)
            if webp_images:
                used = gltf.setdefault("extensionsUsed", [])
                if "EXT_texture_webp" not in used:
                    used.append("EXT_texture_webp")
            elif bound_basisu:
                # A KTX2 pass over a previously WebP-optimized GLB strips the
                # EXT_texture_webp bindings it re-encodes past; a declaration
                # with no remaining user is a validator warning shipped for
                # nothing.
                used = gltf.get("extensionsUsed") or []
                if "EXT_texture_webp" in used and not any(
                    "EXT_texture_webp" in (t.get("extensions") or {})
                    for t in gltf.get("textures") or []
                ):
                    used.remove("EXT_texture_webp")
            if bound_basisu:
                # Equivalent to ``if ktx2_images`` by construction -- the encode
                # gate above skips every image no texture samples -- so no
                # ``image/ktx2`` can ship without this declaration. It escalates
                # to ``extensionsRequired`` only when some binding has no
                # core-readable fallback ``source`` (pure-delivery mode, or a
                # fallback encode failed): with every binding backed by one,
                # a stock importer reads the fallbacks and the file must not
                # demand a capability it can degrade without. Never removed if
                # already present -- a prior pure-delivery pass's bindings are
                # still fallback-less.
                used = gltf.setdefault("extensionsUsed", [])
                if "KHR_texture_basisu" not in used:
                    used.append("KHR_texture_basisu")
                if basisu_without_fallback:
                    required = gltf.setdefault("extensionsRequired", [])
                    if "KHR_texture_basisu" not in required:
                        required.append("KHR_texture_basisu")

            edit.replace_rest(new_bin)
            # Re-encoding invalidated every content address the sidecar
            # recorded at apply time. Restamped from the repacked payloads
            # (image INDICES are untouched above, which is what makes this a
            # refresh rather than a rebuild) so the digests describe the bytes
            # actually delivered. A file with no sidecar is a no-op.
            cls._stamp_sidecar_digests(edit)

        summary = {
            "images": len(replacements),
            "bytes_before": before,
            "bytes_after": after,
        }
        logger.info(
            "optimize_glb_textures: %d image(s), %.1f MB -> %.1f MB.",
            summary["images"],
            before / 1e6,
            after / 1e6,
        )
        return summary

    @classmethod
    def set_glb_metallic_roughness(
        cls, glb: GlbTarget, metallic_roughness: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """Pack and write the ORM (metallic/roughness) texture into a GLB, by name.

        The third sibling of :meth:`set_glb_base_color` / :meth:`set_glb_emissive`,
        for the most *destructive* form of the same FBX translation gap. Measured on
        a production room (Maya 2025 StingrayPBS -> FBX2glTF): the converter packed
        the material's roughness+metallic into a **solid-white** ORM -- and since
        glTF reads metallic from the blue channel, that renders the whole room
        metallic=1. A fully metallic surface has no diffuse response, and a baked
        lightmap contributes *only* to diffuse -- so in a lightmap-lit viewer
        (which turns its own lights off) the failure compounds to pure black, the
        single symptom least traceable back to "your roughness texture was lost in
        translation".

        Packing follows the glTF convention (R=occlusion, G=roughness, B=metallic)
        through :meth:`MapFactory.pack_orm_texture` -- the registry's one ORM
        packer -- with R filled white when no AO source resolves, so the
        occlusion binding below stays neutral. The packed image embeds through
        the same session cache as every other writer, so two materials naming the
        same source maps share one embed.

        The packed image is also bound as the material's ``occlusionTexture``
        (glTF's packed-ORM idiom -- occlusion is read from that slot alone, so
        an unbound R channel is dead payload) whenever the slot is free or
        still points at the converted ORM this write replaces; an authored
        separate AO map is never displaced. A lightmap applied afterwards
        recognises the ORM binding by its shared texture index and takes the
        slot silently (:meth:`apply_glb_lightmaps`).

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            metallic_roughness: ``{material_name: {"metallic": path,
                "roughness": path, "occlusion": path}}``. All keys optional --
                a missing metallic fills black (non-metal), missing roughness
                fills black, missing occlusion fills white -- but an entry with
                no readable map at all writes nothing.

        Returns:
            List of records: ``material``, ``metallic``, ``roughness``.
        """
        if not metallic_roughness:
            return []

        from pythontk.core_utils.engines.textures.map_factory import MapFactory

        records: List[Dict] = []
        packed_cache: Dict[Tuple, Optional[int]] = {}
        #: Source paths that actually reached a material, for the summary below.
        #: Collected as the loop writes rather than read back off
        #: *metallic_roughness*, because that input includes materials
        #: `_match_glb_materials` found no match for and maps whose pack failed
        #: -- counting those makes the headline claim work that never happened.
        written: List[str] = []
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            for name, spec, mat in cls._match_glb_materials(
                gltf, metallic_roughness, "set_glb_metallic_roughness"
            ):
                sources = tuple(
                    spec.get(key) or None
                    for key in ("occlusion", "roughness", "metallic")
                )
                if not any(sources):
                    continue
                if sources in packed_cache:
                    tex_index = packed_cache[sources]
                else:
                    tex_index = None
                    try:
                        image = MapFactory.pack_orm_texture(*sources, save=False)
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        tex_index = cls._embed_image_bytes(
                            edit,
                            "|".join(str(s) for s in sources),
                            buffer.getvalue(),
                            "image/png",
                            name=f"orm_{os.path.basename(sources[2] or sources[1] or '')}",
                        )
                    except Exception as error:  # noqa: BLE001 — a bad map must not cost the section
                        logger.warning(
                            "set_glb_metallic_roughness: packing failed for %r: %s",
                            name,
                            error,
                        )
                    packed_cache[sources] = tex_index
                if tex_index is None:
                    continue

                pbr = mat.setdefault("pbrMetallicRoughness", {})
                prior = (pbr.get("metallicRoughnessTexture") or {}).get("index")
                pbr["metallicRoughnessTexture"] = {"index": tex_index}
                # The map is authoritative; factors are multipliers on it.
                pbr["metallicFactor"] = 1.0
                pbr["roughnessFactor"] = 1.0
                # glTF reads occlusion ONLY from ``occlusionTexture``, so the
                # AO packed into R is dead weight unless it is bound there --
                # the same image in both slots is the spec's own packed-ORM
                # idiom. Bind when the slot is free, and REPOINT it when it
                # still names the converted ORM this write just replaced
                # (FBX2glTF binds its own packing there, and leaving that
                # reference samples the stale -- measured, often solid-white
                # -- image). A separate authored AO map is left alone. R
                # fills white when no AO source resolved, so the binding is
                # neutral in that case, never wrong. A lightmap pass running
                # after this recognises the shared index and takes the slot
                # (see apply_glb_lightmaps) -- a bake carries its own
                # occlusion, computed with real bounce.
                occlusion = mat.get("occlusionTexture")
                if occlusion is None or (
                    prior is not None and occlusion.get("index") == prior
                ):
                    mat["occlusionTexture"] = {"index": tex_index}
                written.extend(src for src in sources if isinstance(src, str))
                records.append(
                    {
                        "material": name,
                        "metallic": sources[2],
                        "roughness": sources[1],
                    }
                )

            if not records:
                return []
            edit.dirty = True

        # One highlighted headline for the whole pass. The per-map detail is
        # already logged by `pack_orm_texture`, but in a DCC those lines arrive
        # amid hundreds of others and the artist has no reason to be reading the
        # log at all -- so the actionable summary gets the `highlight` preset
        # (rendered by the DCC log handler; a plain handler ignores the extra
        # and prints the same text). Named counts, not paths: the detail lines
        # carry those, and this has to stay one scannable line.
        foreign = MapFactory.foreign_packings(written)
        if foreign:
            by_type: Dict[str, int] = {}
            for map_type in foreign.values():
                by_type[map_type] = by_type.get(map_type, 0) + 1
            logger.warning(
                "Materials repacked from a non-glTF mask packing: %s. "
                "They render correctly, but roughness is reconstructed rather "
                "than authored -- re-export the source set for an ORM target.",
                ", ".join(
                    f"{count} {name} map{'' if count == 1 else 's'}"
                    for name, count in sorted(by_type.items())
                ),
                extra={"preset": "highlight"},
            )

        return records

    #: glTF fixes the metallic/roughness packing in the SPEC -- occlusion in R,
    #: roughness in G, metallic in B -- so a delivered
    #: ``metallicRoughnessTexture`` is read that way by every consumer whatever
    #: the authoring set was packed for. Held here as glTF's constant rather
    #: than looked up from :class:`MapRegistry`: the registry describes the map
    #: types this pipeline AUTHORS, and reading a spec fact out of a mutable
    #: taxonomy would let an edit there silently change what this checks.
    #: ``test_glb_orm_layout_matches_the_registry`` pins the two together, so
    #: the taxonomy cannot drift away from the spec unnoticed either.
    GLTF_ORM_CHANNELS = {"R": "Ambient_Occlusion", "G": "Roughness", "B": "Metallic"}

    #: The channel whose full-white value is destructive rather than neutral.
    #: R full = no occlusion and G full = fully rough are both ordinary; B full
    #: is metallic=1, which zeroes diffuse response.
    _ORM_HARMFUL_CHANNEL = "B"

    #: The ``finding`` values :meth:`suspect_orm_materials` reports. Named
    #: because callers FILTER on them -- the two are routed to different
    #: audiences (see the method) -- and a filter comparing against a literal
    #: typo fails silently, in the direction of reporting nothing.
    ORM_FINDING_METALLIC_FULL = "metallic=1 everywhere"
    ORM_FINDING_UNVALIDATED = "unvalidated"

    @classmethod
    def suspect_orm_materials(
        cls, glb: GlbTarget, *, described: Optional[Iterable[str]] = None
    ) -> Dict[str, Dict[str, str]]:
        """Materials whose delivered ORM binding this pipeline never validated.

        Two findings, one walk, because both are the same question asked of the
        same slot -- *is what a consumer will read here what the scene meant?*

        ``metallic=1 everywhere``
            The measured production failure. FBX2glTF white-fills a grayscale
            ("L"-mode) PBR source; glTF reads metallic from **blue**; a
            solid-white packing therefore renders metallic=1, which has no
            diffuse response, and a baked lightmap contributes to diffuse
            alone -- so a lightmapped viewer renders it pure black.

        ``unvalidated``
            The material carries an ORM binding that the envelope never
            described, so nothing in this pipeline checked its channel
            semantics. This is the case a whiteness test cannot see: a mask map
            packed for another engine (Unity's MaskMap is R=Metallic,
            G=Occlusion, B=Detail) that reaches the GLB unrepaired is read as
            ORM and misinterpreted channel for channel, while looking like
            perfectly ordinary image data. Reported only when the caller says
            what WAS described, since only they know.

        Parameters:
            glb: Path to a binary glTF (.glb) or an open :class:`GlbEdit`. Read
                only -- nothing here marks the session dirty.
            described: Material names the envelope's ``metallic_roughness``
                section covers, i.e. the ones a repair pass validated and
                rewrote. Their packing comes from the authoring maps rather
                than from the converter, so they are exempt from both findings
                (any whiteness in them is a source question
                :meth:`MapFactory.pack_orm_texture` already logs per map).
                ``None`` means "nothing is known to be described", which
                suppresses the ``unvalidated`` finding rather than reporting
                every material.

        Returns:
            ``{material name: {"image": label, "finding": str}}``; empty when
            none, which is the common case and the one that costs no decode.
        """
        known = set(described) if described is not None else None
        findings: Dict[str, Dict[str, str]] = {}
        with cls.open_glb(glb) as edit:
            for mat in edit.materials:
                name = mat.get("name")
                if not name or (known is not None and name in known):
                    continue
                pbr = mat.get("pbrMetallicRoughness") or {}
                tex = (pbr.get("metallicRoughnessTexture") or {}).get("index")
                if tex is None:
                    continue
                img_idx = edit.image_for_texture(tex)
                if img_idx is None:
                    continue
                # A zero factor cancels the texture, so nothing it carries can
                # be destructive -- but it is still unvalidated data.
                harmful = pbr.get("metallicFactor", 1.0) != 0 and edit.channel_extrema(
                    img_idx, cls._ORM_HARMFUL_CHANNEL
                ) == (255, 255)
                if harmful:
                    finding = cls.ORM_FINDING_METALLIC_FULL
                elif known is not None:
                    finding = cls.ORM_FINDING_UNVALIDATED
                else:
                    continue
                findings[name] = {
                    "image": edit.image_label(img_idx),
                    "finding": finding,
                }
        return findings

    @staticmethod
    def _match_glb_materials(
        gltf: dict, entries: Dict[str, Dict[str, Any]], caller: str
    ) -> List[Tuple[str, Dict[str, Any], dict]]:
        """Pair sidecar *entries* with the GLB materials they name.

        The shared front half of every by-name channel writer here: resolve
        each entry against the GLB's material list and report the misses in
        one loud line. Loud, with both name sets, because a name the converter
        renamed makes the write a total no-op and "my emissive is missing" is
        indistinguishable from "the channel was never read" unless the
        mismatch says so.
        """
        by_name = {
            m.get("name"): m for m in gltf.get("materials", []) or [] if m.get("name")
        }
        matched = [
            (name, spec, by_name[name])
            for name, spec in entries.items()
            if name in by_name
        ]
        unmatched = sorted(set(entries) - set(by_name))
        if unmatched:
            logger.warning(
                "%s: %s material(s) had no match in the GLB and were skipped: %s. "
                "GLB materials are: %s.",
                caller,
                len(unmatched),
                unmatched,
                sorted(by_name) or "<none>",
            )
        return matched

    @classmethod
    def set_glb_emissive(
        cls, glb: GlbTarget, emissive: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """Write emissive color / texture into a GLB's materials, by name.

        Repairs the channel Maya's FBX exporter simply drops. It maps emissive
        only for its own legacy shading models (lambert / blinn / phong via
        ``incandescence``); ``aiStandardSurface``, ``StingrayPBS`` and openPBR
        emission never reach the FBX at all, so the GLB has no ``emissiveFactor``
        to correct and the surface previews as unlit. Measured against Maya
        2025 + MtoA: lambert and blinn arrive with ``emissiveFactor``, the other
        two arrive with the key absent entirely.

        Emissive intensity above 1.0 is preserved rather than clipped, by
        normalizing the color and carrying the magnitude in
        ``KHR_materials_emissive_strength`` — glTF's base ``emissiveFactor`` is
        LDR, so a Maya emission of 5.0 would otherwise flatten to 1.0 and lose
        exactly the over-bright look that motivates using it. The extension is
        declared in ``extensionsUsed`` only when actually applied; loaders that
        don't implement it still get a sensible clamped color.

        Textures are embedded as ``data:`` URIs *for the duration of the
        session*, which keeps every edit inside the JSON chunk — no buffer
        offsets to recompute, which is the part of GLB surgery that silently
        corrupts a file. :meth:`_relocate_embedded_images` then appends them to
        the BIN once on close, so the file that lands pays none of base64's
        ~33% premium. Repeated paths are embedded once per session, so a map
        used as both base colour and emissive costs one copy, not one per
        channel.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            emissive: ``{material_name: {"color": (r, g, b), "texture": path}}``.
                Both keys optional; a texture with no color implies white.
                Names not present in the GLB are reported, not raised.

        Returns:
            List of records: ``material``, ``factor``, ``strength``, ``texture``.
        """
        if not emissive:
            return []

        records: List[Dict] = []
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            used_strength = False
            for name, spec, mat in cls._match_glb_materials(
                gltf, emissive, "set_glb_emissive"
            ):
                color = list(spec.get("color") or (1.0, 1.0, 1.0))[:3]
                if len(color) < 3:
                    color += [0.0] * (3 - len(color))

                # A zero colour is zero emission -- and paired with a texture it
                # is actively harmful rather than merely redundant: glTF
                # emission is ``emissiveFactor * emissiveTexture``, so writing
                # both multiplies the map to black while still reporting a
                # successful transfer. Tested before embedding, so a skipped
                # material leaves no orphaned image in the file.
                if all(c <= 0.0 for c in color):
                    continue

                peak = max(color)
                strength = None
                if peak > 1.0:
                    color = [c / peak for c in color]
                    strength = peak
                    mat.setdefault("extensions", {})[
                        "KHR_materials_emissive_strength"
                    ] = {"emissiveStrength": strength}
                    used_strength = True

                tex_path = spec.get("texture")
                tex_index = cls._embed_image(edit, tex_path) if tex_path else None
                if tex_index is not None:
                    mat["emissiveTexture"] = {"index": tex_index}

                mat["emissiveFactor"] = color

                records.append(
                    {
                        "material": name,
                        "factor": color,
                        "strength": strength,
                        "texture": tex_path if tex_index is not None else None,
                    }
                )

            if not records:
                # `dirty` is left as found rather than cleared: on a shared
                # session a sibling writer's edits may already be pending, and
                # this one having nothing to say is no reason to drop them.
                return []

            if used_strength:
                used = gltf.setdefault("extensionsUsed", [])
                if "KHR_materials_emissive_strength" not in used:
                    used.append("KHR_materials_emissive_strength")

            cls._prune_empty_containers(gltf)
            edit.dirty = True

        return records

    @classmethod
    def prune_glb_unreferenced_textures(cls, glb: GlbTarget) -> Dict[str, int]:
        """Drop textures no material samples, and the images/bufferViews only they used.

        The applier-tail sweep. Every channel writer here REBINDS a slot to the
        image it embeds and leaves whatever that slot named before in place --
        which for the ORM repack is FBX2glTF's own ``ao_met_rough_<mat>``
        packing, a full-size PNG in the BIN chunk. Measured on a production
        delivery (HOOKS_PINS.glb): 4 images, 1 unreferenced, 2 MB of dead
        payload the reviewer flagged as an orphaned texture -- one per repacked
        material, every export. Writers cannot prune for themselves: a texture
        is only provably dead once EVERY writer on the session has run, and
        dropping an image renumbers everything after it.

        Referenced means sampled by a material: any ``textureInfo`` in the
        material tree (a dict under a key ending in ``Texture`` with an integer
        ``index`` -- the spec's own naming for the core slots and every
        ``KHR_materials_*`` extension). Images are kept when a surviving
        texture reads them through ``source`` or an extension source
        (``KHR_texture_basisu``, ``EXT_texture_webp``). A bufferView is dropped
        only when the pruned images were its ONLY readers -- FBX2glTF does point
        two images at one view, and a view could in principle be shared with an
        accessor. Every surviving index is remapped in place: texture indices
        across the material tree and the session's embed cache, image indices
        in ``textures``, and ``bufferView`` keys anywhere in the document (the
        key is spec-uniform, so a generic walk covers accessors, sparse
        indices/values, images and compression extensions alike). The BIN is
        rebuilt from the surviving views only, 4-byte padded like every
        repack here.

        Runs at the tail of :meth:`apply_scene_sidecar`, BEFORE the embedded
        texture map is built, so the indices that map records describe the
        delivered file. Safe to run standalone on any GLB; a file with nothing
        to drop is not rewritten.

        Returns:
            ``{"textures": n, "images": n, "bytes": n}`` -- what was dropped;
            ``bytes`` is BIN payload reclaimed (a data-URI image counts 0 here,
            its saving shows up in the JSON chunk).
        """
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            textures = gltf.get("textures") or []
            images = gltf.get("images") or []
            if not textures and not images:
                return {"textures": 0, "images": 0, "bytes": 0}
            # Image references the material walk cannot see. Every core and
            # KHR_materials_* binding goes through a texture, but this
            # extension names images DIRECTLY from the root -- pruning under
            # it would renumber indices it holds. Bail whole rather than
            # guess: dead payload beats a broken file.
            foreign = cls._IMAGE_REFERRING_EXTENSIONS & set(
                gltf.get("extensionsUsed") or []
            )
            if foreign:
                logger.info(
                    "prune_glb_unreferenced_textures: skipped, %s references "
                    "images outside the material tree.",
                    ", ".join(sorted(foreign)),
                )
                return {"textures": 0, "images": 0, "bytes": 0}
            # The BIN rebuild below re-slices every kept view out of the single
            # GLB-embedded buffer (0). A view on any other buffer (external
            # URI) would keep its ``buffer`` but get a byteOffset into the
            # rebuilt BIN -- garbage reads. Same single-buffer assumption as
            # optimize_glb_textures; bail rather than guess.
            if any(v.get("buffer", 0) != 0 for v in gltf.get("bufferViews") or []):
                logger.info(
                    "prune_glb_unreferenced_textures: skipped, the file has "
                    "bufferViews outside the embedded BIN (buffer 0)."
                )
                return {"textures": 0, "images": 0, "bytes": 0}

            # --- what the materials actually sample --------------------------
            def _texture_refs(node, out):
                """Yield every textureInfo dict under *node* (materials tree)."""
                if isinstance(node, dict):
                    for key, value in node.items():
                        if (
                            key.endswith("Texture")
                            and isinstance(value, dict)
                            and isinstance(value.get("index"), int)
                        ):
                            out.append(value)
                        _texture_refs(value, out)
                elif isinstance(node, list):
                    for item in node:
                        _texture_refs(item, out)
                return out

            refs = _texture_refs(gltf.get("materials") or [], [])
            live_textures = {
                r["index"] for r in refs if 0 <= r["index"] < len(textures)
            }
            texture_map = {}
            for old in range(len(textures)):
                if old in live_textures:
                    texture_map[old] = len(texture_map)

            def _image_sources(texture):
                yield texture.get("source")
                for ext in (texture.get("extensions") or {}).values():
                    if isinstance(ext, dict):
                        yield ext.get("source")

            live_images = {
                src
                for old in texture_map
                for src in _image_sources(textures[old])
                if isinstance(src, int) and 0 <= src < len(images)
            }
            image_map = {}
            for old in range(len(images)):
                if old in live_images:
                    image_map[old] = len(image_map)

            dropped_textures = len(textures) - len(texture_map)
            dropped_images = len(images) - len(image_map)
            if not dropped_textures and not dropped_images:
                return {"textures": 0, "images": 0, "bytes": 0}

            # --- views only the dropped images read --------------------------
            views = gltf.get("bufferViews") or []
            dead_views = {
                img.get("bufferView")
                for old, img in enumerate(images)
                if old not in image_map and isinstance(img.get("bufferView"), int)
            }

            def _view_refs(node, out, skip):
                """Every ``bufferView`` index referenced outside *skip*."""
                if node is skip:
                    return out
                if isinstance(node, dict):
                    for key, value in node.items():
                        if key == "bufferView" and isinstance(value, int):
                            out.add(value)
                        else:
                            _view_refs(value, out, skip)
                elif isinstance(node, list):
                    for item in node:
                        _view_refs(item, out, skip)
                return out

            still_read = set()
            for old, img in enumerate(images):
                if old in image_map:
                    _view_refs(img, still_read, None)
            _view_refs(gltf, still_read, images)  # everything but the images
            dead_views -= still_read
            dead_views = {v for v in dead_views if 0 <= v < len(views)}

            reclaimed = 0
            if dead_views:
                blob = edit.bin_data
                view_map = {}
                chunks: List[bytes] = []
                offset = 0
                kept_views = []
                for old, view in enumerate(views):
                    if old in dead_views:
                        reclaimed += view.get("byteLength", 0)
                        continue
                    start = view.get("byteOffset", 0)
                    data = (
                        bytes(blob[start : start + view["byteLength"]]) if blob else b""
                    )
                    view = dict(view)
                    view["byteOffset"] = offset
                    padded = data + b"\x00" * ((4 - (len(data) % 4)) % 4)
                    chunks.append(padded)
                    offset += len(padded)
                    view_map[old] = len(kept_views)
                    kept_views.append(view)

                def _remap_views(node):
                    if isinstance(node, dict):
                        for key, value in list(node.items()):
                            if key == "bufferView" and isinstance(value, int):
                                if value in view_map:
                                    node[key] = view_map[value]
                            else:
                                _remap_views(value)
                    elif isinstance(node, list):
                        for item in node:
                            _remap_views(item)

                gltf["bufferViews"] = kept_views
                # Images are rebuilt below from the survivors; remap those now
                # so the dropped ones (which name dead views) are never walked.
                surviving_images = [images[old] for old in image_map]
                _remap_views(surviving_images)
                gltf["images"] = surviving_images
                images = surviving_images
                _remap_views({k: v for k, v in gltf.items() if k != "images"})
                new_bin = b"".join(chunks)
                buffers = gltf.setdefault("buffers", [{}])
                buffers[0]["byteLength"] = len(new_bin)
                edit.replace_rest(new_bin)
            else:
                gltf["images"] = [images[old] for old in image_map]

            # --- renumber what survives ---------------------------------------
            kept_textures = []
            for old in texture_map:
                texture = textures[old]
                if isinstance(texture.get("source"), int):
                    texture["source"] = image_map.get(
                        texture["source"], texture["source"]
                    )
                for ext in (texture.get("extensions") or {}).values():
                    if isinstance(ext, dict) and isinstance(ext.get("source"), int):
                        ext["source"] = image_map.get(ext["source"], ext["source"])
                kept_textures.append(texture)
            gltf["textures"] = kept_textures
            for ref in refs:
                if ref["index"] in texture_map:
                    ref["index"] = texture_map[ref["index"]]
            edit.embedded = {
                key: texture_map[index]
                for key, index in edit.embedded.items()
                if index in texture_map
            }
            edit._image_digests = None
            cls._prune_empty_containers(gltf)
            edit.dirty = True

        logger.info(
            "prune_glb_unreferenced_textures: dropped %d texture(s), %d image(s), "
            "%.1f MB of BIN payload.",
            dropped_textures,
            dropped_images,
            reclaimed / 1e6,
        )
        return {
            "textures": dropped_textures,
            "images": dropped_images,
            "bytes": reclaimed,
        }

    @staticmethod
    def _prune_empty_containers(gltf: dict) -> None:
        """Drop container arrays an edit left empty.

        Emitting ``"samplers": []`` is invalid per the glTF schema (minItems 1),
        and every channel writer can leave one behind when its textures all
        failed to embed -- so the cleanup belongs in one place rather than
        copied into the tail of each.
        """
        for key in ("images", "textures", "samplers"):
            if not gltf.get(key):
                gltf.pop(key, None)

    @classmethod
    def _embed_image(cls, edit: "MeshConvert.GlbEdit", path: str) -> Optional[int]:
        """Embed *path* as an image and return its texture index.

        Shared by every channel writer here. The payload is staged as a
        ``data:`` URI so the whole edit stays inside the JSON chunk -- no
        buffer offsets to recompute, which is the part of GLB surgery that
        silently corrupts a file -- and
        :meth:`_relocate_embedded_images` moves it into the BIN when the
        session closes, so base64's ~33% premium never reaches disk.

        Repeated paths resolve to one image via the session's embed cache, so
        the sharing now spans every writer on that session rather than only the
        one call -- a map assigned as both base colour and emissive used to be
        written into the file twice.
        """
        cache = edit.embedded
        if path in cache:
            return cache[path]
        if not os.path.isfile(path):
            logger.warning("GLB texture embed: file not found: %s", path)
            return None
        mime = cls.IMAGE_MIME_TYPES.get(os.path.splitext(path)[1].lower())
        if mime is not None:
            # Guarded like the re-encode branch below, which this asymmetry had
            # left as the only safe one: a PNG that exists but cannot be read
            # (permissions, a network path that dropped, deleted between the
            # isfile check and here) raised straight out of the channel writer,
            # while the exact same failure on a TGA warned and skipped. It also
            # made this the one file operation left inside an applier, so a
            # locked texture could abort a whole sidecar section.
            try:
                with open(path, "rb") as f:
                    raw = f.read()
            except OSError as error:
                logger.warning("GLB texture embed: could not read %s: %s", path, error)
                return None
        else:
            # glTF accepts only PNG/JPEG, but TGA/TIFF/BMP are routine in a
            # DCC source tree -- TGA especially, all over game art -- and
            # written as-is they would be an unloadable data URI rather than a
            # reported failure. Re-encode to PNG when Pillow can read them;
            # only what it cannot (EXR, or no Pillow at all) is rejected.
            raw = cls._reencode_as_png(path)
            if raw is None:
                logger.warning("GLB texture embed: unsupported image type: %s", path)
                return None
            mime = "image/png"

        return cls._embed_image_bytes(edit, path, raw, mime)

    @classmethod
    def _embed_image_bytes(
        cls,
        edit: "MeshConvert.GlbEdit",
        cache_key: str,
        raw: bytes,
        mime: str = "image/png",
        name: Optional[str] = None,
        clamp: bool = False,
    ) -> int:
        """Embed already-encoded image bytes; return the texture index.

        The tail of :meth:`_embed_image`, split out so writers that *produce* their
        bytes (the lightmap applier encodes EXR -> PNG in memory) share the same
        images/textures/samplers plumbing and the same session dedupe cache --
        keyed by *cache_key* (the SOURCE file's abspath), so an atlas shared by six
        objects costs one embed.

        ``clamp=True`` samples the new texture CLAMP_TO_EDGE instead of the
        default REPEAT -- for atlases. Atlas rects can legally extend past
        [0, 1] (an island crop folded into the published rect), and REPEAT
        turns any tap past an atlas edge into the OPPOSITE edge's texels --
        an unrelated object's lighting. Clamping returns the nearest real
        content instead. Only the texture created here is affected; sampler 0
        (shared by the file's ordinary materials) keeps its wrap.
        """
        gltf, cache = edit.gltf, edit.embedded
        if cache_key in cache:
            return cache[cache_key]
        # Before base64'ing a second copy, check whether the file already holds
        # these exact bytes (the converter's own embedded media, normally).
        digest = hashlib.sha256(raw).hexdigest()
        digests = edit.image_digests
        existing = digests.get(digest)
        if existing is not None:
            cache[cache_key] = edit.texture_for_image(existing)
            return cache[cache_key]
        images = gltf.setdefault("images", [])
        textures = gltf.setdefault("textures", [])
        samplers = gltf.setdefault("samplers", [])
        data = base64.b64encode(raw).decode("ascii")
        entry = {
            "name": name or os.path.basename(cache_key),
            "uri": f"data:{mime};base64,{data}",
            "mimeType": mime,
        }
        images.append(entry)
        # Carried as a data URI only until the session closes: writing it here
        # keeps every editor's edit inside the JSON chunk (no offsets to
        # recompute mid-session), and the single relocation pass on close moves
        # it into the BIN. See :meth:`_relocate_embedded_images`.
        edit.pending_images.append(entry)
        # Registered so a LATER embed of the same bytes under a different source
        # path reuses this one instead of adding a third copy.
        digests.setdefault(digest, len(images) - 1)
        if not samplers:  # one repeat sampler is enough for a preview
            samplers.append({"wrapS": 10497, "wrapT": 10497})
        sampler_index = 0
        if clamp:
            wanted = {"wrapS": 33071, "wrapT": 33071}  # CLAMP_TO_EDGE
            sampler_index = next(
                (i for i, s in enumerate(samplers) if s == wanted), None
            )
            if sampler_index is None:
                samplers.append(dict(wanted))
                sampler_index = len(samplers) - 1
        # Named for the same reason the image is: FBX2glTF names the textures it
        # writes, so an unnamed one is a tell that a later pass added it -- and
        # the name is the only human-readable handle on which map a slot samples
        # when someone opens the deliverable to check it.
        texture: Dict[str, Any] = {
            "source": len(images) - 1,
            "sampler": sampler_index,
        }
        image_name = entry.get("name")
        if image_name:
            texture["name"] = os.path.splitext(image_name)[0]
        textures.append(texture)
        cache[cache_key] = len(textures) - 1
        return cache[cache_key]

    @staticmethod
    def _reencode_as_png(path: str) -> Optional[bytes]:
        """PNG bytes for an image glTF can't hold natively, or ``None``.

        Pillow is deliberately optional (this package's zero-dep rule): without
        it, or for a format it can't read (EXR), the caller falls back to the
        same rejected-by-name warning as before. The bare ``save`` is tried
        first so PNG-representable modes (16-bit gray, palette) pass through
        losslessly; only modes PNG itself refuses (CMYK, float) are converted.
        """
        try:
            from PIL import Image
        except ImportError:
            # Distinguishable in the log from a truly unsupported format: the
            # caller's warning says "unsupported image type", which would send
            # someone hunting a format problem when the fix is `pip install
            # pillow`.
            logger.debug(
                "GLB texture embed: Pillow unavailable; cannot re-encode %s", path
            )
            return None
        import io

        try:
            with Image.open(path) as img:
                buf = io.BytesIO()
                try:
                    img.save(buf, format="PNG")
                except (OSError, ValueError):
                    buf = io.BytesIO()
                    mode = "RGBA" if "A" in img.getbands() else "RGB"
                    img.convert(mode).save(buf, format="PNG")
                return buf.getvalue()
        except (OSError, ValueError) as error:
            logger.warning("GLB texture embed: could not re-encode %s: %s", path, error)
            return None

    @classmethod
    def set_glb_base_color(
        cls, glb: GlbTarget, base_color: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """Write base colour / texture into a GLB's materials, by name.

        The sibling of :meth:`set_glb_emissive`, for the same underlying gap.
        Measured against Maya 2025 + MtoA: ``lambert`` / ``blinn`` / ``phong``
        carry their colour through FBX (scaled by Maya's ``diffuse`` weight),
        while ``aiStandardSurface`` and ``standardSurface`` arrive with
        ``baseColorFactor`` at a flat **[1,1,1,1]** -- Maya's exporter does not
        map them at all, so every modern shader previews as white plastic. That
        also swamps emissive: a surface already at white leaves no headroom for
        an additive term to read against.

        Unlike emissive there is no strength extension and no zero-skip: black
        is a legitimate base colour, so values are clamped into [0,1] rather
        than normalized, and an all-zero colour is written as authored.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            base_color: ``{material_name: {"color": (r, g, b), "texture": path}}``.
                Both keys optional. Names absent from the GLB are reported.

        Returns:
            List of records: ``material``, ``factor``, ``texture``.
        """
        if not base_color:
            return []

        records: List[Dict] = []
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            for name, spec, mat in cls._match_glb_materials(
                gltf, base_color, "set_glb_base_color"
            ):
                pbr = mat.setdefault("pbrMetallicRoughness", {})
                entry_written = False

                tex_path = spec.get("texture")
                tex_index = None
                if tex_path:
                    tex_index = cls._embed_image(edit, tex_path)
                    if tex_index is not None:
                        pbr["baseColorTexture"] = {"index": tex_index}
                        entry_written = True

                color = spec.get("color")
                factor = None
                if color is not None:
                    factor = [min(1.0, max(0.0, float(c))) for c in list(color)[:3]]
                    while len(factor) < 3:
                        factor.append(0.0)
                    # Preserve any alpha the converter already established; this
                    # writer is about colour and must not silently turn a
                    # transparent material opaque.
                    existing = pbr.get("baseColorFactor") or [1.0, 1.0, 1.0, 1.0]
                    factor.append(existing[3] if len(existing) > 3 else 1.0)
                    pbr["baseColorFactor"] = factor
                    entry_written = True
                elif tex_index is not None:
                    # A texture with no colour beside it: the TEXTURE is the
                    # albedo, so the factor must be neutral or it tints what it
                    # was only meant to carry. glTF multiplies the two, and the
                    # converter's fallback is not authored intent -- measured on
                    # a production room (StingrayPBS -> FBX2glTF 0.13.1): the FBX
                    # carries no DiffuseColor at all for a Stingray material, so
                    # every material arrived at a flat 0.5 grey and the whole
                    # room shipped at HALF its authored albedo, texture correctly
                    # rebound on top of it. The Maya side confirms the 0.5 is not
                    # a choice: it is StingrayPBS's `-dv 0.5` attribute default,
                    # inert under `use_color_map`, never set by the artist.
                    #
                    # Alpha is preserved for the same reason the colour branch
                    # preserves it, and the write is skipped when the factor is
                    # already neutral so an untouched material stays byte-stable.
                    existing = pbr.get("baseColorFactor") or [1.0, 1.0, 1.0, 1.0]
                    if [float(c) for c in existing[:3]] != [1.0, 1.0, 1.0]:
                        factor = [
                            1.0,
                            1.0,
                            1.0,
                            existing[3] if len(existing) > 3 else 1.0,
                        ]
                        pbr["baseColorFactor"] = factor

                if entry_written:
                    records.append(
                        {
                            "material": name,
                            "factor": factor,
                            # Gate on the embed RESULT, not the request: an image
                            # that could not be embedded (missing, EXR, no Pillow)
                            # writes no baseColorTexture, so reporting its path
                            # claims a channel the GLB does not carry. Mirrors the
                            # emissive writer.
                            "texture": tex_path if tex_index is not None else None,
                        }
                    )

            if not records:
                # As in `set_glb_emissive`: on a shared session another writer's
                # pending edits are not this one's to discard.
                return []

            cls._prune_empty_containers(gltf)
            edit.dirty = True

        return records
