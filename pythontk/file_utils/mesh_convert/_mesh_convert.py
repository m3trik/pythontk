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
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

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
    #: FLOOR for a conversion, in seconds. Kept at the historical value so no
    #: small file converts on a shorter leash than before; the effective budget
    #: for a large one is derived by :meth:`conversion_timeout`.
    DEFAULT_TIMEOUT = 300
    #: Seconds of conversion budget allowed per MB of input FBX. Measured on a
    #: production assembly (250 MB FBX, 757 meshes, 98k triangles, 30 embedded
    #: textures): ~0.8 s/MB with the machine otherwise idle. 3 s/MB was that
    #: with room for a workstation doing something else -- and a second
    #: incident (2026-08-31: a 173 MB assembly timed out at 495s while a test
    #: suite shared the machine) blew through it, discarding another finished
    #: export's GLB. The asymmetry only sharpens with margin: a generous
    #: budget merely delays the report of a genuinely hung process, a tight
    #: one discards a deliverable.
    TIMEOUT_SECONDS_PER_MB = 10.0
    #: ``timeout=AUTO_TIMEOUT`` (the default) derives the budget from the input.
    #: Negative because no real timeout can be, so it cannot collide with a
    #: caller's value -- and unlike ``None`` it is not already meaningful to
    #: ``subprocess.run``, where None means "wait forever".
    AUTO_TIMEOUT = -1.0

    @classmethod
    def conversion_timeout(cls, src: str) -> float:
        """Seconds to allow FBX2glTF for *src* -- :attr:`DEFAULT_TIMEOUT` or more.

        A flat budget cannot fit both a prop and a production assembly, and the
        cost of getting it wrong is asymmetric: too generous only delays the
        report of a genuinely hung process, while too tight discards a finished
        export's whole deliverable ("produced no file") on a scene that was
        converting normally. It also fails by wall-clock rather than by content,
        so it passes on a quiet machine and fails mid-workday -- which is how it
        reached production unnoticed.

        An unreadable size falls back to the floor: a budget must never be the
        reason a conversion is not attempted.
        """
        try:
            megabytes = os.path.getsize(src) / (1024 * 1024)
        except OSError:
            return float(cls.DEFAULT_TIMEOUT)
        return float(max(cls.DEFAULT_TIMEOUT, megabytes * cls.TIMEOUT_SECONDS_PER_MB))

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
        "occlusion instead. When 'extras.animation_web' is present it names "
        "every animation in this file by its index, says which ones a shot "
        "declared, and gives the frame range and 'offset' (seconds) each "
        "declared clip occupied on the authoring timeline, since every clip's "
        "own keyframe times are rebased to zero; 'default_clip' is the one a "
        "player opens on, which is not necessarily animations[0] -- an FBX "
        "interchange step can retain a whole-timeline take alongside the split "
        "ones. 'handoff.rendering' records the lighting setup the "
        "reference viewer used to produce the look this asset was approved in: "
        "the asset carries no lights of its own, so a viewer that lights it "
        "differently renders something different without either side being "
        "wrong. Check 'version' against the schema you expect."
    )

    #: Schema version of the FBX-side handoff block (:meth:`build_fbx_handoff`).
    #: Separate from :attr:`SIDECAR_VERSION`: that versions the GLB envelope
    #: this block does not live in, and tying them would force a bump on one
    #: carrier every time the other changed.
    FBX_HANDOFF_VERSION = 1

    #: ``data_export`` channel the FBX handoff block is published on. A channel
    #: like any other, so it rides into the FBX as a user property with no
    #: export-path change and no second carrier -- the deliverable has one
    #: in-band metadata node and this joins it.
    FBX_HANDOFF_CHANNEL = "handoff"

    #: What each known ``data_export`` channel holds, for the block's ``reads``
    #: map. Descriptions only: the channel LIST is taken from the carrier at
    #: stamp time, so a producer added later still appears (described
    #: generically) rather than silently falling out of the contract -- the
    #: block must never claim a channel the file lacks, nor omit one it has.
    FBX_HANDOFF_CHANNELS: Dict[str, str] = {
        "lightmap_metadata": (
            "per-object baked-lightmap records: map file name, uvIndex, "
            "intensity, scaleOffset"
        ),
        "shot_metadata": "shot definitions (name, frame range, notes)",
        "fbx_takes": "the take list realized on this FBX, one per shot",
        "audio_manifest": "audio events with the frames they fire on",
        "shadow_metadata": "shadow-proxy geometry pairing",
        "emissive_groups": "named emissive material groups and their weights",
        "visibility_tracks": (
            "keyed visibility per node, as stepped on/off frames, with the "
            "authored opacity ramp and each take's first/last authored frame"
        ),
    }

    #: The standalone-reader contract for an **FBX** deliverable -- the twin of
    #: :attr:`HANDOFF_INSTRUCTIONS`, kept beside it so the two carriers'
    #: accounts of the same pipeline cannot drift.
    #:
    #: Same rules as the glTF text and for the same reasons: plain declarative
    #: sentences about this file's own structure, no instructions to the reader
    #: about what to do next (which is what makes it safe for an agent to read
    #: as untrusted content), and it names the asset through the block's own
    #: ``asset`` key rather than saying "this file".
    #:
    #: The load-bearing sentence is the one about lightmaps NOT being embedded.
    #: An FBX embeds what its MATERIALS reference, and a lighting-only bake
    #: deliberately does not wire its map into a material -- that is the point,
    #: the PBR material survives the bake -- so the maps the manifest names are
    #: the one part of the deliverable that genuinely does not travel with it.
    #: Measured on a delivered room: 23.0 MB of the 23.5 MB file is embedded
    #: material textures and not one byte of it is the lightmap. Saying so in
    #: the file is what stops that being a surprise the recipient has to be
    #: told about out of band.
    #:
    #: Unlike the glTF text this refers to "the FBX carrying this block" rather
    #: than to an ``asset`` key. The block is stamped by an export PREPARER,
    #: which runs before any FBX path is chosen -- the session hook fires for
    #: File > Export and the Game Exporter too -- so an asset name here could
    #: only ever have been the SCENE's, naming a ``.ma`` as though it were the
    #: deliverable. The authoring scene rides under ``source`` instead, where
    #: it is provenance rather than a false identity.
    FBX_HANDOFF_INSTRUCTIONS = (
        "The FBX carrying this block embeds every texture its MATERIALS "
        "reference, so material assignment resolves with no external files "
        "and no filesystem paths. Tool-authored metadata rides as user "
        "properties on the 'data_export' node; 'reads' names each channel "
        "present on it in this file and what that channel holds, and every "
        "channel value is a JSON string. 'lightmap_metadata' names each baked "
        "object's map by FILE NAME, with 'uvIndex' (0-based, so 1 is the "
        "second UV set), 'intensity' (the multiplier restoring the bake's "
        "original range) and 'scaleOffset' (that object's [scaleX, scaleY, "
        "offsetX, offsetY] rect within a shared atlas). Those maps are NOT "
        "embedded: an FBX carries what its materials reference and a "
        "lighting-only bake leaves the map unwired so the authored material "
        "survives, so the file name is a join token against maps supplied "
        "separately, and the directory it sat in is deliberately not carried. "
        "Each baked object additionally carries its own 'lightmapInfo' user "
        "property repeating that object's record. Geometry, materials and "
        "their embedded textures are otherwise complete. The asset carries no "
        "lights of its own; 'rendering' records the lighting setup the "
        "reference viewer used to produce the look this asset was approved "
        "in, so a consumer that lights it differently renders something "
        "different without either side being wrong. Check 'version' against "
        "the schema you expect."
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

    #: Longest edge a web deliverable's textures keep, and the container they
    #: are re-encoded to. See :meth:`web_delivery_texture_params` for why these
    #: are named constants rather than each producer's own literal.
    WEB_DELIVERY_MAX_SIZE = 2048
    WEB_DELIVERY_FORMAT = "WEBP"

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

    #: Slot semantics a chroma-subsampled codec may be used on. The WebP twin of
    #: :attr:`BASIS_BY_SEMANTIC`'s ETC1S row, and deliberately the same rule --
    #: lossy only where the channels ARE colour and the eye is the judge. An
    #: image nothing samples (semantic ``None``) is off the list for the reason
    #: the ``None`` row above takes UASTC: a mislabel should cost bytes, not
    #: pixels.
    #:
    #: The glTF-structural twin of ``MapRegistry.is_lossy_safe``, which decides
    #: the same question for a map on DISK by its filename type. Neither can
    #: stand in for the other: a GLB image has no filename, only the slots that
    #: sample it, and the registry rule cannot see a slot. The registry's own
    #: measurement stands for both -- a 4K normal at WebP q95 deviates by
    #: 122/255 against 9/255 for a base colour.
    LOSSY_SAFE_SEMANTICS = frozenset({"color"})

    #: WebP save kwargs for everything else. Lossless WebP still comes in well
    #: under the source PNG, so this is a container win rather than a size cost
    #: against the authored map.
    LOSSLESS_WEBP: Dict[str, Any] = {"lossless": True, "quality": 100}

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
        "alpha_mode": "set_glb_alpha_mode",
    }
    #: Root-extras key holding :meth:`apply_scene_sidecar`'s per-section
    #: outcome. Its presence is also the fact "the envelope has been applied
    #: to this file", which a later stage reads to avoid applying it twice.
    SIDECAR_APPLIED_KEY = "scene_sidecar_applied"
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
            for binding in MeshConvert.TEXTURE_CONTAINER_EXTENSIONS:
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
        prompt: Union[bool, Callable[[str], bool]] = True,
    ) -> Optional[str]:
        """Resolve the FBX2glTF executable from PATH or managed installs.

        Parameters:
            required:      Raise FileNotFoundError when missing.
            auto_install:  Download FBX2glTF if not found.
            prompt:        Consent policy for the download -- ``True`` asks on
                           the console (no console = refuse), ``False`` needs
                           none, a callable ``(question) -> bool`` is asked
                           instead (a GUI's dialog). See
                           :meth:`AppInstaller.consent`.

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

        answer = AppInstaller.consent(
            prompt,
            f"FBX2glTF v{FBX2GLTF_VERSION} is not installed. "
            "Download to ~/.pythontk/tools/ now?",
        )
        if answer is None:
            # No interactive console (CI, GUI host, pythonw.exe, etc.).
            # Refuse to silently download — caller must opt-in via prompt=False.
            if required:
                raise FileNotFoundError(
                    "FBX2glTF is not installed and no interactive console "
                    "is available to confirm the download. Pass "
                    "prompt=False to install non-interactively."
                )
            return None
        if not answer:
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
        prompt: Union[bool, Callable[[str], bool]] = True,
        timeout: Optional[float] = AUTO_TIMEOUT,
        extra_args: Optional[List[str]] = None,
        sidecar: Optional[Dict[str, Any]] = None,
        lightmaps: bool = True,
        lightmap_dirs: Sequence[str] = (),
    ) -> str:
        """Convert an FBX file to a binary glTF 2.0 (GLB) file.

        Parameters:
            src:           Input FBX path.
            dst:           Output GLB path. Defaults to src with .glb extension.
                           ``.glb`` is appended if absent.
            overwrite:     Replace existing destination.
            auto_install:  Download FBX2glTF if missing.
            prompt:        Consent policy for that download (see
                           :meth:`resolve_binary`).
            timeout:       Subprocess timeout in seconds. The default derives
                it from the input's size (:meth:`conversion_timeout`), because
                a flat budget that suits a prop discards a production
                assembly's finished deliverable. An explicit number is used
                as given; ``None`` disables the limit.
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
            lightmap_dirs: Extra directories to resolve the manifest's EXR
                           basenames against, forwarded as that method's
                           ``search_dirs``. The manifest carries its own
                           authoring-directory hint, but that is recorded when
                           the bake is COMMITTED and goes stale the moment the
                           project is reorganised or handed to another machine
                           -- at which point the bind silently finds nothing.
                           A host that knows where its textures live now (a
                           DCC's workspace, the scene's own folder) passes them
                           here rather than relying on a historical hint.

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

        if timeout is not None and timeout < 0:
            # The default. An explicit number wins outright (a caller that says
            # 60 means 60) and ``None`` still means no limit.
            timeout = cls.conversion_timeout(src_abs)

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
                # Before every pass that reads or rewrites images, because it
                # RENUMBERS them: the sidecar's texture map and the optimizer's
                # per-image work both describe indices, and both would then
                # describe a file that no longer exists. Also the cheapest
                # point at which to remove work -- an image collapsed here is
                # one the texture pass never decodes.
                try:
                    cls.dedupe_glb_images(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("dedupe_glb_images skipped: %s", exc)
                # Before the sidecar writes the GLB's OWN handoff, so the
                # file never holds both accounts at once.
                try:
                    dropped = cls.strip_fbx_handoff(edit.gltf)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("strip_fbx_handoff skipped: %s", exc)
                else:
                    if dropped:
                        edit.dirty = True
                        logger.debug(
                            "Dropped the FBX handoff block from %d node(s); the "
                            "GLB carries its own.",
                            dropped,
                        )
                if sidecar:
                    # Guarded like every other pass in this chain: the apply
                    # handles its own per-section and container failures, and
                    # anything past those must cost the repairs, never the
                    # conversion -- the preview routes its envelope through
                    # here now, and a push must not fail on a sidecar.
                    try:
                        cls.apply_scene_sidecar(edit, sidecar)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Scene sidecar skipped: %s", exc)
                if lightmaps:
                    # Guarded like the alpha repair: a lightmap failure must
                    # never cost the sidecar or the conversion.
                    try:
                        bound = cls.apply_glb_lightmaps(edit, search_dirs=lightmap_dirs)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("GLB lightmaps skipped: %s", exc)
                    else:
                        if bound:
                            logger.info(
                                "Lightmaps wired into %d material binding(s).",
                                len(bound),
                            )
                # FIRST of the three animation passes: it REPLACES the declared
                # clips, so a gate or a manifest entry written before it would
                # describe clips that no longer exist.
                try:
                    cls.apply_glb_clips(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB clip rebuild skipped: %s", exc)
                # BEFORE the animation manifest, which reports what each clip
                # holds: a shot whose only content is visibility is empty until
                # this has run, and would be reported empty and passed over as
                # the file's default clip.
                try:
                    cls.apply_glb_visibility(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB visibility skipped: %s", exc)
                # After the gate (which makes a fading node present) and before
                # the manifest (which reports a clip carrying only a fade as
                # having content rather than as an empty shot).
                try:
                    faded = cls.apply_glb_fades(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB fades skipped: %s", exc)
                else:
                    if faded:
                        logger.info(
                            "Fades: %d ramp(s) on %d material(s) written as "
                            "%d KHR_animation_pointer channel(s).",
                            faded["nodes"],
                            faded["materials"],
                            faded["channels"],
                        )
                # Unconditional and self-feeding, like the lightmap pass: it
                # reads the take list out of the file and no-ops on a GLB with
                # no animation, so there is no flag for a caller to forget.
                try:
                    animation = cls.apply_glb_animations(edit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GLB animation manifest skipped: %s", exc)
                else:
                    if animation:
                        declared = sum(1 for c in animation["clips"] if c["declared"])
                        logger.info(
                            "Animation: %d clip(s) named in extras.%s (%d from "
                            "declared shots); opens on %r.",
                            len(animation["clips"]),
                            cls.ANIMATION_WEB_KEY,
                            declared,
                            animation["default_clip"],
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
                    f"extras.{cls.ANIMATION_WEB_KEY}": (
                        "the file's animation clips: which are declared shots, "
                        "their authoring frame ranges, and which to open on. "
                        "A clip's STEP 'scale' channels driving a node to zero "
                        "are its VISIBILITY, not authored scale -- glTF has no "
                        "visibility channel, so keyed visibility is carried "
                        "that way and plays with no extension. Each clip's "
                        "'zero_frame' is the authored frame its t=0 sits on "
                        "(NOT 'start_frame': a take is rebased to its first "
                        "key, so the two differ by the lead-in) -- it is what "
                        "converts a playhead to the frame numbers this block "
                        "quotes. Authored alpha fades are KHR_animation_pointer "
                        "channels on each faded subtree's own material "
                        "(extensionsUsed, never required); a runtime without "
                        "the extension can play them by binding each channel "
                        "to that material's alpha, which is what the preview "
                        "page does"
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
    def strip_fbx_handoff(cls, gltf: dict) -> int:
        """Drop the FBX's handoff block from a converted glTF's node extras.

        FBX2glTF transcribes every user property on the ``data_export`` carrier
        into ``nodes[N].extras.fromFBX.userProperties`` (probe-measured), so the
        block :meth:`build_fbx_handoff` stamps for the FBX arrives inside the
        GLB as well -- where it is not provenance but a WRONG self-description:
        it states that the lightmaps the manifest names are not embedded, which
        is true of the FBX and false of a GLB that embeds them, and its
        ``reads`` map names ``data_export.<channel>`` paths that do not exist in
        a glTF. A deliverable carrying two accounts of itself, one of them
        wrong, is worse than one carrying none -- and the GLB has its own, in
        ``extras.scene_sidecar.handoff``.

        Removed even when no sidecar was applied (a bare conversion): "no
        self-description" is a recoverable state, "a confidently wrong one" is
        not. Every other transcribed channel is left alone -- ``lightmap_metadata``
        is this applier's own designed input, and the per-object markers are
        read by consumers outside this repo.

        Returns:
            How many nodes were stripped.
        """
        stripped = 0
        for node in gltf.get("nodes") or []:
            props = ((node.get("extras") or {}).get("fromFBX") or {}).get(
                "userProperties"
            )
            if isinstance(props, dict) and props.pop(cls.FBX_HANDOFF_CHANNEL, None):
                stripped += 1
        return stripped

    @classmethod
    def build_fbx_handoff(
        cls,
        channels: Iterable[str],
        source: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """The standalone-reader contract for an FBX, ready to publish.

        The FBX twin of the ``handoff`` section :meth:`build_scene_sidecar`
        puts in a GLB. Both deliverables of a pair are routinely handed on
        alone -- the GLB to a web consumer, the FBX to an engine -- and until
        now only one of them could be read without a covering note.

        Deliberately NOT the scene-sidecar envelope. That envelope carries
        repairs for what the FBX *format* mistranslates, which is meaningless
        inside the FBX itself; and this package's standing split (see
        :meth:`build_scene_sidecar`'s scope note) is that FBX-side metadata
        rides as ``data_export`` channels while the sidecar is the GLB's. This
        is one more channel on the carrier that already exists, not a second
        carrier and not a companion file.

        Parameters:
            channels: The channel names actually present on the carrier --
                normally ``DataNodes.dump()["data_export"]``. Taken from the
                caller rather than assumed so the block describes THIS file:
                a channel the scene never wrote must not be claimed, and one
                a newer producer wrote must not be omitted. The handoff
                channel itself is dropped if passed (it does not describe
                itself).
            source: Producer identity and provenance, e.g. ``{"application":
                "maya", "version": "2025", "scene": "PROD_ROOM.ma"}``. No
                asset name is carried: this is stamped before any FBX path is
                chosen, so the only name available would be the scene's.

        Returns:
            The block, JSON-serializable. Empty dict when *channels* holds
            nothing to describe -- a carrier with no metadata has no handoff
            to make, and stamping one would put a lone self-referential
            channel in an otherwise empty node.
        """
        present = [c for c in channels if c != cls.FBX_HANDOFF_CHANNEL]
        if not present:
            return {}
        return {
            "version": cls.FBX_HANDOFF_VERSION,
            # Empty entries dropped rather than published as nulls: an unsaved
            # scene has no name, and "scene": null in a delivered artifact reads
            # as a field that failed rather than one that does not apply.
            "source": {k: v for k, v in (source or {}).items() if v} or None,
            "instructions": cls.FBX_HANDOFF_INSTRUCTIONS,
            "reads": {
                f"data_export.{name}": cls.FBX_HANDOFF_CHANNELS.get(
                    name, "tool-authored channel"
                )
                for name in sorted(present)
            },
            # The same policy the GLB publishes, from the same constant: a
            # recipient who lights a baked asset normally blows out every baked
            # surface, and that reads as a bake regression whichever container
            # it arrived in.
            "rendering": copy.deepcopy(cls.RENDERING_POLICY),
        }

    @staticmethod
    def _sidecar_section_scope(present: Set[str], data: Any) -> Optional[int]:
        """How many of *data*'s entries name a material *present* in the GLB.

        ``None`` when the section is not a map keyed by material name -- the
        registry is an extension point, so a future section may be keyed on
        anything, and reporting every entry of one as "not in this export"
        would be worse than not scoping it at all. The test is an intersection
        rather than a type check: a material-keyed section on a GLB that
        carries none of its names is indistinguishable from a section keyed on
        something else, and both want the same answer -- do not scope this.
        """
        if not isinstance(data, dict):
            return None
        overlap = len(present.intersection(map(str, data)))
        return overlap or None

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
            ``{section: outcome}`` — ``"N of M"`` (with a
            ``" (K not in this export)"`` clause when the envelope names
            materials this GLB does not carry), ``"0 of M matched"`` or
            ``"failed (...)"`` per offered section; empty when none offered.
        """
        if not sidecar:
            return {}
        sections = sidecar.get("sections") or {}
        summary: Dict[str, str] = {}
        try:
            with cls.open_glb(glb) as edit:
                present = {
                    str(m.get("name"))
                    for m in (edit.gltf.get("materials") or [])
                    if m.get("name")
                }
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
                    # The envelope describes the SCENE and the GLB carries the
                    # exported subset, so entries naming a material that is not
                    # in this file were never in scope. Counting them made a
                    # correct deliverable read "10 of 23" on a production
                    # assembly -- 13 reference/ID materials that never
                    # exported, reported as if 13 repairs had failed. The same
                    # distinction ``apply_glb_lightmaps`` draws with
                    # ``out_of_scope``; ``None`` means the section is not keyed
                    # by material name and the plain count is all there is.
                    in_scope = cls._sidecar_section_scope(present, data)
                    # Counted by NAME, not by record: a by-name writer lands
                    # on every material carrying the name (fade clones, the
                    # converter's own duplicates), and each landing is a
                    # record. The outcome the panel reads is "how many of the
                    # scene's materials were repaired", which is names. A
                    # record without a name -- or not a record at all, from a
                    # writer that reports differently -- counts as one.
                    landed = len(
                        {
                            record.get("material", index)
                            if isinstance(record, dict)
                            else index
                            for index, record in enumerate(applied)
                        }
                    )
                    if in_scope is not None and landed > in_scope:
                        # More landed than the exact-name scope allows, so this
                        # applier does not match on the name alone (a
                        # namespace-tolerant or fuzzy resolver would do this).
                        # The scope model does not describe it, and "12 of 10"
                        # is worse than the plain count -- fall back rather
                        # than print a number that cannot be true.
                        in_scope = None
                    out_of_scope = len(data) - in_scope if in_scope is not None else 0
                    scope = (
                        f" ({out_of_scope} not in this export)" if out_of_scope else ""
                    )
                    if not applied and (in_scope is None or in_scope):
                        # The section was read but nothing landed — almost
                        # always a name mismatch, which the applier has just
                        # logged in full. Only reported as a miss when
                        # something WAS in scope to match.
                        logger.warning(
                            "Sidecar %r matched none of its %s entr(ies) in the GLB.",
                            section,
                            len(data),
                        )
                        summary[section] = f"0 of {len(data)} matched"
                        continue
                    denominator = len(data) if in_scope is None else in_scope
                    logger.info(
                        "Sidecar %r applied to %s of %s in this export%s.",
                        section,
                        landed,
                        denominator,
                        f"; {out_of_scope} scene material(s) not in it"
                        if scope
                        else "",
                    )
                    summary[section] = f"{landed} of {denominator}{scope}"
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
                extras[cls.SIDECAR_APPLIED_KEY] = dict(summary)
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
            ``animation`` (clip counts, rate and default clip, when the file
            carries that block), ``extensions`` (``required`` / ``used`` --
            what a reader must SUPPORT to open the file at all), ``lightmap``
            and ``generator``.
            ``problems`` and ``ok`` are kept
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
            # What a reader must SUPPORT to open this at all. `extensionsRequired`
            # is not advice: the spec says a reader that does not implement one
            # of these must refuse the file. A web-delivery GLB requires
            # `EXT_texture_webp` (nothing core-readable survives that pass) and
            # a pure-delivery KTX2 one requires `KHR_texture_basisu`, so the
            # deliverable most likely to reach a third party is exactly the one
            # with a hard prerequisite -- and this report, which exists to tell
            # a recipient what they are holding, did not mention it. Reported
            # rather than failed: the requirement is correct, it just has to be
            # readable without parsing the JSON chunk by hand.
            required = sorted(edit.gltf.get("extensionsRequired") or [])
            report["extensions"] = {
                "required": required,
                "used": sorted(edit.gltf.get("extensionsUsed") or []),
            }
            if required:
                report["notes"].append(
                    "requires a reader supporting "
                    + ", ".join(required)
                    + " -- a viewer without it must refuse this file"
                )
            # The spec's own subset rule, and a real failure rather than a note:
            # a file demanding a capability it never declares is invalid glTF
            # and stock validators reject it outright.
            undeclared = sorted(set(required) - set(report["extensions"]["used"]))
            if undeclared:
                fail(
                    f"extensionsRequired names {', '.join(undeclared)}, which "
                    "extensionsUsed does not declare -- invalid glTF"
                )
            # Reported before the envelope gate below: an asset can carry
            # clips and no sidecar (another producer's file, or one converted
            # before the envelope existed), and "how many clips, how many of
            # them shots" is exactly what a recipient asks first of an animated
            # deliverable.
            animation = extras.get(cls.ANIMATION_WEB_KEY)
            if isinstance(animation, dict):
                clips = animation.get("clips") or []
                report["animation"] = {
                    "clips": len(clips),
                    "declared": sum(1 for c in clips if c.get("declared")),
                    "default_clip": animation.get("default_clip"),
                    "fps": animation.get("fps"),
                }
            elif edit.gltf.get("animations"):
                # Not a failure -- the block is only written by our own
                # conversion -- but a note, because a consumer that went
                # looking for shot names here found none.
                report["notes"].append(
                    f"{len(edit.gltf['animations'])} animation(s) with no "
                    f"extras.{cls.ANIMATION_WEB_KEY}: clip identity is names only"
                )

            hollow = [
                str(a.get("name") or f"animations[{i}]")
                for i, a in enumerate(edit.gltf.get("animations") or [])
                if not (a.get("channels") and a.get("samplers"))
            ]
            if hollow:
                # glTF requires BOTH arrays to carry at least one entry, so this
                # is not a quality note -- the file fails validation. Measured
                # on a production deliverable: a take split emits a named
                # AnimStack for a shot whose content is entirely visibility,
                # and the converter writes it out with neither channel nor
                # sampler. A strict reader is entitled to reject the whole file.
                fail(
                    f"{len(hollow)} animation(s) carry no channels or samplers "
                    f"({', '.join(hollow)}) -- glTF requires at least one of "
                    "each, so this file does not validate"
                )

            # The file's own handoff promising clips the file does not carry.
            # Measured on a production deliverable (2026-08-30):
            # the export wrote `data_export.fbx_takes` naming 12 shots while the
            # FBX itself was written with animation disarmed, so the GLB shipped
            # 12 declared clips and ZERO animations -- and every check here
            # passed, because textures, sections and envelope were all sound.
            # A recipient reading the handoff plans for clips that do not exist,
            # which makes this a defect of the artifact, not a note.
            declared_takes = cls.data_export_channel(edit.gltf, cls.FBX_TAKES_KEY)
            if (
                isinstance(declared_takes, list)
                and declared_takes
                and not (edit.gltf.get("animations") or [])
            ):
                fail(
                    f"the handoff declares {len(declared_takes)} take(s) "
                    f"(data_export.{cls.FBX_TAKES_KEY}) but the file carries no "
                    "animations -- the FBX was written with animation off "
                    "(bake/takes disarmed at export)"
                )
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
            applied = extras.get(cls.SIDECAR_APPLIED_KEY)
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
                    #
                    # Grouped BY IMAGE, because the finding is a property of the
                    # image and the lightmap pass clones a material per
                    # instance: on a production room one unvalidated ORM
                    # produced 46 identical lines naming 46 clones of one
                    # material, which is the length that stops a note being
                    # read at all. The per-material detail stays in
                    # ``report["orm"]`` for anything that wants it.
                    by_image: Dict[str, List[str]] = {}
                    for material in unvalidated:
                        image = str((suspect[material] or {}).get("image") or "?")
                        by_image.setdefault(image, []).append(material)
                    described = []
                    for image, names in sorted(by_image.items()):
                        shown = ", ".join(sorted(names)[:3])
                        more = ", ..." if len(names) > 3 else ""
                        described.append(
                            f"{image} ({len(names)} material(s): {shown}{more})"
                        )
                    report["notes"].append(
                        "ORM binding not described by the envelope on "
                        + "; ".join(described)
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
    #: EVERY build-time locate hint, so a scrub cannot know about one and
    #: miss another. ``dirs`` (plural) joined it when a single folder proved
    #: too weak a hint -- a scene with maps in two places published nothing
    #: and the consumer then found a map by basename alone, binding a stale
    #: atlas. Both are absolute authoring paths and both must come out.
    LOCATE_HINT_KEYS = ("dir", "dirs")
    #: Joins a lightmap clone's name to the material it was cloned from. The
    #: lightmap pass makes one material per INSTANCE (each needs its own atlas
    #: rect), so a base material becomes ``<base>~lm<N>``. Named because the
    #: convention has two ends: whoever writes the clone and whoever has to
    #: recognise one -- an envelope section names the BASE materials, and a
    #: check matching clone names against it literally finds nothing and
    #: reports every clone as undescribed (measured: 46 spurious notes on one
    #: production room, from a single base material's clones).
    LIGHTMAP_CLONE_SUFFIX = "~lm"

    @classmethod
    def _lightmap_clone_base(cls, name: str) -> str:
        """The material *name* was cloned from, or *name* when it is not a clone.

        Only a ``~lm`` followed by digits is a clone marker: the suffix is legal
        in an authored material name, and stripping at a bare ``~lm`` would
        rewrite one.
        """
        text = str(name)
        base, sep, tail = text.rpartition(cls.LIGHTMAP_CLONE_SUFFIX)
        return base if sep and base and tail.isdigit() else text

    @classmethod
    def data_export_channel(cls, gltf: dict, key: str) -> Optional[Any]:
        """Decoded value of one ``data_export`` channel in a parsed glTF, or ``None``.

        Every in-band metadata system publishes onto the same carrier -- the
        lightmap manifest, the shot definitions, the take list, the audio events
        -- so reading one is the same puzzle every time, and the puzzle is the
        two on-disk SHAPES, not the channel. Hand-rolled per channel, the second
        copy is the one that probes only the nested shape and turns its whole
        feature into a silent no-op on a natively exported GLB.

        Split from :meth:`read_glb_lightmap_manifest` so an applier can read from
        the session it is ALREADY holding -- a second ``open_glb`` on the path
        would re-read and re-parse the file, which is exactly what
        :class:`GlbEdit` exists to avoid.

        Returns whatever the channel decodes to (the producers publish JSON, so
        in practice a dict or a list); a caller wanting one shape checks. An
        unparsable channel is warned and read as absent -- a half-decoded
        manifest is worse than none.
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
                entry = props.get(key)
                if entry is None:
                    continue
                # FBX2glTF wraps each property as {"type": ..., "value": ...}.
                raw = entry.get("value") if isinstance(entry, dict) else entry
                try:
                    return json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, ValueError) as error:
                    logger.warning(
                        "Unparsable %s on node %r: %s",
                        key,
                        node.get("name"),
                        error,
                    )
                    return None
        return None

    @classmethod
    def _lightmap_manifest(cls, gltf: dict) -> Optional[Dict[str, Any]]:
        """The ``lightmap_metadata`` manifest in a parsed glTF, or ``None``.

        The channel read is :meth:`data_export_channel`; this adds only the
        applier's own expectation that the manifest is an OBJECT, so a channel
        holding anything else reads as absent rather than reaching the walk as
        something without ``.get``.
        """
        data = cls.data_export_channel(gltf, cls.LIGHTMAP_METADATA_KEY)
        return data if isinstance(data, dict) else None

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
        if not isinstance(section, dict) or not any(
            k in section for k in cls.LOCATE_HINT_KEYS
        ):
            # Nothing to strip: hand the snapshot straight back rather than
            # churning a copy of a payload that is already clean (pinned by
            # test_without_locate_hints_passes_clean_snapshots_through). Note
            # this is the one path whose result is the caller's own object --
            # safe because the contract is only that this never MUTATES the
            # snapshot, and treating the return as read-only satisfies both.
            return data_export
        scrubbed = dict(data_export)
        scrubbed[cls.LIGHTMAP_METADATA_KEY] = {
            k: v for k, v in section.items() if k not in cls.LOCATE_HINT_KEYS
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
                    had_hint = any(k in data for k in cls.LOCATE_HINT_KEYS)
                    if not had_hint and not corrected:
                        continue
                    for hint_key in cls.LOCATE_HINT_KEYS:
                        data.pop(hint_key, None)
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
    def _lightmap_node_index(cls, gltf: dict):
        """Index a glTF's MESH nodes for manifest lookup.

        Returns ``(nodes_by_name, leaf_index, mesh_users)``: nodes grouped by
        their exact name, the namespace-stripped leaf of each of those names
        mapped back to the full names carrying it, and how many nodes reference
        each mesh (which is what identifies an instanced mesh needing its own
        clone before a per-instance rect can bind to it).
        """
        nodes_by_name: Dict[str, List[dict]] = {}
        mesh_users: Dict[int, int] = {}  # mesh index -> node reference count
        for node in gltf.get("nodes", []) or []:
            if "mesh" in node:
                nodes_by_name.setdefault(node.get("name", ""), []).append(node)
                mesh_users[node["mesh"]] = mesh_users.get(node["mesh"], 0) + 1
        leaf_index: Dict[str, List[str]] = {}
        for full in nodes_by_name:
            leaf_index.setdefault(full.rsplit(":", 1)[-1], []).append(full)
        return nodes_by_name, leaf_index, mesh_users

    @classmethod
    def _resolve_lightmap_node(
        cls,
        name: Optional[str],
        nodes_by_name: Dict[str, List[dict]],
        leaf_index: Dict[str, List[str]],
    ) -> Tuple[Optional[List[dict]], str, List[str]]:
        """Resolve one manifest object name to the GLB's mesh nodes.

        The single owner of the match rule, because two readers of it is how
        the binder and the report came to disagree about what "missing" means.
        Exact first, then namespace-tolerant: manifests and exports can
        disagree about namespaces without either being wrong (an older
        publisher stripped them, some exporters flatten them), but a leaf
        matching several nodes is never guessed at -- that would put one
        object's lighting on another.

        Returns ``(nodes, status, leaves)`` where *status* is one of
        ``"exact"``, ``"leaf"``, ``"ambiguous"`` (in the file, unbindable) or
        ``"absent"`` (no such node -- on a selection-scoped export, simply not
        part of it). *leaves* carries the candidates behind an ambiguous match.
        """
        nodes = nodes_by_name.get(name or "")
        if nodes:
            return nodes, "exact", []
        if name:
            leaves = leaf_index.get(name.rsplit(":", 1)[-1]) or []
            if len(leaves) == 1:
                return nodes_by_name[leaves[0]], "leaf", leaves
            if len(leaves) > 1:
                return None, "ambiguous", leaves
        return None, "absent", []

    @classmethod
    def lightmap_manifest_coverage(cls, glb: GlbTarget) -> Dict[str, List[str]]:
        """Split a GLB's bake manifest by whether this GLB actually carries each object.

        The manifest is a **scene** record -- the bake commits to the scene, and
        every export of any subset of it carries the whole thing -- so the
        manifest's length is not the number of objects a given deliverable was
        supposed to light. Reading it as one makes a selection export report
        every unselected object as unlit, which is a false alarm raised
        precisely when the deliverable is correct.

        Returns ``{"present", "ambiguous", "absent"}`` name lists, where
        *present* is in scope and bindable, *ambiguous* is in scope but matched
        several nodes by leaf name (a real failure), and *absent* has no node in
        this GLB at all (out of scope -- or, if it was meant to be exported, a
        name mismatch this cannot tell apart from a scope boundary).
        """
        with cls.open_glb(glb) as edit:
            manifest = cls._lightmap_manifest(edit.gltf) or {}
            nodes_by_name, leaf_index, _ = cls._lightmap_node_index(edit.gltf)
            buckets: Dict[str, List[str]] = {
                "present": [],
                "ambiguous": [],
                "absent": [],
            }
            for entry in manifest.get("objects") or []:
                name = entry.get("name")
                if not name:
                    continue
                _, status, _ = cls._resolve_lightmap_node(
                    name, nodes_by_name, leaf_index
                )
                bucket = "present" if status in ("exact", "leaf") else status
                buckets[bucket].append(str(name))
            return buckets

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

            # The manifest's OWN folders first, then the caller's search paths.
            # ``dirs`` (plural) is the publisher's full set: a scene whose maps
            # do not all live in one folder -- the moment ANY object keeps a
            # marker from an earlier bake -- has no single ``dir`` to state, and
            # publishing nothing there dropped the hint for every object that
            # did agree. What followed was silent and severe: the basename
            # search reached the workspace's texture folder and bound a
            # 17-day-old atlas of the same name, so 46 objects sampled a stale
            # map through rects computed for the current bake.
            # ``isinstance`` rather than a bare truthiness test: this reads a
            # manifest out of whatever GLB the caller points at, and a ``dirs``
            # that arrived as a STRING would splat into one bogus directory per
            # character -- a silent, unbounded stat storm rather than an error.
            plural = manifest.get("dirs")
            authored = [
                d
                for d in [
                    manifest.get("dir"),
                    *(plural if isinstance(plural, (list, tuple)) else []),
                ]
                if isinstance(d, str) and d
            ]
            dirs = [d for d in [*authored, *search_dirs] if d]
            dirs.append(os.path.dirname(os.path.abspath(edit.path)))
            authored_norm = {os.path.normcase(os.path.abspath(d)) for d in authored}

            nodes_by_name, leaf_index, mesh_users = cls._lightmap_node_index(gltf)
            #: Manifest entries with no node in this GLB. The manifest is a
            #: SCENE record and every export carries all of it, so on a
            #: selection export these are simply the objects that were not
            #: selected -- counted for one summary line, never warned per
            #: object (which turned a correct 3-object push of a 50-object
            #: scene into 47 warnings saying it shipped unlit).
            out_of_scope: List[str] = []

            # exr abspath -> encode scalar; ``None`` records a failed encode so a map
            # shared by several objects is not retried (and re-logged) per object.
            scalars: Dict[str, Optional[float]] = {}
            #: Unfindable map -> how many manifest entries wanted it. Counted
            #: rather than warned inline for two reasons: an atlas is shared by
            #: every object baked into it, so a line per ENTRY turned one moved
            #: folder into 48 identical ones (measured on a delivered room --
            #: which is how the failure got lost in an export log and shipped a
            #: lightmap-less deliverable); and the number of objects at stake,
            #: the one thing that says how bad it is, is not known until the
            #: walk is over.
            missing: Dict[Optional[str], int] = {}
            #: map basename -> the path it resolved to OUTSIDE the manifest's
            #: own folders (found by basename alone; see the bind site).
            fallback_binds: Dict[Optional[str], str] = {}
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
                nodes, status, leaves = cls._resolve_lightmap_node(
                    name, nodes_by_name, leaf_index
                )
                if status == "ambiguous":
                    # In the file, and genuinely unbindable: still loud.
                    logger.warning(
                        "Lightmap for %r: leaf name matches several GLB "
                        "nodes (%s) -- ambiguous, not bound.",
                        name,
                        ", ".join(sorted(leaves)),
                    )
                    continue
                if status == "absent":
                    # Not in this export. Kept at debug because it is also what
                    # a genuine name mismatch looks like, and the node list is
                    # what tells the two apart.
                    out_of_scope.append(str(name))
                    logger.debug(
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
                    missing[basename] = missing.get(basename, 0) + 1
                    continue
                src = os.path.abspath(src)
                # A map found OUTSIDE every folder the manifest names was found
                # by basename alone, and a basename is not an identity: the
                # workspace's texture folder routinely holds an atlas from an
                # earlier bake under the same name (and, on a case-insensitive
                # filesystem, under a differently-cased one). Binding it pairs
                # THIS bake's rects with THAT bake's pixels -- every object
                # sampling someone else's lighting, with nothing in the log to
                # say so. Legitimate whenever the maps have simply moved, so it
                # is a warning and not a refusal; named once per map.
                if (
                    authored_norm
                    and os.path.normcase(os.path.dirname(src)) not in authored_norm
                ):
                    fallback_binds.setdefault(basename, src)

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
                            clone["name"] = (
                                f"{base_name}{cls.LIGHTMAP_CLONE_SUFFIX}"
                                f"{len(gltf['materials'])}"
                            )
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

            for basename, resolved in fallback_binds.items():
                logger.warning(
                    "Lightmap %r was not in the folder(s) the manifest names "
                    "(%s); bound %s, found by name alone. If that is an atlas "
                    "from an earlier bake, its rects do not match this one and "
                    "every object on it samples the wrong patch.",
                    basename,
                    ", ".join(authored) or "<none published>",
                    resolved,
                )

            for basename, wanted in missing.items():
                # Deliberately does NOT advise passing ``search_dirs``: every
                # shipped caller already does (the DCC exporters and the preview
                # hand over the host's live texture folders), so by the time
                # this fires the directories listed ARE the full search and the
                # map is in none of them. Naming them, and what it costs, is the
                # whole actionable content.
                logger.warning(
                    "Lightmap %r not found in %s -- the %d object(s) baked into "
                    "it are not bound and will render unlit.",
                    basename,
                    dirs,
                    wanted,
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
            elif len(entries) > len(out_of_scope):
                # The one outcome silence gets wrong. Every miss above is warned
                # individually, but each reads as a per-object detail; a caller
                # -- and an exporter's log -- needs the TOTAL said once, because
                # "the scene was never baked" (a clean no-op, returned far
                # above) and "the bake exists and none of it reached the file"
                # are the same empty list and wildly different deliverables.
                # Counted over what this GLB CARRIES: a selection export holds a
                # subset of a scene-wide manifest, so measuring against the
                # whole of it called a correct push unlit.
                logger.warning(
                    "Lightmaps NOT wired: %d object(s) in this GLB are baked "
                    "and none bound -- it ships unlit. Searched %s.",
                    len(entries) - len(out_of_scope),
                    dirs,
                )
            if out_of_scope:
                logger.info(
                    "Lightmap manifest covers %d object(s) not in this export "
                    "(scene-wide manifest, exported subset); ignored.",
                    len(out_of_scope),
                )
        return records

    # ------------------------------------------------------------------ #
    # Animation: say which clip is which in a shot-split deliverable
    # ------------------------------------------------------------------ #

    #: ``data_export`` channels the shot system publishes (mayatk/blendertk
    #: ``ShotStore.publish_export_view``): the take list the FBX exporter splits
    #: its AnimStacks by, and the per-shot extras a take NAME cannot carry.
    FBX_TAKES_KEY = "fbx_takes"
    SHOT_METADATA_KEY = "shot_metadata"
    #: Root-extras key the web viewer reads to choose and place clips
    #: (``preview_viewer.html``) -- the animation twin of
    #: :attr:`LIGHTMAP_WEB_KEY`, and written by the same kind of applier:
    #: derived from the in-band channels at conversion time, so the decoded,
    #: index-bound view exists in one place instead of in every consumer.
    ANIMATION_WEB_KEY = "animation_web"
    #: Schema version of that block.
    ANIMATION_WEB_VERSION = 1
    #: ``data_export`` channel carrying the animated visibility that glTF has no
    #: channel for (mayatk/blendertk ``RenderOpacity.refresh_export_metadata``).
    #: See :meth:`apply_glb_visibility` for why it cannot ride the FBX.
    VISIBILITY_TRACKS_KEY = "visibility_tracks"
    #: Highest ``visibility_tracks`` schema this applier knows how to read.
    VISIBILITY_TRACKS_VERSION = 1

    @staticmethod
    def _animation_span(gltf: dict, animation: dict) -> Optional[Tuple[float, float]]:
        """``(first, last)`` keyframe time of *animation*, in seconds, or ``None``.

        Read off the sampler INPUT accessors' ``min``/``max``, which glTF
        requires an animation sampler input to carry -- so this costs no buffer
        decode, and a file that omits them (not a legal one, but this is public
        API pointed at whatever it is handed) reports no span rather than a
        wrong one.
        """
        accessors = gltf.get("accessors") or []
        lo: Optional[float] = None
        hi: Optional[float] = None
        for sampler in animation.get("samplers") or []:
            index = sampler.get("input")
            if not isinstance(index, int) or not 0 <= index < len(accessors):
                continue
            accessor = accessors[index] or {}
            low, high = accessor.get("min"), accessor.get("max")
            # Shape-checked, not just truthiness: this is public API pointed at
            # whatever GLB it is handed, and an accessor whose min is a bare
            # number rather than the spec's array would raise out of a pass
            # whose whole contract is that it degrades to "no span".
            if not (isinstance(low, list) and isinstance(high, list)):
                continue
            if not low or not high:
                continue
            if not isinstance(low[0], (int, float)) or not isinstance(
                high[0], (int, float)
            ):
                continue
            lo = low[0] if lo is None else min(lo, low[0])
            hi = high[0] if hi is None else max(hi, high[0])
        return None if lo is None or hi is None else (float(lo), float(hi))

    @staticmethod
    def _numeric_pairs(keys: Any) -> List[Sequence[Any]]:
        """The ``[frame, value]`` entries of *keys* that are actually numbers.

        The tracks arrive as JSON decoded out of a string channel on a node in
        the deliverable, so their shape is whatever a producer wrote -- and a
        non-numeric entry would otherwise raise out of a pass whose whole
        contract is that a malformed channel degrades to "no tracks". Same rule
        :meth:`_animation_span` applies to accessor bounds, for the same reason.
        """
        out: List[Sequence[Any]] = []
        for key in keys if isinstance(keys, (list, tuple)) else ():
            if not isinstance(key, (list, tuple)) or len(key) < 2:
                continue
            # bool passes the int check, which is correct: a JSON ``true`` is a
            # legitimate way to write a visibility value.
            if isinstance(key[0], (int, float)) and isinstance(key[1], (int, float)):
                out.append(key)
        return out

    @classmethod
    def _presence_keys(cls, track: Dict[str, Any]) -> List[Sequence[float]]:
        """The on/off timeline to GATE this track on -- the fade's, when it has one.

        A DCC that keys opacity as a custom attribute (mayatk's ``RenderOpacity``
        in its recommended "attribute" mode is one) drives nothing with it: the
        attribute is metadata for the engine downstream, and the viewport shows
        only the boolean ``visibility`` mirrored from it. That mirror is
        ``opacity > 0``, evaluated at the KEYS -- so a fade-in keyed 0 at frame
        8 and 1 at frame 23 mirrors to visibility 0 at 8 and 1 at 23, and a
        stepped boolean holds 0 across the whole ramp. Gate on that and the
        object is ABSENT for the entire fade-in and fully opaque for the entire
        fade-out: the fade cannot be seen, in Maya or in anything reading the
        mirror, which is why an authored fade has always arrived as a pop.

        So a track carrying a real ramp is gated on the ramp instead: present
        wherever the alpha is, or is about to become, non-zero. The object is
        then there to BE faded, and the ``KHR_animation_pointer`` channels
        :meth:`apply_glb_fades` writes supply the alpha.

        A track whose "ramp" only ever holds 0 or 1 is not a fade and keeps the
        mirrored boolean exactly as before -- same predicate as
        :meth:`_authored_fades`, so the two cannot disagree about what a fade is.
        """
        keys = cls._numeric_pairs(track.get("opacity") or [])
        pairs = [(float(k[0]), float(k[1])) for k in keys]
        if len(pairs) < 2 or not cls._is_fade(pairs):
            return track.get("visibility") or []
        runs: List[Sequence[float]] = []
        for (t0, v0), (_t1, v1) in zip(pairs, pairs[1:]):
            # A segment is PRESENT when either end is non-zero: that covers the
            # ramp out of zero (which is the frame the object has to appear on)
            # as well as every fully-visible stretch.
            runs.append([t0, 1.0 if (v0 > 0.0 or v1 > 0.0) else 0.0])
        runs.append([pairs[-1][0], 1.0 if pairs[-1][1] > 0.0 else 0.0])
        if runs[0][1] and pairs[0][1] <= 0.0:
            # BEFORE the ramp begins the object is not there, and a stepped
            # track is read by holding its first key BACKWARDS -- so a rising
            # ramp whose first key says "present" would make the object present
            # for the whole file up to it. Measured: seven nodes turned up in
            # every shot before the one they fade into. The extra key sits on
            # the same frame and loses to it (ties resolve to the LAST value),
            # so it changes nothing from the ramp onward and everything before.
            runs.insert(0, [pairs[0][0], 0.0])
        return runs

    @staticmethod
    def _is_fade(pairs: Sequence[Sequence[float]]) -> bool:
        """Does this ramp actually fade, or is it a boolean in float clothing?

        Either an intermediate alpha exists, or two keys differ in BOTH time
        and value -- an endpoint-only ramp still fades when its endpoints
        straddle time. What makes one a STEP is both keys landing on the same
        frame.
        """
        if any(0.0 < value < 1.0 for _t, value in pairs):
            return True
        return any(a[0] != b[0] and a[1] != b[1] for a, b in zip(pairs, pairs[1:]))

    @classmethod
    def _visibility_runs(
        cls, keys: Sequence[Sequence[float]], window: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """Sample a stepped on/off track across *window*, as ``(frame, value)``.

        Three rules, and the last is the one that matters most:

        * The value at the window's first frame is the one HELD there -- the
          last key at or before it, or the first key's value when the window
          opens before the track does (a DCC holds the first key backwards).
        * Keys strictly inside the window follow, in order.
        * Consecutive equal values collapse, so a track that never switches
          inside a clip costs one key rather than one per key authored.

        The held-value rule is what stops an object switched off in shot 5 from
        reappearing in shot 7: shot 7's window contains no key at all, and the
        naive reading of "no keys here" is "nothing to write", which leaves the
        node at its full authored scale for the whole clip.
        """
        ordered = sorted((float(k[0]), float(k[1])) for k in cls._numeric_pairs(keys))
        if not ordered:
            return []
        start, end = window
        held = ordered[0][1]
        for time, value in ordered:
            if time <= start:
                held = value
            else:
                break
        samples = [(start, held)]
        samples.extend((t, v) for t, v in ordered if start < t <= end)
        runs: List[Tuple[float, float]] = []
        for time, value in samples:
            if not runs or runs[-1][1] != value:
                runs.append((time, value))
        return runs

    @staticmethod
    def _strictly_increasing(
        samples: Sequence[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Collapse *samples* onto strictly increasing times, then onto runs.

        glTF requires an animation sampler's input times to strictly increase,
        and clamping negative times to zero (a key authored before the clip's
        own zero) is exactly what produces a tie. The LAST value at a repeated
        time wins, because it is the later state; the run-collapse then drops
        any key that changes nothing.
        """
        merged: List[Tuple[float, float]] = []
        for time, value in samples:
            if merged and merged[-1][0] >= time:
                merged[-1] = (merged[-1][0], value)
            else:
                merged.append((time, value))
        runs: List[Tuple[float, float]] = []
        for time, value in merged:
            if not runs or runs[-1][1] != value:
                runs.append((time, value))
        return runs

    @classmethod
    def apply_glb_clips(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """Rebuild the declared shot clips as exact slices of the whole timeline.

        Maya's take split is lossy in a way that does not announce itself: it
        restricts each curve to the take's window and bakes what is left, so a
        curve with no key inside a shot contributes NO channel to it and the
        node plays its rest pose for the shot's whole duration. Measured on a
        12-shot production assembly (358 keys over 2635 frames), Shot_1 through
        Shot_11 were wrong on EVERY frame -- by up to 3.73 m -- while the
        whole-timeline stack the same export retains was right on all 2629.

        So the shots are cut here, from that stack, on the deliverable: no scene
        to reach into, nothing mutated, and the exporter and the preview push
        get identical clips because they run the same pass.
        :mod:`~pythontk.file_utils.mesh_convert.glb_clips` does the cutting;
        this half reads the file (which takes, which rate, which origin).

        Runs FIRST, before :meth:`apply_glb_visibility` writes the presence
        gates -- the gates belong to the rebuilt clips, and a gate written into
        a clip this pass then replaces would be discarded with it.

        Needs to know the authoring frame the source stack places at its own
        ``t=0``, because the converter rebases every stack onto its first key.
        Three ways, in order: the stack's own ``extras`` (there after this pass
        has run once, which is what makes a second run exact rather than
        merely harmless); the producer's published ``clip_span`` for the whole
        timeline; and otherwise nothing -- a guessed origin would slide every
        shot by the same wrong amount, which is worse than the split this
        replaces, so the clips are left as exported.

        Returns:
            ``{"clips", "channels", "source", "bytes"}``, or ``None`` when
            there was nothing to rebuild.
        """
        from pythontk.file_utils.mesh_convert.glb_clips import GlbClips

        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            takes = cls.data_export_channel(gltf, cls.FBX_TAKES_KEY)
            if not isinstance(takes, list) or not takes:
                return None
            metadata = cls.data_export_channel(gltf, cls.SHOT_METADATA_KEY)
            channel = cls.data_export_channel(gltf, cls.VISIBILITY_TRACKS_KEY)
            fps = cls._resolve_clip_fps(
                metadata if isinstance(metadata, dict) else {}, []
            )
            if not fps and isinstance(channel, dict):
                try:
                    fps = float(channel.get("fps") or 0.0)
                except (TypeError, ValueError):
                    fps = 0.0
            if not fps:
                logger.warning(
                    "Clips: the declared takes are quoted in frames and the "
                    "file carries no frame rate, so they cannot be placed in "
                    "time -- clips left as exported."
                )
                return None

            windows = cls._take_windows(gltf)
            animations = gltf.get("animations") or []
            if not animations:
                # Loud, because the combination is wrong in BOTH deliverables at
                # once: the handoff names shots a consumer will plan for, and
                # the file cannot play any of them. Measured on a production
                # assembly: the FBX was written with bake/takes disarmed, so
                # the conversion arrived animationless and this pass -- the one
                # place that knows both halves -- said nothing.
                logger.warning(
                    "Clips: the scene declares %d take(s) but the file carries "
                    "no animations at all -- the FBX was written with animation "
                    "off (bake/takes disarmed at export), so there is nothing "
                    "to cut the shots from.",
                    len(takes),
                )
                return None
            index = GlbClips._source_stack(animations, list(windows))
            if index is None:
                return None
            source = animations[index]

            zero = (source.get("extras") or {}).get(GlbClips.ZERO_FRAME_KEY)
            if not isinstance(zero, (int, float)):
                spans = (channel or {}).get("clip_span")
                span = cls._numeric_pairs(
                    [
                        spans.get(cls.DEFAULT_CLIP_SPAN)
                        if isinstance(spans, dict)
                        else None
                    ]
                )
                if not span:
                    logger.warning(
                        "Clips: nothing says which authored frame %r puts at "
                        "t=0 (no %r span published), and the converter rebases "
                        "every stack onto its first key -- clips left as "
                        "exported.",
                        source.get("name"),
                        cls.DEFAULT_CLIP_SPAN,
                    )
                    return None
                zero = float(span[0][0])

            return GlbClips.rebuild(edit, takes, float(fps), float(zero))

    @classmethod
    def apply_glb_visibility(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """Realize keyed visibility as STEP ``scale`` channels the file can play.

        glTF animates four things -- translation, rotation, scale and morph
        weights -- and visibility is none of them. So a DCC's keyed visibility
        survives the FBX (which has a ``Visibility`` property) and dies at the
        glTF hop, silently: the objects are all present, all visible, all the
        time. Measured on a production assembly, that one gap produced three
        separate-looking complaints -- shots that arrived EMPTY (their only
        content was visibility), objects that never left after their shot, and
        "broken" animation in the shots that mixed both.

        The repair writes the one presence channel every glTF viewer already
        plays. Each gated node gets a ``scale`` channel with ``STEP``
        interpolation, driving the node between its authored scale and zero; a
        zero-scale node collapses to a point and rasterizes nothing, which is
        the established glTF idiom for boolean visibility precisely because it
        needs no extension. Nothing here is optional for the consumer, so the
        deliverable a developer loads in their own viewer behaves like the
        preview without being told anything.

        Smooth *fades* are the other half, and they belong to
        :meth:`apply_glb_fades` -- alpha is a material property, so it needs
        ``KHR_animation_pointer`` rather than a node channel. The two have to
        agree about WHEN, and this pass is where that is decided: a node
        carrying a real ramp is gated on the ramp instead of on the DCC's
        mirrored boolean, because the mirror hides it for exactly the frames it
        should be fading over (see :meth:`_presence_keys`). A node without a
        ramp steps exactly where the DCC's own playback steps.

        Runs BEFORE :meth:`apply_glb_animations`, which reports on what the
        clips hold: a shot whose content is entirely visibility is empty until
        this pass has run, and would otherwise be reported empty and skipped
        as the file's default clip.

        Reads its inputs out of the deliverable itself -- the
        :attr:`VISIBILITY_TRACKS_KEY` channel on the ``data_export`` carrier,
        joined to :attr:`FBX_TAKES_KEY` for the clip windows -- so it is
        self-feeding and needs no argument from a caller that may not know the
        scene.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.

        Returns:
            ``{"nodes": n, "channels": n, "clips": n}`` describing what was
            written, or ``None`` when the file carries no tracks to apply.
        """
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            channel = cls.data_export_channel(gltf, cls.VISIBILITY_TRACKS_KEY)
            if not isinstance(channel, dict):
                return None
            version = channel.get("version")
            if isinstance(version, int) and version > cls.VISIBILITY_TRACKS_VERSION:
                logger.warning(
                    "Visibility: the file declares %s schema v%d and this reader "
                    "knows v%d -- skipped rather than half-applied.",
                    cls.VISIBILITY_TRACKS_KEY,
                    version,
                    cls.VISIBILITY_TRACKS_VERSION,
                )
                return None
            tracks = [
                t
                for t in (channel.get("tracks") or [])
                if isinstance(t, dict) and t.get("node") and t.get("visibility")
            ]
            animations = gltf.get("animations") or []
            if not tracks or not animations:
                return None

            metadata = cls.data_export_channel(gltf, cls.SHOT_METADATA_KEY)
            fps = channel.get("fps")
            if not fps and isinstance(metadata, dict):
                fps = metadata.get("fps")
            try:
                fps = float(fps)
            except (TypeError, ValueError):
                fps = 0.0
            if fps <= 0:
                # Every key here is a FRAME number, and without the rate it was
                # authored at there is no time to place it at. Refusing beats
                # guessing 24 and stepping every gate at the wrong moment.
                logger.warning(
                    "Visibility: %d track(s) carry no frame rate, so their frame "
                    "numbers cannot be placed in time -- visibility not applied.",
                    len(tracks),
                )
                return None

            windows = cls._take_windows(gltf)
            # A clip that is not a declared take -- the exporter's retained
            # whole-timeline stack -- spans every take, which is the range the
            # bake covered. With no takes at all (an animated prop that was
            # never cut into shots) the tracks' own extent is the only window
            # there is, and it is the right one: gating still has to happen.
            if windows:
                union = (
                    min(s for s, _ in windows.values()),
                    max(e for _, e in windows.values()),
                )
            else:
                frames = [
                    float(key[0])
                    for track in tracks
                    for key in cls._numeric_pairs(track["visibility"])
                ]
                union = (min(frames), max(frames)) if frames else None

            by_name: Dict[str, List[int]] = {}
            for index, node in enumerate(gltf.get("nodes") or []):
                name = node.get("name")
                if name is not None:
                    by_name.setdefault(str(name), []).append(index)

            spans = channel.get("clip_span")
            return cls._write_visibility_channels(
                edit,
                tracks,
                animations,
                windows,
                union,
                by_name,
                fps,
                spans if isinstance(spans, dict) else {},
            )

    #: ``clip_span`` entry describing the clip a take split does NOT name -- the
    #: converter's retained whole-timeline stack. Not a legal take name, so it
    #: cannot collide with one.
    DEFAULT_CLIP_SPAN = "*"

    @classmethod
    def clip_spans(
        cls,
        frames: Iterable[float],
        takes: Iterable[Any],
        stack_range: Optional[Sequence[float]] = None,
    ) -> Dict[str, List[float]]:
        """Per take, the first and last authored frame inside its window.

        The producer half of what :meth:`_clip_zero` reads back, kept here
        rather than in each DCC package so the two cannot write the schema
        differently -- the same reason :meth:`build_scene_sidecar` owns the
        sidecar envelope. Each toolkit keeps only the part that needs a scene:
        collecting *frames* (every animated channel's key times, transforms and
        visibility alike, because the converter sizes a take from all of them).

        The :attr:`DEFAULT_CLIP_SPAN` entry is the whole timeline, for the
        full-range stack a take split does not name.

        Parameters:
            frames: Every authored key time in the scene, in frames.
            takes: ``fbx_takes`` entries -- ``{"name", "start", "end"}``.
            stack_range: ``(first, last)`` frame the SOURCE STACK will
                actually carry -- the exported/baked range. Pass it whenever
                the caller knows it. The converter rebases every stack onto
                its first key, so the whole-timeline zero must be the first
                frame the FILE holds, not the first frame the SCENE holds: a
                key authored before the exported range never reaches the FBX,
                and letting it set the zero slides every clip cut from that
                stack by the difference. Measured on a production assembly as
                a 33-frame slide (first take at 33, scene's first key at 0),
                which reads as up to 90 cm of apparent distortion while the
                geometry is exact. Also why a bake matters: it writes a key on
                every frame of the range, so the stack's first key IS the
                range start even when the pre-bake scene had none there.

        Returns:
            ``{take name: [first, last]}``, empty when nothing is animated.
        """
        every = sorted(float(f) for f in frames if isinstance(f, (int, float)))
        bounds = None
        if stack_range is not None:
            try:
                lo, hi = float(stack_range[0]), float(stack_range[1])
                bounds = [lo, hi]
            except (TypeError, ValueError, IndexError):
                bounds = None
        if not every:
            return {cls.DEFAULT_CLIP_SPAN: bounds} if bounds else {}
        spans: Dict[str, List[float]] = {
            cls.DEFAULT_CLIP_SPAN: bounds if bounds else [every[0], every[-1]]
        }
        for take in takes or ():
            if not isinstance(take, dict):
                continue
            try:
                name = str(take["name"])
                start, end = float(take["start"]), float(take["end"])
            except (KeyError, TypeError, ValueError):
                continue
            inside = [t for t in every if start <= t <= end]
            if inside:
                spans[name] = [inside[0], inside[-1]]
        return spans

    @classmethod
    def build_visibility_tracks(
        cls,
        tracks: Sequence[Dict[str, Any]],
        fps: Optional[float] = None,
        clip_spans: Optional[Dict[str, List[float]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Wrap *tracks* in the versioned ``visibility_tracks`` envelope.

        The one place that schema exists. Both DCC packages publish through it
        rather than shaping the dict themselves, so the channel
        :meth:`apply_glb_visibility` reads cannot fork between the toolkit that
        wrote it and the one that did not -- and the two cannot import each
        other to share it.

        Returns ``None`` when there is nothing to publish, which is the
        producers' signal to CLEAR the channel rather than stamp an empty one.
        """
        if not tracks:
            return None
        payload: Dict[str, Any] = {
            "version": cls.VISIBILITY_TRACKS_VERSION,
            "tracks": list(tracks),
        }
        if fps:
            payload["fps"] = float(fps)
        if clip_spans:
            payload["clip_span"] = clip_spans
        return payload

    @classmethod
    def _take_windows(cls, gltf: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """``{take name: (start frame, end frame)}`` from the ``fbx_takes`` channel.

        Shared by the visibility gate and the clip manifest so a take whose
        bounds one of them cannot read is skipped by BOTH -- the alternative
        being a clip that publishes an origin the gates were never placed
        against.
        """
        windows: Dict[str, Tuple[float, float]] = {}
        for take in cls.data_export_channel(gltf, cls.FBX_TAKES_KEY) or []:
            if not isinstance(take, dict) or take.get("name") is None:
                continue
            try:
                windows[str(take["name"])] = (
                    float(take.get("start")),
                    float(take.get("end")),
                )
            except (TypeError, ValueError):
                continue
        return windows

    @classmethod
    def _clip_span_for(
        cls,
        name: str,
        spans: Dict[str, Any],
        windows: Dict[str, Any],
    ) -> Optional[Sequence[float]]:
        """The authored span that applies to the clip called *name*.

        One rule, in one place, because both readers of it place a clip against
        the frame it returns: the visibility gate and the manifest's
        ``zero_frame``. Two copies that disagree would put a node's switch and
        the clip's own published origin at different instants in the same file.

        :attr:`DEFAULT_CLIP_SPAN` describes the converter's retained
        whole-timeline stack ONLY. Letting a DECLARED take fall back to it
        would place that take against the whole timeline's zero -- measured at
        42.5s of drift on a shot 42.5s into the sequence.
        """
        span = spans.get(name)
        if span is None and name not in windows:
            span = spans.get(cls.DEFAULT_CLIP_SPAN)
        return span

    @classmethod
    def _clip_zero(
        cls,
        gltf: Dict[str, Any],
        animation: Dict[str, Any],
        window: Tuple[float, float],
        span: Optional[Sequence[float]],
        fps: float,
        verify: bool = True,
    ) -> float:
        """The authored frame the converter placed at this clip's ``t=0``.

        Not the take's start frame, which is the intuitive answer and the wrong
        one. Measured against FBX2glTF 0.13.1: a take is emitted spanning its
        authored keys, not its declared window, and rebased so the FIRST of
        them lands at zero -- counting the VISIBILITY keys, which size the take
        even though no channel is emitted for them. A gate placed against the
        window start therefore drifts by the take's lead-in, which on a
        production assembly ran to 43 frames.

        The producer publishes that span (it is the only party that can see the
        curves), and this VERIFIES it against the clip actually in the file:
        the two agree when the producer's view of the export set matched the
        converter's. A clip with no channels has nothing to check against and
        nothing to be misaligned with -- its zero is whatever this returns.

        Parameters:
            verify: Compare the published span against the clip in the file.
                Only meaningful BEFORE the gate pass writes, because a gate
                holds its final state to the end of the shot's window and so
                legitimately makes the clip longer than the keys the producer
                measured. A caller reading the zero back out of a finished file
                (the clip manifest) would otherwise report that growth as a
                mismatch on every clip it succeeded on.
        """
        # A clip built by :meth:`apply_glb_clips` was cut to its window, so it
        # KNOWS its origin and says so. Everything below is the inference for a
        # clip that came straight off the converter, where nobody does.
        from pythontk.file_utils.mesh_convert.glb_clips import GlbClips

        declared = (animation.get("extras") or {}).get(GlbClips.ZERO_FRAME_KEY)
        if isinstance(declared, (int, float)):
            return float(declared)

        pair = cls._numeric_pairs([span])
        if not pair:
            return window[0]
        first, last = float(pair[0][0]), float(pair[0][1])
        if not verify:
            return first
        measured = cls._animation_span(gltf, animation)
        if measured is None:
            return first  # empty clip: this pass alone defines its zero
        # One frame of tolerance: the span arrives as authored frames and comes
        # back as seconds through the converter's own rounding.
        if abs((last - first) - measured[1] * fps) > 1.0:
            logger.warning(
                "Visibility: clip %r spans %.3fs in the file but the scene "
                "reports frames %g-%g (%.3fs) -- the export set the gate was "
                "computed against is not the one that shipped, so its switches "
                "may sit up to %.2fs off.",
                animation.get("name"),
                measured[1],
                first,
                last,
                (last - first) / fps,
                abs((last - first) / fps - measured[1]),
            )
        return first

    @classmethod
    def _write_visibility_channels(
        cls,
        edit: "MeshConvert.GlbEdit",
        tracks: List[Dict[str, Any]],
        animations: List[Dict[str, Any]],
        windows: Dict[str, Tuple[float, float]],
        union: Optional[Tuple[float, float]],
        by_name: Dict[str, List[int]],
        fps: float,
        spans: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Sample every track into every clip and write the channels. One BIN append.

        Split from :meth:`apply_glb_visibility` so the reading of the file
        (which channel, which schema, which rate) stays separate from the
        writing of it, and because the write is the half with the ordering
        constraint: the samples for EVERY clip are computed first, so the
        buffer grows once rather than once per clip.
        """
        nodes = edit.gltf.get("nodes") or []
        # What each track is GATED on -- its fade's timeline where it has one,
        # its mirrored boolean where it does not (see :meth:`_presence_keys`).
        # Paired with the track and resolved once, rather than per clip: the
        # answer is a property of the track, and deriving it inside the clip
        # loop would redo the same work once per clip per node.
        to_gate = [(track, cls._presence_keys(track)) for track in tracks]
        # (times, values) -> payload slot, so the three nodes that switch on the
        # same frame with the same authored scale share one accessor pair.
        payloads: List[bytes] = []
        slots: Dict[bytes, int] = {}

        def payload_slot(values: Sequence[float]) -> int:
            raw = struct.pack(f"<{len(values)}f", *values)
            if raw not in slots:
                slots[raw] = len(payloads)
                payloads.append(raw)
            return slots[raw]

        planned: List[Tuple[Dict[str, Any], int, int, int, float, float, int]] = []
        missing: Set[str] = set()
        conflicted: Set[str] = set()
        baked: Set[str] = set()
        gated: Set[str] = set()
        clips: Set[str] = set()

        def report_skips() -> None:
            """Name every node that asked for a gate and did not get one.

            Runs on BOTH exits -- a pass that wrote nothing has exactly the
            same thing to report as one that wrote something, and the first
            draft of this only said it on the way out of the second.
            """
            if missing:
                logger.warning(
                    "Visibility: %d keyed node(s) are not in this GLB (%s) -- "
                    "they will be visible for the whole deliverable.",
                    len(missing),
                    ", ".join(sorted(missing)),
                )
            if conflicted:
                logger.warning(
                    "Visibility: %d node(s) already carry a scale animation "
                    "(%s), so their visibility gate was NOT written -- they "
                    "stay visible.",
                    len(conflicted),
                    ", ".join(sorted(conflicted)),
                )
            if baked:
                logger.warning(
                    "Visibility: %d node(s) carry a baked matrix (%s), which "
                    "glTF forbids animating -- their gate was NOT written. "
                    "Re-export with separate translation/rotation/scale.",
                    len(baked),
                    ", ".join(sorted(baked)),
                )

        for animation in animations:
            name = str(animation.get("name") or "")
            window = windows.get(name) or union
            if window is None:
                continue
            zero = cls._clip_zero(
                edit.gltf,
                animation,
                window,
                cls._clip_span_for(name, spans, windows),
                fps,
            )
            # Nodes this clip already scales: a second channel on the same
            # node/path is undefined behaviour, so the gate stands down rather
            # than corrupt an authored scale animation.
            taken = {
                (c.get("target") or {}).get("node")
                for c in (animation.get("channels") or [])
                if (c.get("target") or {}).get("path") == "scale"
            }
            for track, presence in to_gate:
                target = str(track["node"])
                indices = by_name.get(target)
                if not indices:
                    missing.add(target)
                    continue
                runs = cls._visibility_runs(presence, window)
                if not runs or (len(runs) == 1 and runs[0][1]):
                    # Visible for the whole clip: the default, so no channel.
                    continue
                for index in indices:
                    if index in taken:
                        conflicted.add(target)
                        continue
                    node = nodes[index]
                    if node.get("matrix"):
                        # glTF forbids animating a node that carries a matrix;
                        # TRS is required. Never seen from FBX2glTF, but this is
                        # public API pointed at whatever it is handed. Reported
                        # apart from a scale collision: the two need different
                        # things done to them, and one warning naming both would
                        # send the reader looking for the wrong thing.
                        baked.add(target)
                        continue
                    # glTF requires three components; a file that carries fewer
                    # would raise on the axis walk below rather than degrade.
                    scale = node.get("scale")
                    base = (
                        scale
                        if isinstance(scale, (list, tuple))
                        and len(scale) >= 3
                        and all(isinstance(v, (int, float)) for v in scale[:3])
                        else [1.0, 1.0, 1.0]
                    )
                    # Clamped, because a run can open before the clip's zero --
                    # and then deduplicated, because clamping is what ties two
                    # keys to the same instant.
                    placed = cls._strictly_increasing(
                        [(max(0.0, (f - zero) / fps), v) for f, v in runs]
                    )
                    if not placed or (len(placed) == 1 and placed[0][1]):
                        continue
                    # Hold the final state to the END of the shot's window. A
                    # clip is only as long as its longest sampler, and the
                    # converter sizes a take from its authored KEYS -- so a
                    # shot whose content is one object appearing would arrive
                    # as a clip that ends the instant it appears (measured:
                    # Shot_1, 93 authored frames, a 0.5s clip). The extra key
                    # changes nothing on screen and makes the clip last as long
                    # as the shot it is named after.
                    tail = max(0.0, (window[1] - zero) / fps)
                    if tail > placed[-1][0]:
                        placed.append((tail, placed[-1][1]))
                    times = [t for t, _ in placed]
                    output: List[float] = []
                    for _, value in placed:
                        on = 1.0 if value else 0.0
                        output.extend(float(base[axis]) * on for axis in range(3))
                    planned.append(
                        (
                            animation,
                            index,
                            payload_slot(times),
                            payload_slot(output),
                            times[0],
                            times[-1],
                            len(times),
                        )
                    )
                    gated.add(target)
                    clips.add(name)

        if not planned:
            report_skips()
            return None

        views = cls._append_bin_views(edit, payloads)
        if not views:
            logger.warning(
                "Visibility: this GLB's buffer is external, so the tracks have "
                "nowhere to live -- visibility not applied."
            )
            return None

        accessors = edit.gltf.setdefault("accessors", [])
        # accessor index per (payload slot, kind), so a shared payload is also a
        # shared accessor -- the same three nodes again.
        built: Dict[Tuple[int, str], int] = {}

        def accessor(slot: int, kind: str, count: int, span=None) -> int:
            key = (slot, kind)
            if key not in built:
                entry: Dict[str, Any] = {
                    "bufferView": views[slot],
                    "componentType": 5126,  # FLOAT
                    "count": count,
                    "type": kind,
                }
                if span is not None:  # required on an animation sampler input
                    entry["min"], entry["max"] = [span[0]], [span[1]]
                accessors.append(entry)
                built[key] = len(accessors) - 1
            return built[key]

        for animation, index, in_slot, out_slot, first, last, count in planned:
            samplers = animation.setdefault("samplers", [])
            samplers.append(
                {
                    "input": accessor(in_slot, "SCALAR", count, (first, last)),
                    "output": accessor(out_slot, "VEC3", count),
                    "interpolation": "STEP",
                }
            )
            animation.setdefault("channels", []).append(
                {
                    "sampler": len(samplers) - 1,
                    "target": {"node": index, "path": "scale"},
                }
            )
        edit.dirty = True

        report_skips()
        logger.info(
            "Visibility: %d node(s) gated across %d clip(s) via %d STEP scale "
            "channel(s) -- keyed visibility now plays in any glTF viewer.",
            len(gated),
            len(clips),
            len(planned),
        )
        return {"nodes": len(gated), "channels": len(planned), "clips": len(clips)}

    @classmethod
    def _authored_fades(cls, gltf: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """Per-node alpha ramps from the visibility channel, or ``{}``.

        Only ramps that actually ramp: a track whose ``opacity`` merely mirrors
        its on/off keys says nothing the stepped scale channel does not already
        say, and writing it as an alpha channel would animate a "fade" that
        was never authored as one.
        """
        channel = cls.data_export_channel(gltf, cls.VISIBILITY_TRACKS_KEY)
        if not isinstance(channel, dict):
            return {}
        fades: Dict[str, List[List[float]]] = {}
        for track in channel.get("tracks") or []:
            if not isinstance(track, dict) or not track.get("node"):
                continue
            keys = [
                [float(k[0]), float(k[1])]
                for k in cls._numeric_pairs(track.get("opacity") or [])
            ]
            if len(keys) < 2:
                continue
            # One predicate, in one place: the gate that makes a node PRESENT
            # for its fade reads the same answer, and a node gated as fading
            # whose ramp was not published (or the reverse) would be a node
            # visible with no alpha to apply.
            if cls._is_fade(keys):
                fades[str(track["node"])] = keys
        return fades

    @classmethod
    def apply_glb_fades(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """Realize authored opacity ramps as animated material alpha.

        The other half of :meth:`apply_glb_visibility`. That pass answers
        "is it there", which glTF can express as a scale channel every viewer
        already plays; this one answers "how solid is it", which glTF cannot
        express at all without ``KHR_animation_pointer`` -- so the ramp is
        written through that extension, declared in ``extensionsUsed``, and the
        file fades by itself anywhere the extension is implemented.

        Runs AFTER the gate, and the order is load-bearing in both directions:
        a node has to be PRESENT during its ramp for the ramp to be visible
        (see :meth:`_presence_keys`), and the clip has to end up with a channel
        so it is not reported as an empty shot -- for a shot whose only content
        was a fade, this pass is that channel.

        Reads its inputs out of the deliverable itself, like its neighbours.
        :mod:`~pythontk.file_utils.mesh_convert.glb_fades` does the writing.

        Returns:
            ``{"nodes", "materials", "channels"}``, or ``None`` when the file
            carries no authored ramp.
        """
        from pythontk.file_utils.mesh_convert.glb_fades import GlbFades

        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            fades = cls._authored_fades(gltf)
            if not fades:
                return None
            metadata = cls.data_export_channel(gltf, cls.SHOT_METADATA_KEY)
            channel = cls.data_export_channel(gltf, cls.VISIBILITY_TRACKS_KEY)
            fps = cls._resolve_clip_fps(
                metadata if isinstance(metadata, dict) else {}, []
            )
            if not fps and isinstance(channel, dict):
                try:
                    fps = float(channel.get("fps") or 0.0)
                except (TypeError, ValueError):
                    fps = 0.0
            if not fps:
                logger.warning(
                    "Fades: %d authored ramp(s) carry no frame rate, so their "
                    "frame numbers cannot be placed in time -- not applied.",
                    len(fades),
                )
                return None

            windows = cls._take_windows(gltf)
            spans = (channel or {}).get("clip_span")
            if not isinstance(spans, dict):
                spans = {}
            union = (
                (
                    min(s for s, _ in windows.values()),
                    max(e for _, e in windows.values()),
                )
                if windows
                else None
            )
            # Each clip's own window and origin, resolved exactly the way the
            # gate resolves them -- a ramp placed against a different zero than
            # the gate it accompanies would fade at a different instant than
            # the object appears.
            clip_windows: Dict[str, Tuple[float, float]] = {}
            zeros: Dict[str, float] = {}
            for animation in gltf.get("animations") or []:
                name = str(animation.get("name") or "")
                window = windows.get(name) or union
                if window is None:
                    continue
                clip_windows[name] = window
                zeros[name] = cls._clip_zero(
                    gltf,
                    animation,
                    window,
                    cls._clip_span_for(name, spans, windows),
                    float(fps),
                    verify=False,
                )
            if not clip_windows:
                return None
            return GlbFades.apply(edit, fades, clip_windows, zeros, float(fps))

    @classmethod
    def apply_glb_animations(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """Publish the GLB's clips as ``extras.animation_web``, joined to the shots.

        The GLB consumer half of the shots contract, and the answer to three
        things a shot-split deliverable cannot say for itself (all measured on
        Maya 2025 -> FBX2glTF 0.13.1):

        * **Which clip is a shot.** Maya's exporter keeps its whole-timeline
          ``Take 001`` AnimStack alongside the takes it was asked to split out,
          and it converts to ``animations[0]`` -- so the naive
          ``clipAction(animations[0])`` plays the entire timeline rather than
          the first shot. Each clip here carries ``declared`` -- whether a shot
          asked for it -- and ``default_clip`` names the one to open on.
        * **Where a clip sits on the timeline.** The converter rebases every
          clip to t=0, which is right for playback and loses the sequence: a
          shot authored at frames 20-30 and one at 1-10 both start at zero.
          The declared frame range rides on each clip, with ``offset`` (its
          first DECLARED frame in seconds) and ``zero_frame`` -- the authored
          frame the converter actually put at t=0. The two differ, and only
          the second one converts a playhead: a take is rebased to its first
          authored KEY rather than to its window, so a clip whose motion
          starts late carries a lead-in (measured at 43 frames on a production
          assembly). ``zero_frame`` is what maps clip time back to the frame
          numbers every other field here is quoted in.
        * **What the frames MEAN.** Frame numbers need the rate they were
          authored at; ``fps`` carries it.

        Reads its inputs out of the deliverable itself (the ``fbx_takes`` and
        ``shot_metadata`` channels on the ``data_export`` carrier) rather than
        from the host, so it is self-feeding: it runs unconditionally after
        every conversion, is a clean no-op on a GLB with no animation, and
        needs no argument from a caller that may not know the scene. A GLB with
        animation but no declared shots still gets the block -- names, spans
        and rate are what a player needs whether or not shots exist.

        Also the one place the double encoding is undone. The channels arrive
        as JSON *strings* nested under ``extras.fromFBX.userProperties`` (a
        consumer has to parse the JSON it just parsed), and they are left there
        as provenance; this block is the decoded reading, and it cannot drift
        from them because it is derived from them in the same pass.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.

        Returns:
            The published manifest, or ``None`` when the file has no animation.
        """
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            animations = gltf.get("animations") or []
            if not animations:
                return None

            takes = cls.data_export_channel(gltf, cls.FBX_TAKES_KEY) or []
            metadata = cls.data_export_channel(gltf, cls.SHOT_METADATA_KEY) or {}
            if not isinstance(takes, list):
                takes = []
            if not isinstance(metadata, dict):
                metadata = {}
            by_name = {
                str(t.get("name")): t
                for t in takes
                if isinstance(t, dict) and t.get("name") is not None
            }
            by_clip = {
                str(s.get("clip")): s
                for s in (metadata.get("shots") or [])
                if isinstance(s, dict) and s.get("clip") is not None
            }

            clips: List[Dict[str, Any]] = []
            for index, animation in enumerate(animations):
                name = animation.get("name") or f"animation_{index}"
                clip: Dict[str, Any] = {"name": name, "animation": index}
                if not (animation.get("channels") or []):
                    # Named, listed, and carrying nothing. Maya's split emits an
                    # AnimStack per declared range but bakes no curve for a range
                    # in which nothing moves (a hold, or a shot whose motion
                    # belongs to objects outside the export), so this is a normal
                    # authoring outcome rather than a conversion failure -- and
                    # it is invisible from the clip list, which is what made it
                    # read as broken animation instead of an empty shot.
                    clip["empty"] = True
                span = cls._animation_span(gltf, animation)
                if span:
                    clip["duration"] = round(span[1] - span[0], 6)
                take = by_name.get(name)
                # "declared", not "is a shot": a clip the shot system asked for
                # is the one a player should offer; the rest (Maya's retained
                # full-range stack, anything authored outside the Shots panel)
                # is playable but unnamed by the pipeline.
                clip["declared"] = take is not None
                if take is not None:
                    clip["start_frame"] = take.get("start")
                    clip["end_frame"] = take.get("end")
                shot = by_clip.get(name)
                if shot is not None:
                    for key in ("description", "objects", "section"):
                        if shot.get(key):
                            clip[key] = shot[key]
                clips.append(clip)

            fps = cls._resolve_clip_fps(metadata, clips)
            if fps:
                windows = cls._take_windows(gltf)
                union = (
                    (
                        min(s for s, _ in windows.values()),
                        max(e for _, e in windows.values()),
                    )
                    if windows
                    else None
                )
                channel = cls.data_export_channel(gltf, cls.VISIBILITY_TRACKS_KEY)
                spans = (channel or {}).get("clip_span")
                if not isinstance(spans, dict):
                    spans = {}
                for index, clip in enumerate(clips):
                    start = clip.get("start_frame")
                    if isinstance(start, (int, float)):
                        # Where this clip's first frame sits on the AUTHORING
                        # timeline, in seconds. The clip's own times start at
                        # zero, so without this a sequence cannot be rebuilt
                        # from the file.
                        clip["offset"] = round(start / fps, 6)
                    window = windows.get(clip["name"]) or union
                    span = cls._clip_span_for(clip["name"], spans, windows)
                    if window is None and span is None:
                        # Neither a declared window nor a published span: there
                        # is nothing to place this clip against, and a guessed
                        # origin is worse than an absent one.
                        continue
                    # NOT start_frame, and that is the whole point: the
                    # converter rebases a take to its first authored KEY, not
                    # to its declared window, so t=0 sits at the take's lead-in
                    # -- measured at 43 frames (1.43s) into Shot_5 of a
                    # production assembly. Every frame number published
                    # elsewhere in this block is an AUTHORING frame, so this is
                    # the one value that lets a
                    # consumer convert between the two:
                    #
                    #     authoring_frame = zero_frame + clip_time * fps
                    #
                    # Falls back to the window start when the scene published
                    # no spans -- the best available answer, and the one a
                    # consumer would have assumed anyway.
                    clip["zero_frame"] = round(
                        cls._clip_zero(
                            gltf,
                            animations[index],
                            # ``_clip_zero`` reads the window ONLY when there is
                            # no span, and the guard above has ruled out both
                            # being absent -- so this placeholder can never be
                            # the answer it returns.
                            window or (0.0, 0.0),
                            span,
                            fps,
                            # The gate pass already checked this span against
                            # the file, and has since extended these clips to
                            # their window ends; re-checking here would report
                            # its own tail-hold as a misalignment on every clip.
                            verify=False,
                        ),
                        6,
                    )

            declared = [c for c in clips if c["declared"]]
            # The clip to open on: the first DECLARED one that can actually
            # PLAY, because a deliverable that went to the trouble of splitting
            # shots means the shots -- but opening on an empty one shows 0.00s
            # of nothing and reads as broken animation rather than as a shot
            # that holds. Measured on a production assembly whose Shot_1 was a
            # hold: the preview opened dead with 11 populated clips behind it.
            # Each fallback is narrower than the last, and the final one is
            # unconditional: a file where every clip is empty still gets a
            # default, since naming no clip at all is worse than naming a
            # quiet one.
            playable = [c for c in declared if not c.get("empty")] or [
                c for c in clips if not c.get("empty")
            ]
            manifest: Dict[str, Any] = {
                "version": cls.ANIMATION_WEB_VERSION,
                "clips": clips,
                "default_clip": (playable or declared or clips)[0]["name"],
            }
            if fps:
                manifest["fps"] = fps

            hollow = [c["name"] for c in declared if c.get("empty")]
            if hollow:
                # Not an error -- a declared range can legitimately hold still --
                # but the one thing a reviewer needs told, because the clip is
                # in the list, in the player's dropdown, and plays as nothing.
                logger.info(
                    "Animation: %d declared shot(s) carry no keyframes (%s); "
                    "listed and marked empty, and not opened on.",
                    len(hollow),
                    ", ".join(hollow),
                )

            missing = sorted(set(by_name) - {c["name"] for c in clips})
            if missing:
                # Loud, like every lightmap miss: the takes were declared and
                # the file does not carry them, which means the split did not
                # run (the exporter's task is off) or the names disagree. Both
                # ship a deliverable whose metadata describes clips it lacks.
                #
                # The CONSEQUENCE is spelled out because the bare fact reads as
                # bookkeeping: measured on a production assembly, the WebXR
                # preview of the same scene carried 12 named shots and this
                # deliverable carried one continuous clip -- the reviewer's
                # only clue that they were not looking at the same thing.
                logger.warning(
                    "Animation: %d declared take(s) have no clip in the GLB "
                    "(%s) -- the take split did not reach this file, so it "
                    "carries %d clip(s) where the scene declares %d shot(s). "
                    "A consumer that plays a named shot will not find one.",
                    len(missing),
                    ", ".join(missing),
                    len(clips),
                    len(by_name),
                )

            gltf.setdefault("extras", {})[cls.ANIMATION_WEB_KEY] = manifest
            edit.dirty = True
            return manifest

    @staticmethod
    def _resolve_clip_fps(
        metadata: Dict[str, Any], clips: List[Dict[str, Any]]
    ) -> Optional[float]:
        """The rate the frame numbers were authored at, or ``None``.

        Published by the shot system (``shot_metadata.fps``) and taken from
        there when present. Older producers did not publish it, so it is
        derived from the first declared clip that carries both a frame range
        and a measured span -- the two describe the same interval, one in
        frames and one in seconds, so their ratio IS the rate. Derivation needs
        at least two frames: a single-frame take spans zero seconds and would
        divide by nothing.
        """
        published = metadata.get("fps")
        if isinstance(published, (int, float)) and published > 0:
            return round(float(published), 6)
        for clip in clips:
            start, end = clip.get("start_frame"), clip.get("end_frame")
            duration = clip.get("duration")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            if not duration or end <= start:
                continue
            return round((end - start) / duration, 6)
        return None

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
    def _append_bin_views(
        edit: "MeshConvert.GlbEdit", payloads: Sequence[bytes]
    ) -> List[int]:
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

    @classmethod
    def _relocate_embedded_images(cls, edit: "MeshConvert.GlbEdit") -> int:
        """Move this session's embedded images from the JSON chunk into the BIN.

        Every channel writer embeds as a ``data:`` URI, which keeps its edit
        inside the JSON chunk -- no buffer offsets to recompute, which is the
        part of GLB surgery that silently corrupts a file. That was priced for
        a local preview, but the same writers build deliverables: measured on
        TURRETS_WIRES.glb, the packed ORM's base64 put the JSON chunk at 45% of
        an 8.9 MB file, 1.0 MB of it pure base64 premium, all of it parsed
        before a loader can draw. This pays the JSON back down once, on close,
        after every writer has had its say.

        Safe because it only ever **appends** (see :meth:`_append_bin_views`,
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
        views = cls._append_bin_views(
            edit, [base64.b64decode(image["uri"].split(",", 1)[1]) for image in live]
        )
        if not views:
            return 0
        for image, view in zip(live, views):
            image["bufferView"] = view
            image.pop("uri", None)
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
    def describe_texture_pass(
        cls,
        summary: Dict[str, Any],
        image_format: str,
        max_size: int = 0,
    ) -> str:
        """Human-readable outcome of :meth:`optimize_glb_textures`, for log lines.

        Twin of :meth:`MapOptimizer.describe_size_clamp`, and here for the same
        reason: the wording is a claim about THIS method's contract, and both
        DCC exporters have to make it. They cannot import each other, so a
        sentence written in both drifts in both -- which it already had, the
        Blender copy trailing Maya's.

        The claim worth centralising is that *max_size* is a **ceiling, not a
        target**. An asset authored at the ceiling resamples nothing, so a
        caller that reports the MODE ("resized") tells its user the exporter
        rescaled textures it never touched -- read, on a 2048 set under a 2048
        ceiling, as "it upscaled my maps to 2K". ``summary["resized"]`` is the
        count that actually happened, and this renders it.

        Parameters:
            summary: What :meth:`optimize_glb_textures` returned. Empty means
                the pass ran and replaced nothing, which gets its own sentence
                -- "asked for and got nothing" must not read like "never ran".
            image_format: The carrier the pass was asked for (``"KTX2"``, ...).
            max_size: The ceiling that was in force; ``0`` = never resample.

        Returns:
            One complete sentence, ready to log.
        """
        if not summary:
            return (
                f"GLB texture pass ({image_format}) changed nothing: no "
                "embedded image improved on its original bytes."
            )
        resized = summary.get("resized") or 0
        if not max_size:
            # "Never CLAMP", which is not "never resample": KTX2 snaps every
            # non-exempt image down to a power of two whatever the ceiling
            # (KHR_texture_basisu needs multiple-of-4 edges and a full mip
            # pyramid), so an unconditional "pixels untouched" here is the same
            # false claim this method exists to remove -- over the delivery mode
            # where a silently halved map is what someone would go looking for.
            did = (
                f"{resized} snapped down to power-of-two for {image_format}"
                if resized
                else "container only, pixels untouched"
            )
        elif resized:
            did = f"{resized} resampled down to fit {max_size}px"
        else:
            did = f"none resampled - all were already within {max_size}px"
        return (
            f"GLB textures delivered as {image_format}: "
            f"{summary['images']} image(s), "
            f"{summary['bytes_before'] / 1e6:.1f} MB -> "
            f"{summary['bytes_after'] / 1e6:.1f} MB ({did})."
        )

    #: Texture-container extensions this class binds, and therefore owns the
    #: declarations for. Anything else in ``extensionsUsed`` (material
    #: extensions, ``KHR_texture_transform`` from the lightmap pass) is another
    #: writer's claim and is never touched by the reconciliation below.
    #:
    #: Also the complete list of places a texture can name its image BESIDE the
    #: core ``source``, which is why `GlbEdit.image_for_texture` (resolving a
    #: binding) and `dedupe_glb_images` (rewriting one) both read it: three
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

    @classmethod
    def _reconcile_texture_extensions(cls, gltf: Dict[str, Any]) -> None:
        """Re-derive the texture-container extension declarations from the bindings.

        ``extensionsUsed`` and ``extensionsRequired`` are claims about what a
        file's textures ACTUALLY carry, and every re-encode can invalidate
        them. Declaring them incrementally -- each pass appending what it just
        did -- cannot retract a superseded pass's claim: a WebP delivery
        requires ``EXT_texture_webp`` (nothing core-readable survives it), a
        later KTX2 pass re-encodes past those bindings, and the requirement
        outlived the last binding needing it. glTF 2.0 makes
        ``extensionsRequired`` a **subset** of ``extensionsUsed``, so that
        leftover did not merely warn -- it shipped a file demanding a
        capability it no longer declared, which stock validators reject.

        Derived instead, from the only place the truth lives (the textures):

        - **used** when some texture binds the extension;
        - **required** only when some binding of it has no core-readable
          ``source`` for a stock reader to degrade to -- by construction a
          ``source`` survives beside a container extension only when it points
          at a real PNG/JPEG twin.

        Both arrays carry ``minItems: 1``, so one emptied of this class's
        extensions is removed rather than shipped as ``[]``. Idempotent, and
        self-healing on a file that arrives with stale declarations.
        """
        textures = gltf.get("textures") or []
        for name in cls.TEXTURE_CONTAINER_EXTENSIONS:
            bindings = [t for t in textures if name in (t.get("extensions") or {})]
            claims = (
                ("extensionsUsed", bool(bindings)),
                ("extensionsRequired", any("source" not in t for t in bindings)),
            )
            for key, holds in claims:
                declared = gltf.get(key) or []
                if holds and name not in declared:
                    declared.append(name)
                    gltf[key] = declared
                elif name in declared and not holds:
                    declared.remove(name)
                if key in gltf and not gltf[key]:
                    del gltf[key]

    @classmethod
    def web_delivery_texture_params(
        cls,
        image_format: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """:meth:`optimize_glb_textures` kwargs for a WEB deliverable.

        The one definition of "finished for the web", so the producers of that
        deliverable cannot each hold their own. They did, and it showed:
        measured on one production assembly, in one session, from one scene --
        the WebXR preview shipped 8.71 MB of WebP, the scene exporter with its
        panel dials untouched shipped 280.13 MB of full-resolution PNG, and the
        same exporter with the dials set to WebP shipped 22.06 MB because its
        size ceiling resolved from an absent template budget to "never
        resample". An artist approves the first and hands a developer the
        second; nothing in either log says they differ.

        Both parameters distinguish *unspecified* from *chosen*: ``None`` takes
        the policy, and any other value -- ``0`` for "keep every pixel"
        included -- is the caller's decision. That matters because the callers
        pass values resolved from UI dials whose own "unset" is falsy, and a
        falsy default reaching :meth:`optimize_glb_textures` as an explicit
        argument stops inheriting anything and starts meaning something.

        Deliberately NOT covering ``ktx2_fallback``: it answers "must this open
        in a stock glTF importer?", which is a property of the CONSUMER rather
        than of the delivery. The preview streams to a page that wires
        ``KTX2Loader`` and says ``False``; an exporter handing a developer an
        asset that must also open in Blender or Unreal says ``True``. A shared
        default there would silently make one of them wrong.

        Parameters:
            image_format: Container override; ``None``/empty takes
                :attr:`WEB_DELIVERY_FORMAT`.
            max_size: Longest-edge ceiling in pixels; ``None`` takes
                :attr:`WEB_DELIVERY_MAX_SIZE`, ``0`` skips resizing.

        Returns:
            ``{"image_format": str, "max_size": int}``.
        """
        return {
            "image_format": image_format or cls.WEB_DELIVERY_FORMAT,
            "max_size": (
                cls.WEB_DELIVERY_MAX_SIZE if max_size is None else int(max_size)
            ),
        }

    @classmethod
    def optimize_glb_textures(
        cls,
        glb: GlbTarget,
        max_size: int = WEB_DELIVERY_MAX_SIZE,
        image_format: str = WEB_DELIVERY_FORMAT,
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
        resized so its longest edge is *max_size*, and re-encoded. WebP is the
        default, and because nothing core-readable survives that pass (the
        image IS the WebP), ``EXT_texture_webp`` lands in ``extensionsRequired``
        -- the file says it needs a WebP-capable reader instead of handing a
        core one WebP bytes through a plain ``source``, which glTF 2.0, whose
        core permits image/jpeg and image/png only, does not allow. WebP is
        alpha-capable, universally decoded by WebXR-class browsers, and roughly
        an order of magnitude smaller than PNG at visually equal quality.

        The lossy encoder is used only where the channels ARE colour -- base
        colour and emissive (:attr:`LOSSY_SAFE_SEMANTICS`). Normal and
        metallic-roughness/occlusion maps re-encode LOSSLESS, because WebP's
        lossy mode is YUV 4:2:0 and their X/Z and metalness/occlusion channels
        sit in the half-resolution chroma planes, where they are resampled as
        if the eye were judging them; the same split
        :attr:`BASIS_BY_SEMANTIC` makes for KTX2. Lightmaps are additionally
        exempt from the resize (the bake sized them deliberately) and lossless
        whatever they are sampled as; that exemption is both by the names the
        ``lightmap_web`` manifest lists and structurally, by texCoord-1
        occlusion/emissive binding, so a digest-deduped image whose name lies
        is still protected. A re-encode that comes out larger keeps the
        original bytes.

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
        * **Lightmaps never take a Basis codec** even in KTX2 mode: ETC1S
          would blotch them for the same reason lossy WebP does, UASTC would
          re-quantise a deliberately-authored bake, and their carrier slot's
          colorspace handling is viewer-rebound -- fidelity wins over GPU
          residency for exactly these images. Their container then follows
          *ktx2_fallback*: with fallbacks ON they keep the core-readable PNG,
          because EXT_texture_webp may only stay out of ``extensionsRequired``
          when the texture carries a PNG/JPEG twin, and WebP + twin costs more
          than the PNG alone (measured: 103 KB + 209 KB vs 209 KB). Pure
          delivery takes lossless WebP and requires the extension.
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
            ``bytes_after`` (image payload totals), and ``resized`` -- how many
            of those images were actually RESAMPLED. *max_size* is a ceiling,
            never a target: a source already within it keeps its pixels, so
            ``resized`` is 0 for a whole asset authored at the ceiling and a
            caller reporting the mode ("resized") rather than this count tells
            its user their textures were rescaled when nothing was. Empty when
            Pillow is unavailable or there is nothing to do.
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

            # Both non-core containers encode per SLOT semantic -- KTX2 picks
            # the codec and transfer, WebP picks lossy or lossless -- so the
            # classification has to happen while the glTF structure is in hand.
            semantic_aware = is_ktx2 or image_format == "WEBP"
            semantic_by_image = cls._image_semantics(edit) if semantic_aware else {}
            #: Whether this pass hands a non-exempt image a container glTF core
            #: cannot read. Both are legal only through a TEXTURE-level
            #: extension (``KHR_texture_basisu`` / ``EXT_texture_webp``), which
            #: needs a texture to bind -- so the question "may this image take
            #: the new container" is the same question for both, and gating
            #: only the basisu one left a WebP delivery re-encoding every
            #: orphan it found into a mime nothing in the file enables.
            non_core_output = is_ktx2 or image_format == "WEBP"
            # The images some texture actually samples -- resolved through the
            # shadow-aware walk, so a re-run over an already-optimized GLB sees
            # the EFFECTIVE binding rather than a stale plain ``source``. Gates
            # the encode itself (below) whenever the output is non-core.
            sampled: Set[Optional[int]] = (
                {
                    edit.image_for_texture(t_index)
                    for t_index in range(len(gltf.get("textures") or []))
                }
                if non_core_output
                else set()
            )

            # Pass 1, phase A (serial, cheap): classify every image and collect
            # ONE job per distinct (payload, exemption, semantic) triple. The
            # exemption is part of the key because the same bytes named both as
            # a source texture and as a lightmap must not share the resized
            # encoding; the semantic for the same reason -- bytes sampled as a
            # normal map in one material and as base color in another need a
            # UASTC and an ETC1S encode respectively, and in WebP mode a
            # lossless and a lossy one. (In PNG mode the semantic is a constant
            # None and the key degenerates to the pair.)
            before = after = 0
            jobs: Dict[Tuple[str, bool, Optional[str]], bytes] = {}
            #: job key -> did its pixels actually get resampled. Reported so a
            #: caller can say what HAPPENED instead of which mode ran: a ceiling
            #: is a clamp, so a run whose sources are all already within it
            #: resizes nothing, and a log that calls that "resized" reads as an
            #: upscale to whoever set the ceiling.
            resized_by_key: Dict[Tuple[str, bool, Optional[str]], bool] = {}
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
                if non_core_output and not is_exempt and index not in sampled:
                    # No texture samples this image, so nothing can rebind it
                    # through the container's texture extension -- and that
                    # declaration is gated on an actual rebind *deliberately*:
                    # it can land in extensionsREQUIRED, which would hard-require
                    # a capable viewer for a binding no texture has. Encoding
                    # anyway left the other half of that pair ungated: the mime
                    # rewrite below is driven by ``replacements``, so a GLB
                    # whose textures resolve to none of them shipped
                    # ``image/ktx2`` -- or ``image/webp`` -- with no extension
                    # enabling it, and glTF 2.0 core permits image/jpeg and
                    # image/png only. Keeping the bytes as found is valid
                    # either way, and an image no texture reads is dead payload
                    # whichever format it is in. Exempt lightmaps are bound by
                    # the pass that marks them, so they are never orphans.
                    after += len(payload)
                    continue
                semantic = semantic_by_image.get(index) if semantic_aware else None
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
                # Written from the worker that owns this key -- one call per
                # key, so distinct keys never collide.
                resized_by_key[key] = target != pil.size
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
                # Exempt (lightmap) images in KTX2 mode never take a Basis
                # codec -- the mode's docstring bullet says why -- so they need
                # a container of their own. WebP is the smallest lossless one,
                # but glTF core cannot read it, and EXT_texture_webp may only
                # stay OUT of ``extensionsRequired`` when the texture also
                # carries a PNG/JPEG fallback. Carrying both costs MORE than
                # the plain PNG does (measured on a delivered room: 103 KB WebP
                # + 209 KB fallback vs 209 KB PNG alone), so the whole point of
                # the WebP is gone the moment a fallback is required. Hence:
                # fallbacks on -- the "opens in any reader" mode -- keeps the
                # core-readable PNG and declares no extension at all; pure
                # delivery takes the WebP and requires the extension below.
                if is_ktx2 and is_exempt:
                    pil_format = "WEBP" if not ktx2_fallback else "PNG"
                else:
                    pil_format = image_format
                if pil_format == "PNG":
                    save_kwargs = {}
                elif pil_format == "WEBP" and (
                    is_exempt or semantic not in cls.LOSSY_SAFE_SEMANTICS
                ):
                    # Lossy WebP is YUV 4:2:0 -- chroma at half resolution,
                    # quantized -- so it is only safe where the channels ARE
                    # colour (:attr:`LOSSY_SAFE_SEMANTICS`).
                    #
                    # A lightmap (*is_exempt*) must round-trip pixel-exact: on
                    # near-black texels the subsampling shows as magenta/green
                    # blotching and smears colour across atlas rect borders.
                    # A normal or ORM map is worse off still, because its
                    # channels are not colour at all -- X and Z of a normal,
                    # and occlusion/metalness of an ORM, sit in the chroma
                    # planes and get resampled as if the eye were judging them.
                    # Measured on this pipeline's own maps, 4K sources through
                    # the 2K ceiling at quality 85: base colour holds 37.6 dB,
                    # while normal X falls to 31.7 dB and ORM metalness to
                    # 30.8 dB. That reads as smeared normals and flat, uniform
                    # roughness -- the deliverable looking like it shipped
                    # without those maps rather than with damaged ones.
                    save_kwargs = dict(cls.LOSSLESS_WEBP)
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

            # In KTX2 mode the exempt (lightmap) images took a container of
            # their own (see the encode branch), so the mime is per image
            # rather than per run.
            exempt_mime = "image/webp" if not ktx2_fallback else "image/png"
            for index in replacements:
                images[index]["mimeType"] = (
                    exempt_mime if (is_ktx2 and key_by_index[index][1]) else mime
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
                        else:
                            texture["source"] = fb_index
                    elif src in webp_images:
                        # Same fallback rule as the basisu branch above, and it
                        # has to be: this previously left ``source`` pointing at
                        # the image it had just REPLACED with WebP bytes and
                        # called that a fallback. It is not one -- it is the
                        # same image -- so the file declared EXT_texture_webp as
                        # merely *used* while a core reader, which glTF 2.0
                        # permits only image/jpeg and image/png, resolved the
                        # texture to WebP. Measured on a delivered room: both
                        # lightmap textures shipped that way with
                        # ``extensionsRequired`` unset, i.e. the file advertised
                        # itself as readable by anyone and was not.
                        extensions = texture.setdefault("extensions", {})
                        extensions["EXT_texture_webp"] = {"source": src}
                        fb_index = fallback_image_of.get(src)
                        if fb_index is None:
                            texture.pop("source", None)
                        else:
                            texture["source"] = fb_index
            # Declarations are DERIVED from the bindings above rather than
            # accumulated by whichever branch ran, which is what kept a
            # superseded pass's claim alive (see the method's own docstring).
            cls._reconcile_texture_extensions(gltf)

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
            "resized": sum(
                1
                for index, key in key_by_index.items()
                if index in replacements and resized_by_key.get(key)
            ),
        }
        logger.info(
            "optimize_glb_textures: %d image(s), %.1f MB -> %.1f MB (%d resampled).",
            summary["images"],
            before / 1e6,
            after / 1e6,
            summary["resized"],
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
                A described material covers its LIGHTMAP CLONES too (see
                :attr:`LIGHTMAP_CLONE_SUFFIX`): the envelope was written before
                they existed, so matching literally reports the whole set. Safe
                by construction rather than by assumption -- a clone is a deep
                copy of its base that rebinds only the lightmap slot, and the
                sidecar repair runs BEFORE the lightmap pass, so the binding
                being exempted here is the repaired one the base was checked on.
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
                if not name or (
                    known is not None
                    and (name in known or cls._lightmap_clone_base(name) in known)
                ):
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

        EVERY material carrying the name is paired, not one of them. glTF does
        not require unique material names and this pipeline relies on that:
        the fade pass clones a material per faded subtree and keeps its name
        (the lightmap manifest binds by name), and FBX2glTF itself emits two
        ``ITA_Extras_MAT`` from one production scene. A ``{name: material}``
        dict here silently kept the LAST copy only, so the sidecar's
        metallic-roughness repair landed on a one-primitive fade clone while
        the eleven-primitive original kept the converter's own packing --
        roughness and metalness at 255 everywhere, a fully metallic screen.
        """
        by_name: Dict[str, List[dict]] = {}
        for material in gltf.get("materials", []) or []:
            if material.get("name"):
                by_name.setdefault(material["name"], []).append(material)
        matched = [
            (name, spec, material)
            for name, spec in entries.items()
            for material in by_name.get(name, ())
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
    def _image_prune_refusal(cls, gltf: Dict[str, Any]) -> Optional[str]:
        """Why images may NOT be dropped from *gltf*, or ``None`` when they may.

        Shared by :meth:`prune_glb_unreferenced_textures`, which enforces it,
        and :meth:`dedupe_glb_images`, which must ask the SAME question before
        it rebinds anything: its rebind is only sound if the orphans it creates
        can then be collected, so a dedupe that rebinds into a prune that
        refuses leaves permanently unreachable payload behind and reports
        having removed nothing.

        Both refusals are "bail whole rather than guess" — dead payload beats a
        broken file:

        * an extension that names images DIRECTLY from the root
          (:attr:`_IMAGE_REFERRING_EXTENSIONS`) holds indices the material walk
          cannot see, and renumbering would break them;
        * a ``bufferView`` on any buffer but the embedded BIN (0) would keep
          its ``buffer`` and get an offset into the rebuilt one — garbage
          reads. The same single-buffer assumption `optimize_glb_textures`
          makes.
        """
        foreign = cls._IMAGE_REFERRING_EXTENSIONS & set(
            gltf.get("extensionsUsed") or []
        )
        if foreign:
            return (
                f"{', '.join(sorted(foreign))} references images outside the "
                "material tree."
            )
        if any(v.get("buffer", 0) != 0 for v in gltf.get("bufferViews") or []):
            return "the file has bufferViews outside the embedded BIN (buffer 0)."
        return None

    @classmethod
    def dedupe_glb_images(cls, glb: GlbTarget) -> Dict[str, int]:
        """Collapse byte-identical embedded images onto one copy.

        FBX2glTF embeds per MATERIAL, so two DCC materials wiring one texture
        file arrive as two images with identical bytes -- and glTF has no
        reason to keep both, since a texture names an image by index and any
        number may name the same one. The session's ``image_digests`` dedupe
        cannot reach this: it is write-side, stopping a WRITER from appending a
        payload already embedded, and by the time it runs the converter's own
        pair is already in the file. Measured on a production assembly: two
        duplicate pairs, 154.7 KB of a delivered 5.9 MB -- and, because this
        runs before :meth:`optimize_glb_textures`, two full-size decodes,
        resizes and re-encodes that bought nothing.

        Content-addressed, never name-addressed, for the same reason
        ``image_digests`` is: the copies arrive by different routes and their
        names are the one thing that may lie in either direction -- two
        different maps can share a name, and one map can carry two.

        Every texture pointing at a dropped image is rebound to the survivor
        (``source`` and every entry of :attr:`TEXTURE_CONTAINER_EXTENSIONS`,
        the same set `image_for_texture` resolves a binding through), then
        the orphaned images and their exclusive bufferViews are reclaimed by
        :meth:`prune_glb_unreferenced_textures` -- which already owns index
        remapping and BIN repacking, so this pass only has to decide WHAT is
        redundant. Textures themselves are kept: two textures sampling one
        image may still differ by sampler, and collapsing them is a separate
        question this pass does not answer.

        Parameters:
            glb: Path to a ``.glb``, modified in place, or an open session.

        Returns:
            ``{"images": n, "bytes": n}`` -- images dropped and BIN payload
            reclaimed. A file with nothing to collapse is not rewritten.
        """
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            images = gltf.get("images") or []
            if len(images) < 2:
                return {"images": 0, "bytes": 0}
            # BEFORE touching anything: the rebind below is only sound if the
            # orphans it creates can then be collected, and the prune that
            # collects them has two refusals. Rebinding into one of those would
            # strand payload no later pass can ever reach.
            refusal = cls._image_prune_refusal(gltf)
            if refusal:
                logger.info("dedupe_glb_images: skipped, %s", refusal)
                return {"images": 0, "bytes": 0}

            # digest -> the index that keeps the payload, first occurrence
            # wins: the numbering a reader already saw stays as stable as
            # dropping anything allows. Computed here rather than read off the
            # session's ``image_digests``, which holds the same rule but only
            # the SURVIVING index per digest -- it cannot say which other
            # indices carried those bytes, which is the whole question here,
            # and looking each one up would mean hashing the set twice.
            survivor: Dict[str, int] = {}
            replacement: Dict[int, int] = {}
            for index, image in enumerate(images):
                payload = edit.image_bytes(image)
                if not payload:
                    continue
                first = survivor.setdefault(hashlib.sha256(payload).hexdigest(), index)
                if first != index:
                    replacement[index] = first
            if not replacement:
                return {"images": 0, "bytes": 0}

            for texture in gltf.get("textures") or []:
                source = texture.get("source")
                if source in replacement:
                    texture["source"] = replacement[source]
                extensions = texture.get("extensions") or {}
                for name in cls.TEXTURE_CONTAINER_EXTENSIONS:
                    binding = extensions.get(name)
                    if (
                        isinstance(binding, dict)
                        and binding.get("source") in replacement
                    ):
                        binding["source"] = replacement[binding["source"]]
            edit.dirty = True
            # The duplicates are now unreferenced; the prune owns dropping them
            # (and their exclusive views), remapping every surviving index and
            # rebuilding the BIN.
            dropped = cls.prune_glb_unreferenced_textures(edit)

        if dropped["images"]:
            logger.info(
                "dedupe_glb_images: collapsed %d duplicate image(s), %.1f MB of "
                "BIN payload.",
                dropped["images"],
                dropped["bytes"] / (1024 * 1024),
            )
        return {"images": dropped["images"], "bytes": dropped["bytes"]}

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
            refusal = cls._image_prune_refusal(gltf)
            if refusal:
                logger.info("prune_glb_unreferenced_textures: skipped, %s", refusal)
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
    def set_glb_alpha_mode(
        cls, glb: GlbTarget, alpha_mode: Dict[str, Dict[str, Any]]
    ) -> List[Dict]:
        """Write ``alphaMode`` / ``alphaCutoff`` into a GLB's materials, by name.

        The sibling of :meth:`set_glb_base_color` for the one material fact
        FBX2glTF has to guess: it derives ``alphaMode`` from the base colour's
        alpha channel alone, so an RGBA base colour is always ``BLEND`` -- a
        cutout material (``MASK``: alpha test, depth writes, the opaque queue)
        cannot be expressed by the FBX at all, and a solid body wearing a
        blended material sorts its own back faces through its front in every
        WebXR viewer. The DCC knows which it authored (Maya's
        ``SceneState._read_alpha_mode`` reads the StingrayPBS graph) and says
        so through this section.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            alpha_mode: ``{material_name: {"mode": "OPAQUE"|"MASK"|"BLEND",
                "cutoff": float}}``. ``cutoff`` is written only for ``MASK``
                (absent = glTF's default 0.5); any other mode drops a stale
                one. Names absent from the GLB are reported, an unknown mode
                is skipped with a warning rather than written.

        Returns:
            List of records: ``material``, ``alphaMode``, ``alphaCutoff``.
        """
        if not alpha_mode:
            return []
        records: List[Dict] = []
        with cls.open_glb(glb) as edit:
            for name, spec, mat in cls._match_glb_materials(
                edit.gltf, alpha_mode, "set_glb_alpha_mode"
            ):
                mode = str(spec.get("mode") or "").upper()
                if mode not in ("OPAQUE", "MASK", "BLEND"):
                    logger.warning(
                        "set_glb_alpha_mode: %s has no glTF alphaMode %r -- skipped.",
                        name,
                        spec.get("mode"),
                    )
                    continue
                mat["alphaMode"] = mode
                edit.dirty = True
                cutoff = spec.get("cutoff") if mode == "MASK" else None
                if cutoff is None:
                    mat.pop("alphaCutoff", None)
                else:
                    mat["alphaCutoff"] = float(cutoff)
                records.append(
                    {
                        "material": name,
                        "alphaMode": mode,
                        "alphaCutoff": mat.get("alphaCutoff"),
                    }
                )
        return records

    @classmethod
    def set_glb_normal_scale(
        cls, glb: GlbTarget, scale: float, lightmapped_only: bool = True
    ) -> int:
        """Write ``normalTexture.scale`` into a GLB's materials.

        How strongly a normal map reads is a DELIVERY decision, not an
        authoring one, and it is felt hardest on baked geometry: a lightmap
        contributes irradiance with no direction in it, so on a lightmapped
        surface the normal map survives only through the environment term --
        which the viewer deliberately dims, because that surface's lighting is
        already in its bake. The result is a correct but flat-looking room, and
        the dial that brings the detail back without touching the bake is this
        one.

        Written as ``normalTexture.scale`` because glTF already has the field:
        it is core (no extension), every runtime honours it, and three.js reads
        it straight into ``material.normalScale``. So the preview's slider and
        the delivered file say the same thing in the same place, and a value
        saved here comes back on the next load with nothing to re-apply it.

        Parameters:
            glb: Path to a binary glTF (.glb), modified in place, or an open
                :class:`GlbEdit` session whose owner will write it.
            scale: The multiplier. ``1.0`` is glTF's default and REMOVES the
                key rather than writing it, so a reset leaves the file as it
                would have been had the dial never moved.
            lightmapped_only: Restrict to the materials
                ``extras.lightmap_web`` names -- the ones the flattening
                applies to. ``False`` writes every material carrying a normal
                map. A file with no manifest matches nothing under ``True``.

        Returns:
            How many materials were changed.
        """
        scale = float(scale)
        changed = 0
        with cls.open_glb(glb) as edit:
            named = None
            if lightmapped_only:
                manifest = (edit.gltf.get("extras") or {}).get(cls.LIGHTMAP_WEB_KEY)
                named = set((manifest or {}).get("materials") or ())
            for material in edit.materials:
                if named is not None and material.get("name") not in named:
                    continue
                normal = material.get("normalTexture")
                # No normal map means nothing to scale -- writing the key
                # anyway would be a claim about a texture the material does
                # not have.
                if not isinstance(normal, dict) or normal.get("index") is None:
                    continue
                current = normal.get("scale", 1.0)
                if scale == 1.0:
                    if normal.pop("scale", None) is None:
                        continue
                elif current == scale:
                    continue
                else:
                    normal["scale"] = scale
                edit.dirty = True
                changed += 1
        if changed:
            logger.info(
                "Normal scale %.3g written to %d material(s)%s.",
                scale,
                changed,
                " (lightmapped only)" if lightmapped_only else "",
            )
        return changed

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
