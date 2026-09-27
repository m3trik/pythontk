# !/usr/bin/python
# coding=utf-8
"""The web-delivery texture pass: downsize and re-encode a GLB's images.

Owns the delivery policy (format, size ceilings, Basis/UASTC choices per
semantic) and :meth:`optimize_glb_textures`, which applies it and declares
the texture-container extensions it binds.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through ``cls``.
"""

import hashlib
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pythontk.core_utils.class_property import ClassProperty
from pythontk.file_utils.mesh_convert.glb.edit import GlbTarget

logger = logging.getLogger(__name__)


class _TexturesMixin:
    """Web-delivery texture pass and its policy.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    @ClassProperty
    def OPTIMIZE_WORKERS(cls) -> int:
        """Deprecated alias of :attr:`ImgUtils.ENCODE_WORKERS`, for one release.

        The encode pass's worker cap moved to :meth:`ImgUtils.encode_workers`,
        the one policy every texture pass shares, and this reads it. Read-only:
        assigning it no longer changes the pass -- set
        ``ImgUtils.ENCODE_WORKERS``, or pass ``workers=``.
        """
        from pythontk.img_utils._img_utils import ImgUtils

        return ImgUtils.ENCODE_WORKERS

    #: Longest edge a web deliverable's textures keep, the container they are
    #: re-encoded to, and whether a KTX2 image also carries a PNG/JPEG twin
    #: (no). See :meth:`web_delivery_texture_params` for why each is a named
    #: constant rather than each producer's own literal.
    WEB_DELIVERY_MAX_SIZE = 2048
    WEB_DELIVERY_FORMAT = "WEBP"
    WEB_DELIVERY_KTX2_FALLBACK = False
    #: A LOWER ceiling for the packed data maps (:attr:`SECONDARY_SEMANTICS`);
    #: 0 = the same ceiling as everything else. Off by policy: which maps may
    #: lose resolution is a lookdev call the producer makes per deliverable.
    WEB_DELIVERY_SECONDARY_MAX_SIZE = 0
    #: UASTC RDO lambda (``--uastc_rdo_l``) for the KTX2 encodes; None = off.
    #: Off by policy: it trades a measured quality cost for bytes (ORM packs
    #: -30% at 1.0 at PSNR 50/44/48 dB, a noisy normal map only -3.5%) at 3-4x
    #: the encode time -- the producer's call, not the container's.
    WEB_DELIVERY_UASTC_RDO: Optional[float] = None
    #: RDO dictionary size (``--uastc_rdo_d``) whenever an RDO lambda is on;
    #: None = toktx's own (4096). The dictionary is where RDO's time goes.
    #: Measured through this pass on a production set's own maps (8 normals
    #: at 4K, 8 ORM packs at 2K, lambda 1, 2026-09-19): 245 s -> 129 s at
    #: 1024 for +2.2% image bytes (normals +1.8%, ORM +3.5%); 256 bought only
    #: 20 s more for +4.6%. A web export is time-bound, and the maintainer's
    #: call is that those bytes do not justify the time; a standalone map tool
    #: exposes the dial instead.
    WEB_DELIVERY_UASTC_RDO_DICTIONARY: Optional[int] = 1024

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

    #: The semantics a secondary (lower) size ceiling applies to: the packed
    #: data maps -- metallic-roughness and occlusion -- smooth masks that read
    #: the same at half the resolution. NOT normals, which carry the surface
    #: detail a resample visibly softens (toktx caps their RDO for the same
    #: reason), and not color, the perceptual detail.
    SECONDARY_SEMANTICS: Tuple[str, ...] = ("data",)

    @ClassProperty
    def UASTC_RDO_NORMAL_MAX(cls) -> float:
        """The RDO lambda a normal map is capped at whatever the caller asks:
        toktx's guidance, owned by :attr:`Ktx2Encoder.UASTC_RDO_NORMAL_MAX` so
        the map optimizer applies the same number. Read-only."""
        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        return Ktx2Encoder.UASTC_RDO_NORMAL_MAX

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
    #: against the authored map. In lossless mode ``quality`` is encoder
    #: EFFORT, not fidelity: measured over a production room's 29 images,
    #: 75 wrote the same 27.7 MB as 100 in 14% less time, and the encodes
    #: are this pass's critical path (one 2K normal map is ~10 s of it).
    LOSSLESS_WEBP: Dict[str, Any] = {"lossless": True, "quality": 75}

    @classmethod
    def drop_glb_texture_fallbacks(cls, glb: GlbTarget) -> Dict[str, int]:
        """Drop the PNG/JPEG twin of every texture that also ships KTX2.

        ``KHR_texture_basisu`` is written as an OPTIONAL extension: the texture
        names a supercompressed image through the extension and an ordinary one
        through ``source``, so a reader without the extension still has a
        picture. The cost is that the deliverable carries both encodings of
        every map -- on the PROPS assembly, 40.66 MB of PNG beside 34.71 MB of
        KTX2, so the KTX2 pass more than doubled the texture budget it was
        meant to cut.

        This drops the fallbacks and moves ``KHR_texture_basisu`` into
        ``extensionsRequired``, which is what the spec asks of a file that can
        no longer be read without it. That is a DELIVERY decision, not a
        cleanup: the result needs a reader with the extension (in three.js, a
        registered ``KTX2Loader``). Callers that cannot promise one must not
        run this pass.

        Deliberately NOT wired into :meth:`fbx_to_glb` -- unlike
        :meth:`compact_glb_animations`, which is byte-exact and therefore
        unconditional. The pipeline owns this decision one step earlier:
        :meth:`web_delivery_texture_params` builds with ``ktx2_fallback=False``,
        so ``optimize_glb_textures`` never writes the twin in the first place.
        This is the RETROFIT, for a finished GLB that already carries twins --
        one built before that policy, or by a caller that asked for them.

        Only a texture that HAS a KTX2 twin loses its fallback -- a texture
        shipping a lone PNG keeps it -- and the orphaned images are collected
        by :meth:`prune_glb_unreferenced_textures`, which owns image
        renumbering and BIN repacking.

        Parameters:
            glb: Path to a ``.glb``, modified in place, or an open session.

        Returns:
            ``{"textures": n, "images": n, "bytes": n}``.
        """
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            textures = gltf.get("textures") or []
            if not textures:
                return {"textures": 0, "images": 0, "bytes": 0}
            touched = 0
            for texture in textures:
                extension = (texture.get("extensions") or {}).get("KHR_texture_basisu")
                if not isinstance(extension, dict):
                    continue
                if not isinstance(extension.get("source"), int):
                    continue
                if isinstance(texture.get("source"), int):
                    del texture["source"]
                    touched += 1
            if not touched:
                return {"textures": 0, "images": 0, "bytes": 0}

            required = gltf.setdefault("extensionsRequired", [])
            if "KHR_texture_basisu" not in required:
                required.append("KHR_texture_basisu")
            used = gltf.setdefault("extensionsUsed", [])
            if "KHR_texture_basisu" not in used:
                used.append("KHR_texture_basisu")
            edit.dirty = True

            before = len(gltf.get("images") or [])
            result = cls.prune_glb_unreferenced_textures(edit)
            dropped = before - len(gltf.get("images") or [])
            reclaimed = int((result or {}).get("bytes") or 0)
            logger.info(
                "Textures: dropped %d PNG/JPEG fallback(s) beside their KTX2 "
                "twin, reclaiming %.2f MB -- KHR_texture_basisu is now "
                "REQUIRED, so the reader must support it.",
                dropped,
                reclaimed / 1048576.0,
            )
            return {"textures": touched, "images": dropped, "bytes": reclaimed}

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
    def _uastc_rdo_for(
        cls, semantic: Optional[str], uastc_rdo: Optional[float]
    ) -> Optional[float]:
        """The RDO lambda a UASTC encode of *semantic* takes: the caller's,
        capped at :attr:`UASTC_RDO_NORMAL_MAX` for a normal map; None = off."""
        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        return Ktx2Encoder.rdo_for(uastc_rdo, normal_map=semantic == "normal")

    @classmethod
    def describe_texture_pass(
        cls,
        summary: Dict[str, Any],
        image_format: str,
        max_size: int = 0,
        secondary_max_size: int = 0,
        uastc_rdo: Optional[float] = None,
        uastc_rdo_dictionary: Optional[int] = None,
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
            secondary_max_size: The data-map ceiling in force, if any.
            uastc_rdo: The UASTC RDO lambda in force, if any.
            uastc_rdo_dictionary: The RDO dictionary size in force, if any
                (None is toktx's own); named only beside an RDO lambda.

        Returns:
            One complete sentence, ready to log.
        """
        if not summary:
            return (
                f"GLB texture pass ({image_format}) changed nothing: no "
                "embedded image improved on its original bytes."
            )
        resized = summary.get("resized") or 0
        named_secondary = False
        if not max_size:
            # "Never CLAMP", which is not "never resample": Secondary Map Size
            # is a row of its own and still caps the data maps it names, and
            # KTX2 snaps every non-exempt image down to a power of two
            # whatever the ceiling (KHR_texture_basisu needs multiple-of-4
            # edges and a full mip pyramid). An unconditional "pixels
            # untouched" here is the same false claim this method exists to
            # remove -- over the delivery modes where a silently halved map
            # is what someone would go looking for.
            causes = []
            if secondary_max_size:
                causes.append(f"data maps to {secondary_max_size}px")
            if str(image_format).upper() == "KTX2":
                causes.append(f"power-of-two edges for {image_format}")
            if resized:
                did = f"{resized} resampled down"
                if causes:
                    did += f" ({'; '.join(causes)})"
                    named_secondary = bool(secondary_max_size)
            else:
                did = "container only, pixels untouched"
        elif resized:
            did = f"{resized} resampled down to fit {max_size}px"
        else:
            did = f"none resampled - all were already within {max_size}px"
        dials = []
        if secondary_max_size and not named_secondary:
            dials.append(f"data maps capped at {secondary_max_size}px")
        if uastc_rdo:
            dials.append(
                f"UASTC RDO lambda {uastc_rdo:g}"
                + (
                    f" (dictionary {uastc_rdo_dictionary})"
                    if uastc_rdo_dictionary
                    else ""
                )
            )
        if dials:
            did += "; " + ", ".join(dials)
        return (
            f"GLB textures delivered as {image_format}: "
            f"{summary['images']} image(s), "
            f"{summary['bytes_before'] / 1e6:.1f} MB -> "
            f"{summary['bytes_after'] / 1e6:.1f} MB ({did})."
        )

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
        ktx2_fallback: Optional[bool] = None,
        secondary_max_size: Optional[int] = None,
        uastc_rdo: Optional[float] = None,
        uastc_rdo_dictionary: Optional[int] = None,
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

        Every parameter distinguishes *unspecified* from *chosen*: ``None``
        takes the policy, and any other value -- ``0`` for "keep every pixel"
        included -- is the caller's decision. That matters because the callers
        pass values resolved from UI dials whose own "unset" is falsy, and a
        falsy default reaching :meth:`optimize_glb_textures` as an explicit
        argument stops inheriting anything and starts meaning something.

        ``ktx2_fallback`` is policy too, and off: the PNG/JPEG twin a KTX2 image
        can carry serves only a stock glTF importer, and a web deliverable is
        read by a basisu-capable viewer. Leaving it to each producer ("a
        property of the consumer") let the exporters keep the twins -- a
        production 4K assembly shipped 145.8 MB of them beside 123.1 MB of KTX2,
        48% of the GLB. A caller whose GLB must also open in Blender or Unreal
        asks for the twins here -- the scene exporters' ``KTX2 + PNG/JPEG``
        Texture File Type does.

        Parameters:
            image_format: Container override, as a format id (``"WEBP"``) or a
                file extension (``"jpg"``, the vocabulary of the Scene
                Exporter's Texture File Type row) -- returned as the id
                :meth:`optimize_glb_textures` needs. ``None``/empty takes
                :attr:`WEB_DELIVERY_FORMAT`.
            max_size: Longest-edge ceiling in pixels; ``None`` takes
                :attr:`WEB_DELIVERY_MAX_SIZE`, ``0`` skips resizing.
            ktx2_fallback: Whether each KTX2 image also carries a PNG/JPEG
                twin (KTX2 mode only); ``None`` takes
                :attr:`WEB_DELIVERY_KTX2_FALLBACK` (no twins).
            secondary_max_size: A lower ceiling for the packed data maps
                (:attr:`SECONDARY_SEMANTICS`); ``None`` takes
                :attr:`WEB_DELIVERY_SECONDARY_MAX_SIZE`, ``0`` follows
                ``max_size``.
            uastc_rdo: UASTC RDO lambda for the KTX2 encodes; ``None`` takes
                :attr:`WEB_DELIVERY_UASTC_RDO`, ``0`` is off.
            uastc_rdo_dictionary: RDO dictionary size for those encodes;
                ``None`` takes :attr:`WEB_DELIVERY_UASTC_RDO_DICTIONARY`, ``0``
                is toktx's own.

        Returns:
            ``{"image_format": str, "max_size": int, "ktx2_fallback": bool,
            "secondary_max_size": int, "uastc_rdo": float | None,
            "uastc_rdo_dictionary": int | None}``.
        """
        return {
            "image_format": cls._image_format_id(image_format)
            or cls.WEB_DELIVERY_FORMAT,
            "max_size": (
                cls.WEB_DELIVERY_MAX_SIZE if max_size is None else int(max_size)
            ),
            "ktx2_fallback": (
                cls.WEB_DELIVERY_KTX2_FALLBACK
                if ktx2_fallback is None
                else bool(ktx2_fallback)
            ),
            "secondary_max_size": (
                cls.WEB_DELIVERY_SECONDARY_MAX_SIZE
                if secondary_max_size is None
                else int(secondary_max_size)
            ),
            "uastc_rdo": (
                cls.WEB_DELIVERY_UASTC_RDO
                if uastc_rdo is None
                else (float(uastc_rdo) or None)
            ),
            "uastc_rdo_dictionary": (
                cls.WEB_DELIVERY_UASTC_RDO_DICTIONARY
                if uastc_rdo_dictionary is None
                else (int(uastc_rdo_dictionary) or None)
            ),
        }

    @classmethod
    def _image_format_id(cls, container: Optional[str]) -> str:
        """*container* as the format id :meth:`optimize_glb_textures` needs.

        That pass hands ``image_format`` straight to Pillow AND builds the glTF
        mime as ``image/<lowercased>``, so a container's file extension is not
        always the right token: ``jpg`` is a legal filename suffix and the
        Scene Exporter's Texture File Type row offers it, but Pillow only knows
        ``JPEG`` and glTF only accepts ``image/jpeg`` -- ``JPG`` raises mid-encode
        and, had it not, would write an invalid mime. Canonicalised through
        :attr:`IMAGE_MIME_TYPES` rather than a private alias table, so the
        mapping stays the one glTF itself is keyed on. Both exporters carried a
        copy of this; the web policy and the pass now share one.

        Returns:
            ``"PNG"`` / ``"JPEG"`` / ``"WEBP"`` / ``"KTX2"`` ...; ``""`` for an
            empty *container*.
        """
        token = str(container or "").strip().lower().lstrip(".")
        if not token:
            return ""
        mime = cls.IMAGE_MIME_TYPES.get(f".{token}", "")
        return (mime.split("/")[-1] or token).upper()

    @staticmethod
    def _largest_first(jobs: Dict[Any, bytes]) -> List[Any]:
        """The encode jobs, biggest source first -- the order a pool should
        start them in.

        A pool that starts jobs in insertion order can pick up the one
        4096 UASTC encode with RDO last and run it alone after every worker
        has gone idle: on a production GLB the longest single encode was 260 s
        of the pass's 302 s wall. Longest-first is the classic makespan fix,
        and the source payload's size is the cost proxy on hand (pixels and
        codec decide the time; both scale with it).
        """
        return sorted(jobs, key=lambda key: len(jobs[key]), reverse=True)

    @classmethod
    def optimize_glb_textures(
        cls,
        glb: GlbTarget,
        max_size: int = WEB_DELIVERY_MAX_SIZE,
        image_format: str = WEB_DELIVERY_FORMAT,
        quality: int = 85,
        workers: Optional[int] = None,
        ktx2_fallback: bool = WEB_DELIVERY_KTX2_FALLBACK,
        secondary_max_size: int = WEB_DELIVERY_SECONDARY_MAX_SIZE,
        uastc_rdo: Optional[float] = WEB_DELIVERY_UASTC_RDO,
        uastc_rdo_dictionary: Optional[int] = WEB_DELIVERY_UASTC_RDO_DICTIONARY,
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
        * **No core-readable fallback by default** (*ktx2_fallback*, the
          web-delivery policy): the extension lands in
          ``extensionsRequired`` and the deliverable needs a
          ``KHR_texture_basisu``-capable viewer (three.js ``KTX2Loader``; the
          bundled preview page wires it). ``ktx2_fallback=True`` embeds a
          resized PNG/JPEG twin per converted image, bound as the texture's
          plain ``source`` -- the escape hatch the spec defines -- so the
          extension stays in ``extensionsUsed`` and a stock glTF importer
          (Blender, Unreal, Unity) still opens the file. That premium is a
          second copy, not a modest one: UASTC-class images (normals,
          metallic-roughness/occlusion) fall back to PNG, about the size of
          their KTX2, and ETC1S color to JPEG at *quality* (PNG when it
          carries alpha) -- measured on a production 4K assembly, 145.8 MB
          of twins beside 123.1 MB of KTX2. A fallback whose own encode
          fails is logged and dropped, and that image's binding re-tips the
          extension into ``extensionsRequired`` -- never an unreadable
          texture with a declaration claiming otherwise.
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
                :meth:`ImgUtils.encode_workers` (a memory-capped count shared
                with every other texture pass); 1 forces the serial path.
            ktx2_fallback: KTX2 mode only. ``False`` (default,
                :attr:`WEB_DELIVERY_KTX2_FALLBACK`) ships KTX2 alone and
                hard-requires a basisu-capable viewer
                (``extensionsRequired``). ``True`` embeds a core-readable
                PNG/JPEG twin per converted image and binds it as the
                texture's plain ``source``, keeping the GLB importable
                everywhere (extension in ``extensionsUsed``).
            secondary_max_size: A lower longest-edge ceiling for the packed
                data maps alone (:attr:`SECONDARY_SEMANTICS`: metallic-
                roughness / occlusion), never above *max_size*; ``0`` follows
                *max_size*. Color and normal maps keep the primary ceiling.
            uastc_rdo: KTX2 mode only. UASTC rate-distortion lambda for the
                normal / data encodes (``toktx --uastc_rdo_l``); a normal map
                is capped at :attr:`UASTC_RDO_NORMAL_MAX`. ``None`` is off.
                Measured on a 4K production set: ORM packs -30% at 1.0 (PSNR
                50/44/48 dB), a noisy normal map -3.5%, encode 3-4x slower.
            uastc_rdo_dictionary: KTX2 mode with *uastc_rdo* only. The RDO
                dictionary size (``toktx --uastc_rdo_d``); the default is the
                web-delivery policy (:attr:`WEB_DELIVERY_UASTC_RDO_DICTIONARY`),
                ``None`` toktx's own.

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

        # An extension ("jpg") becomes Pillow's id; an empty one is the policy's.
        image_format = cls._image_format_id(image_format) or cls.WEB_DELIVERY_FORMAT
        is_ktx2 = image_format == "KTX2"
        mime = f"image/{image_format.lower()}"
        encoder = None
        if is_ktx2:
            # Resolve before any work, and never fall back silently: a caller
            # who asked for GPU-resident compression and silently got WebP
            # would ship a deliverable that *looks* optimized while missing
            # the entire point of the request.
            from pythontk.img_utils._img_utils import ImgUtils
            from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

            encoder = ImgUtils.resolve_ktx2_encoder(required=True)
            # A typo in either RDO dial is the caller's, raised before any work:
            # inside the per-image encode it would read as one failed encode per
            # image, each shipping its original bytes.
            Ktx2Encoder.rdo_for(uastc_rdo)
            uastc_rdo_dictionary = Ktx2Encoder.rdo_dictionary(uastc_rdo_dictionary)
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            images = gltf.get("images") or []
            if not images:
                return {}

            # The bake sized the lightmaps deliberately; never resize them.
            exempt: Set[str] = set()
            # Read where the viewer reads it: a native DCC export writes the
            # first scene's extras, which the root alone never saw.
            web_manifest = cls._lightmap_web_manifest(gltf)
            published = (web_manifest or {}).get("materials")
            if isinstance(published, dict):
                for entry in published.values():
                    if isinstance(entry, dict) and entry.get("map"):
                        exempt.add(os.path.basename(str(entry["map"])))
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
            # Shadow-rig maps are KEPT AS FOUND -- neither resized nor
            # re-encoded, in any mode. The horizon map is data (span heights,
            # a distance field and a pyramid in its channels): a lossy encode
            # corrupts it outright, and even a lossless one is free to rewrite
            # the RGB of alpha-0 texels -- which here carry a 16-bit distance
            # or a span's bottom. The silhouettes and their atlas were sized by the
            # rasterizer, and the viewer's shim samples them by rect.
            kept: Set[int] = set()
            shadow_manifest = (gltf.get("extras") or {}).get(cls.SHADOW_WEB_KEY)
            for t_index in cls._shadow_web_texture_indices(shadow_manifest):
                src = edit.image_for_texture(t_index)
                if src is not None:
                    kept.add(src)

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
                if index in kept:
                    after += len(payload)
                    continue
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
                # The packed data maps may take a LOWER ceiling than the rest
                # (the primary still bounds it); color and normals keep the
                # primary -- SECONDARY_SEMANTICS says why.
                ceiling = max_size
                if secondary_max_size and semantic in cls.SECONDARY_SEMANTICS:
                    ceiling = (
                        min(secondary_max_size, max_size)
                        if max_size
                        else secondary_max_size
                    )
                if ceiling and max(target) > ceiling and not is_exempt:
                    scale = ceiling / float(max(target))
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
                            # RDO rides only when asked (Ktx2Encoder.rdo_kwargs:
                            # a registered encoder need not model it), and it
                            # is a UASTC-only stage.
                            rdo = (
                                cls._uastc_rdo_for(semantic, uastc_rdo)
                                if codec == "UASTC"
                                else None
                            )
                            encoder.encode(
                                pil,
                                out,
                                codec=codec,
                                srgb=srgb,
                                quality=quality if codec == "ETC1S" else None,
                                **Ktx2Encoder.rdo_kwargs(rdo, uastc_rdo_dictionary),
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
            # to workers would cost more than the encode saves. The count is
            # the shared encode policy (memory-capped: ImgUtils.encode_workers).
            from pythontk.img_utils._img_utils import ImgUtils

            count = max(1, min(ImgUtils.encode_workers(workers), len(jobs)))
            order = cls._largest_first(jobs)
            if count > 1:
                with ThreadPoolExecutor(
                    max_workers=count, thread_name_prefix="ptk-glb-optimize"
                ) as pool:
                    encoded_by_key = dict(zip(order, pool.map(_encode, order)))
            else:
                encoded_by_key = {key: _encode(key) for key in order}

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
