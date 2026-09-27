# !/usr/bin/python
# coding=utf-8
"""The GLB's image table: embedding, de-duplication and pruning.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through ``cls``.
"""

import base64
import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget

logger = logging.getLogger(__name__)


class _ImagesMixin:
    """Image embedding, de-duplication and pruning.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: Extensions that reference ``images`` from outside the material tree
    #: (root-level ``specularImages`` here). :meth:`prune_glb_unreferenced_textures`
    #: only follows material -> texture -> image, so a file declaring one of
    #: these is left alone rather than renumbered under it.
    _IMAGE_REFERRING_EXTENSIONS = frozenset({"EXT_lights_image_based"})
    # Image types glTF 2.0 accepts natively. Anything else (TIFF, EXR, TGA —
    # all common in a DCC source tree) is re-encoded to PNG via Pillow when
    # available (see `_reencode_as_png`), and otherwise rejected by name
    # rather than written as an unloadable data URI.
    IMAGE_MIME_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    #: Containers a GLB can carry its images in, as file-extension tokens:
    #: glTF core's own (:attr:`IMAGE_MIME_TYPES`) plus the two
    #: :meth:`optimize_glb_textures` declares an extension for -- WebP
    #: (``EXT_texture_webp``) and KTX2 (``KHR_texture_basisu``). A scene-side
    #: container (TGA, EXR, TIFF ...) is not one: asked of the web delivery
    #: policy, it means the policy's own container.
    GLB_IMAGE_FORMATS = frozenset(
        [ext.lstrip(".") for ext in IMAGE_MIME_TYPES] + ["webp", "ktx2"]
    )

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

    @staticmethod
    def _map_sidecar_image_refs(gltf: Dict[str, Any], visit) -> None:
        """Apply *visit* to every image index the embedded scene sidecar names.

        ``visit(index)`` returns where those bytes live now, or ``None`` when
        they left the file -- that reference is dropped, and the envelope's own
        ``validate`` count drops with it so a verifying reader still finds the
        envelope consistent with itself. Digests stay valid: a renumber moves
        bytes, never changes them. The one definition of where the sidecar
        names images, shared by the passes that renumber them
        (:meth:`dedupe_glb_images`, :meth:`prune_glb_unreferenced_textures`).
        """
        extras = gltf.get("extras")
        sidecar = extras.get("scene_sidecar") if isinstance(extras, dict) else None
        entries = sidecar.get("textures") if isinstance(sidecar, dict) else None
        if not isinstance(entries, dict):
            return
        dropped = 0
        for path in list(entries):
            ref = entries[path]
            index = ref.get("image") if isinstance(ref, dict) else None
            if not isinstance(index, int):
                continue
            moved = visit(index)
            if moved is None:
                del entries[path]
                dropped += 1
            else:
                ref["image"] = moved
        claims = sidecar.get("validate")
        if dropped and isinstance(claims, dict):
            if isinstance(claims.get("textures"), int):
                claims["textures"] -= dropped

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
        the same set `image_for_texture` resolves a binding through), as is
        every scene-sidecar reference to it; then the orphaned images and their
        exclusive bufferViews are reclaimed by
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
            # The sidecar names the same bytes by image index: its references
            # follow the survivor rather than leaving with the duplicate.
            cls._map_sidecar_image_refs(
                gltf, lambda index: replacement.get(index, index)
            )
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
        texture map is built, and again wherever a later applier unbinds an
        image -- so that map is renumbered here as well
        (:meth:`_map_sidecar_image_refs`): a reference follows its image, one
        whose image was dropped goes, and one that named no image is left for
        :meth:`verify_glb` to report. Safe to run standalone on any GLB; a file
        with nothing to drop is not rewritten.

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

            # --- and what extras.shadow_web binds by index --------------------
            # The shadow manifest names its maps by INDEX, and the horizon map
            # is sampled by no material -- so the walk above can see neither
            # fact, and without this the prune DELETES the horizon map and
            # renumbers the survivors out from under the manifest. In the
            # preview path the prune always fires (FBX2glTF's diffuse_cube /
            # ibl_brdf_lut sit at low indices), so every projected plane
            # shipped a stale index. extras.lightmap_web is immune only
            # because it stores names.
            gltf_extras = gltf.get("extras")
            shadow_raw = (
                gltf_extras.get(cls.SHADOW_WEB_KEY)
                if isinstance(gltf_extras, dict)
                else None
            )
            shadow_doc = shadow_raw
            if isinstance(shadow_raw, str):
                try:
                    shadow_doc = json.loads(shadow_raw)
                except ValueError:
                    shadow_doc = None
            shadow_slots = cls._shadow_web_texture_slots(shadow_doc)
            live_textures |= {
                h["texture_index"]
                for h in shadow_slots
                if 0 <= h["texture_index"] < len(textures)
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

            # --- drop the images, then collect what they were reading --------
            # Dropping them from the array first removes their bufferView
            # references, and the collector finds the orphans for itself: it
            # reads liveness off the whole file, so nothing here has to compute
            # a dead set, walk the survivors separately, or remap indices.
            gltf["images"] = [images[old] for old in image_map]
            images = gltf["images"]
            reclaimed = cls._compact_bin(edit)

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
            for holder in shadow_slots:
                if holder["texture_index"] in texture_map:
                    holder["texture_index"] = texture_map[holder["texture_index"]]
            if shadow_slots and isinstance(shadow_raw, str):
                # Carried as JSON text: the in-place rewrite above landed on the
                # decoded copy, so it has to be re-serialised to reach the file.
                gltf_extras[cls.SHADOW_WEB_KEY] = json.dumps(shadow_doc)
            # The scene sidecar's texture map names IMAGES by index too, and is
            # recorded before later appliers run -- the highlight pass unbinds
            # emissive maps and prunes after it -- so without this every later
            # reference named its neighbour (28 of 35 on a production GLB).
            before = len(image_map) + dropped_images
            cls._map_sidecar_image_refs(
                gltf,
                # One that named no image BEFORE this prune is not the prune's
                # to erase: left as found, verify_glb still reports it.
                lambda index: image_map.get(index) if 0 <= index < before else index,
            )
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
    def _embed_image(cls, edit: "GlbEdit", path: str) -> Optional[int]:
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
        edit: "GlbEdit",
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
