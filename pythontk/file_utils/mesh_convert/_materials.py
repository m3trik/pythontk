# !/usr/bin/python
# coding=utf-8
"""Materials: the post-conversion sanity checks and the per-channel writers.

Checks and repairs what the conversion made of the authored materials
(phantom opaque alpha, unvalidated ORM packing) and writes base colour,
metallic/roughness, emissive, alpha mode and normal scale by material name.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through ``cls``.
"""

import io
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pythontk.file_utils.mesh_convert.glb.edit import GlbTarget

logger = logging.getLogger(__name__)


class _MaterialsMixin:
    """Materials: sanity checks and the per-channel writers.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: zlib level for PNGs this module writes as an INTERMEDIATE -- the packed
    #: ORM the sidecar embeds -- which the texture pass then decodes and
    #: re-encodes for delivery. Deflate effort spent there is paid twice and
    #: kept nowhere: measured on a 4K map, level 6 took 35% longer than 1 to
    #: write and every production path re-encodes the result anyway.
    INTERMEDIATE_PNG_LEVEL = 1

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
                        image.save(
                            buffer,
                            format="PNG",
                            compress_level=cls.INTERMEDIATE_PNG_LEVEL,
                        )
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
                manifest = cls._lightmap_web_manifest(edit.gltf)
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
