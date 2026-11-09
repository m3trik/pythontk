# !/usr/bin/python
# coding=utf-8
"""Materials: the post-conversion sanity checks and the normal-scale writer.

Checks and repairs what the conversion made of the authored materials
(phantom opaque alpha, unvalidated ORM packing), writes normal scale, and
pairs a record's entries with the GLB materials they name (the materials
record's applier, ``_fbx_materials.py``, writes each one).

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through the facade it holds (``self._mc``).
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pythontk.file_utils.mesh_convert.glb.edit import GlbTarget

logger = logging.getLogger(__name__)


class _Materials:
    """Materials: sanity checks, the normal-scale writer and the by-name match.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: zlib level for PNGs this package writes as an INTERMEDIATE -- the packed
    #: ORM a materials record embeds -- which the texture pass then decodes and
    #: re-encodes for delivery. Deflate effort spent there is paid twice and
    #: kept nowhere: measured on a 4K map, level 6 took 35% longer than 1 to
    #: write and every production path re-encodes the result anyway.
    INTERMEDIATE_PNG_LEVEL = 1

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    def check_glb_materials(self, glb: GlbTarget) -> List[Dict[str, str]]:
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
        with self._mc.open_glb(glb) as edit:
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

    def fix_glb_phantom_opaque_alpha(self, glb: GlbTarget) -> List[Dict]:
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
        with self._mc.open_glb(glb) as edit:
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

    def suspect_orm_materials(
        self, glb: GlbTarget, *, described: Optional[Iterable[str]] = None
    ) -> Dict[str, Dict[str, str]]:
        known = set(described) if described is not None else None
        findings: Dict[str, Dict[str, str]] = {}
        with self._mc.open_glb(glb) as edit:
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
                # Metal everywhere is a full blue channel at a factor of 1 or
                # more: below it the factor is the metal (a constant over a
                # roughness map packs blue white, its fill), and a zero factor
                # cancels the texture -- still unvalidated data, never harmful.
                factor = pbr.get("metallicFactor", 1.0)
                harmful = (
                    isinstance(factor, (int, float))
                    and factor >= 1.0
                    and edit.channel_extrema(img_idx, self._mc._ORM_HARMFUL_CHANNEL)
                    == (255, 255)
                )
                if harmful:
                    finding = self._mc.ORM_FINDING_METALLIC_FULL
                elif known is not None:
                    finding = self._mc.ORM_FINDING_UNVALIDATED
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
        """Pair a record's *entries* with the GLB materials they name.

        The front half of the by-name writer (``apply_glb_materials``): resolve
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

    #: glTF's unlit shading model (three.js: ``MeshBasicMaterial``).
    UNLIT_EXTENSION = "KHR_materials_unlit"

    def set_glb_normal_scale(
        self, glb: GlbTarget, scale: float, lightmapped_only: bool = True
    ) -> int:
        scale = float(scale)
        changed = 0
        with self._mc.open_glb(glb) as edit:
            worn = None
            if lightmapped_only:
                # The materials the nodes the record lights wear (``lit_nodes``):
                # a material those nodes share with unbaked ones is written for
                # all of its wearers, as its occlusion is kept for all of them
                # (``MaterialFamilies.without_baked_occlusion``).
                meshes = edit.gltf.get("meshes") or []
                nodes = edit.gltf.get("nodes") or []
                worn = {
                    prim.get("material")
                    for index in self._mc.lit_nodes(edit.gltf)
                    for prim in meshes[nodes[index]["mesh"]].get("primitives") or []
                }
            for index, material in enumerate(edit.materials):
                if worn is not None and index not in worn:
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
