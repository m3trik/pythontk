# !/usr/bin/python
# coding=utf-8
"""Shadow rigs: bind the planes' maps and publish the viewer manifest.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through ``cls``.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Eager: the signatures name GlbTarget; SceneRecords is the light
# declaration module every pass reads.
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget

logger = logging.getLogger(__name__)


class _ShadowRigsMixin:
    """Shadow rigs: plane maps and the ``extras.shadows`` manifest.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: Sampler a horizon DATA map is bound with: no mipmaps (9729 = LINEAR
    #: for both filters -- a mip would average span heights, distances and
    #: pyramid bounds across texels; the shader reads texels whole, so the
    #: filter itself never applies) and clamped (33071 = CLAMP_TO_EDGE).
    SHADOW_DATA_SAMPLER = {
        "magFilter": 9729,
        "minFilter": 9729,
        "wrapS": 33071,
        "wrapT": 33071,
    }

    @staticmethod
    def _nodes_named(gltf: dict, name: Any, mesh_only: bool = False) -> List[int]:
        """Indices of the nodes *name* denotes: exact matches, else every node
        whose namespace-stripped leaf matches (``NS:name`` binds ``name``).

        Every leaf match is returned rather than the first: the same rig can
        sit under two namespaces in one file, and both planes are real.
        """
        if not isinstance(name, str) or not name:
            return []
        nodes = gltf.get("nodes") or []

        def _eligible(node: dict) -> bool:
            return not mesh_only or "mesh" in node

        exact = [
            i for i, n in enumerate(nodes) if n.get("name") == name and _eligible(n)
        ]
        if exact:
            return exact
        leaf = name.rsplit(":", 1)[-1]
        return [
            i
            for i, n in enumerate(nodes)
            if str(n.get("name") or "").rsplit(":", 1)[-1] == leaf and _eligible(n)
        ]

    @classmethod
    def _shadow_single_node(
        cls, gltf: dict, name: Any, plane: str, role: str
    ) -> Optional[int]:
        """The one node *name* denotes, or ``None`` -- absent (logged at debug:
        a selection export routinely leaves the source behind) or ambiguous
        (warned: a plane must never follow a guess)."""
        if not isinstance(name, str) or not name:
            return None
        found = cls._nodes_named(gltf, name)
        if len(found) == 1:
            return found[0]
        if found:
            logger.warning(
                "Shadow plane %r: its %s %r matches several GLB nodes (%s) -- "
                "ambiguous, the plane keeps its imported keys.",
                plane,
                role,
                name,
                ", ".join(str(i) for i in found),
            )
        else:
            logger.debug(
                "Shadow plane %r: no GLB node named %r for its %s.", plane, name, role
            )
        return None

    @staticmethod
    def _shadow_plane_material(
        gltf: dict, node: dict
    ) -> Tuple[Optional[int], Optional[int]]:
        """``(material index, base-colour texture index)`` of a plane node's
        first primitive -- either ``None`` when the file has no such thing."""
        meshes = gltf.get("meshes") or []
        materials = gltf.get("materials") or []
        textures = gltf.get("textures") or []
        mesh_index = node.get("mesh")
        if not isinstance(mesh_index, int) or not 0 <= mesh_index < len(meshes):
            return None, None
        primitives = meshes[mesh_index].get("primitives") or []
        material = primitives[0].get("material") if primitives else None
        if not isinstance(material, int) or not 0 <= material < len(materials):
            return None, None
        pbr = materials[material].get("pbrMetallicRoughness") or {}
        index = (pbr.get("baseColorTexture") or {}).get("index")
        if not isinstance(index, int) or not 0 <= index < len(textures):
            index = None
        return material, index

    @classmethod
    def _bind_shadow_texture(
        cls, edit: "GlbEdit", src: str, raw: bytes, data: bool
    ) -> int:
        """Embed the already-encoded bytes of *src*; return a texture index.

        The bytes go in as they are -- never decoded, never re-encoded: a
        horizon map is DATA (span heights, a distance field and a pyramid
        packed into its channels) and a silhouette was sized by the rasterizer. *data* binds
        through :attr:`SHADOW_DATA_SAMPLER`; a colour map takes the clamp
        sampler the atlas precedent uses. A texture this call appended is
        simply retargeted (nothing else samples through it yet); a texture
        the embed deduped onto -- the file already held these bytes -- is
        left as it is and a data texture over the same image is found or
        added beside it, so a material sharing that image keeps its sampler.
        """
        mime = "image/jpeg" if src.lower().endswith((".jpg", ".jpeg")) else "image/png"
        name = os.path.basename(src)
        textures = edit.gltf.setdefault("textures", [])
        before = len(textures)
        index = cls._embed_image_bytes(edit, src, raw, mime=mime, name=name, clamp=True)
        if not data:
            return index
        samplers = edit.gltf.setdefault("samplers", [])
        wanted = dict(cls.SHADOW_DATA_SAMPLER)
        sampler = next((i for i, s in enumerate(samplers) if s == wanted), None)
        if sampler is None:
            samplers.append(wanted)
            sampler = len(samplers) - 1
        if index >= before:
            textures[index]["sampler"] = sampler
            return index
        image = textures[index].get("source")
        for i, texture in enumerate(textures):
            if texture.get("source") == image and texture.get("sampler") == sampler:
                return i
        textures.append(
            {"source": image, "sampler": sampler, "name": os.path.splitext(name)[0]}
        )
        return len(textures) - 1

    @staticmethod
    def _texture_for_named_image(edit: "GlbEdit", basename: Any) -> Optional[int]:
        """Index of a texture sampling the image *basename* names, or ``None``.

        The atlas a record names is normally ALREADY in the GLB -- the FBX
        embedded it, because the DCC's plane material points its file node at
        it -- so it is found here before any directory is searched. Matched on
        the image's glTF name, then on its ``uri``'s basename, case-folded
        (the converter names an embedded image by its source file). An image
        no texture samples gets one appended, inheriting the file's own
        sampler for it where there is one.
        """
        if not isinstance(basename, str) or not basename:
            return None
        wanted = os.path.basename(basename).casefold()
        stem = os.path.splitext(wanted)[0]
        for index, image in enumerate(edit.images):
            name = str(image.get("name") or "")
            uri = os.path.basename(str(image.get("uri") or "").split("?", 1)[0])
            candidates = {name.casefold(), os.path.splitext(name)[0].casefold()}
            if uri:
                candidates.add(uri.casefold())
            if wanted in candidates or stem in candidates:
                return edit.texture_for_image(index)
        return None

    @classmethod
    def _shadow_texture_slots(cls, block: Any) -> List[Dict[str, Any]]:
        """Every dict in an ``extras.shadows`` block holding a
        ``texture_index``: a projected plane's colour map (its atlas once
        packed) and a horizon plane's map.

        Yields the CONTAINERS rather than the values, so one walk serves both
        readers of this block: the optimiser asks which textures are bound
        and must be kept byte for byte, and the prune has to REWRITE those
        numbers when it renumbers the survivors. Anything but a block yields
        nothing.
        """
        slots: List[Dict[str, Any]] = []
        for plane in block.get("planes") or [] if isinstance(block, dict) else []:
            if not isinstance(plane, dict):
                continue
            horizon = (
                plane.get("horizon") if isinstance(plane.get("horizon"), dict) else {}
            )
            for holder in (plane, horizon):
                value = holder.get("texture_index")
                if isinstance(value, int) and not isinstance(value, bool):
                    slots.append(holder)
        return slots

    @classmethod
    def _shadow_texture_indices(cls, block: Any) -> Set[int]:
        """Every texture index an ``extras.shadows`` block binds: a projected
        plane's colour map and a horizon plane's map -- the set the texture
        optimiser keeps byte for byte."""
        return {h["texture_index"] for h in cls._shadow_texture_slots(block)}

    @classmethod
    def apply_glb_shadows(
        cls, glb: GlbTarget, *, search_dirs: Sequence[str] = ()
    ) -> Optional[Dict[str, Any]]:
        """Bind a scene's shadow-rig maps into a GLB; publish ``extras.shadows``.

        The GLB half of the shadow-rig contract (``mayatk/docs/
        shadow_rig_morphing.md``, *Contracts*). The DCC rigs publish
        ``shadows`` on the ``data_export`` carrier -- per plane its
        node, source and contact names, the projection model's constants, the
        silhouette texture and, when in use, the atlas rect and the horizon
        map. This reads that channel back out of the file (like the lightmap
        pass), resolves every name to a glTF node index, binds the loose maps
        the FBX does not carry (the horizon PNG, embedded byte for byte under
        a bilinear / no-mip / clamp sampler; a silhouette the material lost)
        and writes the viewer's manifest (``ShadowBlock``): per plane the
        record's projection model, bound by index in place of every name --
        ``node`` / ``source_node`` / ``contact_node`` (glTF node indices, or
        ``None``), a projected plane's ``texture_index`` (its colour map --
        the material's base colour, which IS the atlas once packed, its tile
        the quad's own UVs) and a horizon plane's map with its
        ``texture_index``. The horizon rect is glTF top-left
        (:meth:`ImgUtils.flip_rect_v` of the record's bottom-left one);
        lengths stay DCC units with ``unit_scale`` beside them, as the record
        has them; every plane states every field its shape declares
        (``pythontk.geo_utils.shadow_record``) and nothing more. The
        packaged viewer script
        (``preview/features/shadow_rig/``) reads the manifest on load,
        evaluates ``ShadowProjection.model`` per frame and drives the planes.

        Names resolve exactly first, then by namespace-stripped leaf against
        EVERY node carrying it: a plane name matching several nodes gets one
        manifest entry per node (a rig can sit under two namespaces in one
        file), while an ambiguous source or contact leaves the plane
        unfollowed rather than following a guess. A plane with no node in the
        file is out of scope (a selection export) and counted once; one whose
        horizon map cannot be found in *search_dirs* -- or a projected plane
        with no colour map at all -- is skipped with a warning and keeps its
        DCC material, which is the silhouette fallback.

        Idempotent: the manifest is replaced wholesale on every run, an
        embedded map is found again by content and reused, and the channel
        itself is left in place. A run that binds nothing removes any stale
        manifest.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.
            search_dirs: Directories to resolve the records' texture basenames
                against, in order; the GLB's own directory is tried last.

        Returns:
            The published manifest, or ``None`` when the file carries no
            channel or nothing in it could be bound.
        """
        import copy

        from pythontk.geo_utils.shadow_record import ShadowBlockPlane
        from pythontk.img_utils._img_utils import ImgUtils

        identity = [1.0, 1.0, 0.0, 0.0]

        def _rect(value: Any) -> List[float]:
            try:
                rect = [float(v) for v in value]
            except (TypeError, ValueError):
                rect = []
            return ImgUtils.flip_rect_v(rect if len(rect) == 4 else identity)

        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            payload = cls._current_record(gltf, SceneRecords.SHADOWS)
            if payload is None:
                return None
            try:
                unit_scale = float(payload.get("unit_scale", 1.0))
            except (TypeError, ValueError):
                unit_scale = 0.0
            if not unit_scale > 0.0:
                logger.warning(
                    "%s: unit_scale %r is not a positive number; read as 1.0.",
                    SceneRecords.SHADOWS.key,
                    payload.get("unit_scale"),
                )
                unit_scale = 1.0
            records = [r for r in (payload.get("planes") or []) if isinstance(r, dict)]

            dirs = [d for d in search_dirs if d]
            dirs.append(os.path.dirname(os.path.abspath(edit.path)))
            #: (basename, data) -> texture index, or None once warned missing.
            bound: Dict[Tuple[str, bool], Optional[int]] = {}

            def _bind(basename: Any, data: bool) -> Optional[int]:
                """Texture index for the loose file *basename*, embedded on
                first use; a file that is nowhere is warned once."""
                if not isinstance(basename, str) or not basename:
                    return None
                key = (basename, data)
                if key in bound:
                    return bound[key]
                src = next(
                    (
                        p
                        for d in dirs
                        for p in [os.path.join(d, basename)]
                        if os.path.isfile(p)
                    ),
                    None,
                )
                if src is None:
                    logger.warning(
                        "Shadow map %r not found in %s -- the plane(s) using it "
                        "keep their DCC material.",
                        basename,
                        dirs,
                    )
                    bound[key] = None
                    return None
                with open(src, "rb") as fh:
                    raw = fh.read()
                bound[key] = cls._bind_shadow_texture(
                    edit, os.path.abspath(src), raw, data=data
                )
                return bound[key]

            nodes = gltf.get("nodes") or []
            planes: List[Dict[str, Any]] = []
            out_of_scope: List[str] = []
            skipped = 0
            for record in records:
                plane = copy.deepcopy(record)
                name, kind = plane.get("name"), plane.get("type")
                if not isinstance(name, str) or not name:
                    logger.warning(
                        "%s: a plane record has no name; skipped.",
                        SceneRecords.SHADOWS.key,
                    )
                    skipped += 1
                    continue
                node_indices = cls._nodes_named(gltf, name, mesh_only=True)
                if not node_indices:
                    out_of_scope.append(name)
                    logger.debug(
                        "Shadow plane %r: no mesh node by that name in the GLB.", name
                    )
                    continue
                source_node = cls._shadow_single_node(
                    gltf, plane.get("source"), name, "source"
                )
                contact_node = cls._shadow_single_node(
                    gltf, plane.get("contact"), name, "contact"
                )
                atlas = (
                    plane.get("atlas") if isinstance(plane.get("atlas"), dict) else None
                )
                horizon = (
                    plane.get("horizon")
                    if isinstance(plane.get("horizon"), dict)
                    else None
                )
                horizon_texture = (
                    _bind(horizon.get("texture"), data=True) if horizon else None
                )
                # The ATLAS is the plane's colour map once the rig is packed:
                # its rect names a tile OF THAT IMAGE, so a manifest pointing
                # at the plane's own full-frame PNG instead makes the rect
                # meaningless (and, in the viewer, unbatchable -- planes batch
                # by the texture they sample). Resolved from the file first,
                # then from the search dirs; a record with no atlas, or one
                # whose atlas is nowhere, keeps the old chain below. A horizon
                # plane samples no colour map (the march shades it), so none
                # is looked up -- or embedded -- for it.
                atlas_texture = None
                if atlas and kind == "projected":
                    atlas_texture = cls._texture_for_named_image(
                        edit, atlas.get("texture")
                    )
                    if atlas_texture is None:
                        atlas_texture = _bind(atlas.get("texture"), data=False)
                if kind == "horizon" and horizon_texture is None:
                    logger.warning(
                        "Shadow plane %r: its horizon map is not bound; skipped "
                        "(the silhouette fallback stays).",
                        name,
                    )
                    skipped += 1
                    continue
                for node_index in node_indices:
                    texture_index = None
                    if kind == "projected":
                        _material, base_texture = cls._shadow_plane_material(
                            gltf, nodes[node_index]
                        )
                        texture_index = atlas_texture
                        if texture_index is None:
                            texture_index = (
                                base_texture
                                if base_texture is not None
                                else _bind(plane.get("texture"), data=False)
                            )
                    if kind == "projected" and texture_index is None:
                        logger.warning(
                            "Shadow plane %r (node %d): no colour map -- neither the "
                            "material's base colour nor %r in %s; skipped.",
                            name,
                            node_index,
                            plane.get("texture"),
                            dirs,
                        )
                        skipped += 1
                        continue
                    # The block's plane: the record's, bound by index -- the
                    # names the indices replace, and the atlas tile the quad's
                    # own UVs already address, pruned by the block's shape.
                    entry = copy.deepcopy(plane)
                    entry.update(
                        {
                            "node": node_index,
                            "source_node": source_node,
                            "contact_node": contact_node,
                            "texture_index": texture_index,
                        }
                    )
                    if horizon:
                        entry["horizon"] = {
                            **horizon,
                            "rect": _rect(horizon.get("rect")),
                            "texture_index": horizon_texture,
                        }
                    planes.append(ShadowBlockPlane.prune(entry))

            extras = gltf.get("extras")
            if not isinstance(extras, dict):
                extras = gltf["extras"] = {}
            if out_of_scope:
                logger.info(
                    "Shadow metadata covers %d plane(s) not in this export "
                    "(scene-wide channel, exported subset); ignored.",
                    len(out_of_scope),
                )
            if planes:
                edit.dirty = True
                return SceneRecords.put_glb_block(
                    gltf,
                    SceneRecords.SHADOWS,
                    {"unit_scale": unit_scale, "planes": planes},
                )
            if extras.pop(SceneRecords.SHADOWS.key, None) is not None:
                edit.dirty = True
            if skipped:
                logger.warning(
                    "Shadow rigs NOT wired: %d plane(s) in this GLB, none bound.",
                    skipped,
                )
            return None
