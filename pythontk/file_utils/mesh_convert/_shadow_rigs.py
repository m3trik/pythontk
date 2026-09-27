# !/usr/bin/python
# coding=utf-8
"""Shadow rigs: bind the planes' maps and publish the viewer manifest.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through ``cls``.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Eager: the channel keys below are SceneRecords' own, read at class
# definition.
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget

logger = logging.getLogger(__name__)


class _ShadowRigsMixin:
    """Shadow rigs: plane maps and the ``extras.shadow_web`` manifest.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: ``data_export`` channel the DCC shadow rigs publish (mayatk / blendertk
    #: ``ShadowRig.SHADOW_METADATA``). Read here only -- unlike the lightmap
    #: markers it is never promoted to a top-level extras key.
    SHADOW_METADATA_KEY = SceneRecords.SHADOWS.key
    #: Highest ``shadow_metadata`` schema this applier knows how to read --
    #: the record's declared version, like the lightmap and visibility ones.
    SHADOW_METADATA_VERSION = SceneRecords.SHADOWS.version
    #: Root-extras key the viewer's packaged ``shadow_rig`` script reads.
    SHADOW_WEB_KEY = "shadow_web"
    #: Where ``shadow_web`` binds by glTF NODE INDEX -- the paths a pass that
    #: renumbers nodes (``strip_glb_curve_proxies``) must follow.
    SHADOW_WEB_NODE_FIELDS = (
        ("planes", "node"),
        ("planes", "source_node"),
        ("planes", "contact_node"),
    )
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
    #: What a record leaves unsaid is read as. A v1 channel carries only
    #: ``name`` / ``texture`` / ``intensity``: a projected plane with no
    #: source to follow, no contact, no atlas and no horizon map.
    #: ``max_stretch`` is ``ShadowProjection.DEFAULT_MAX_STRETCH``.
    SHADOW_PLANE_DEFAULTS: Dict[str, Any] = {
        "type": "projected",
        "intensity": 1.0,
        "source": None,
        "source_type": "point",
        "source_size": 0.0,
        "source_angle": 0.0,
        "follow_source": False,
        "contact": None,
        "ground": 0.0,
        "radius": 0.5,
        "height": 1.0,
        "max_stretch": 6.0,
        "canvas": [-1.0, 1.0, -0.5, 0.5],
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
    def _shadow_web_texture_slots(cls, manifest: Any) -> List[Dict[str, Any]]:
        """Every dict in an ``extras.shadow_web`` manifest holding a
        ``texture_index``: the plane's own colour map, its atlas and its
        horizon map.

        Yields the CONTAINERS rather than the values, so one walk serves both
        readers of this manifest: the optimiser asks which textures are bound
        and must be kept byte for byte, and the prune has to REWRITE those
        numbers when it renumbers the survivors. Accepts the dict or its JSON
        string; anything malformed yields nothing.
        """
        if isinstance(manifest, str):
            try:
                manifest = json.loads(manifest)
            except ValueError:
                return []
        slots: List[Dict[str, Any]] = []
        for plane in (
            (manifest or {}).get("planes") or [] if isinstance(manifest, dict) else []
        ):
            if not isinstance(plane, dict):
                continue
            atlas = plane.get("atlas") if isinstance(plane.get("atlas"), dict) else {}
            horizon = (
                plane.get("horizon") if isinstance(plane.get("horizon"), dict) else {}
            )
            for holder in (plane, atlas, horizon):
                value = holder.get("texture_index")
                if isinstance(value, int) and not isinstance(value, bool):
                    slots.append(holder)
        return slots

    @classmethod
    def _shadow_web_texture_indices(cls, manifest: Any) -> Set[int]:
        """Every texture index an ``extras.shadow_web`` manifest binds.

        The plane's own colour map, its atlas and its horizon map -- the set
        the texture optimiser keeps byte for byte. Accepts the dict or its
        JSON string; anything malformed yields nothing.
        """
        return {h["texture_index"] for h in cls._shadow_web_texture_slots(manifest)}

    @classmethod
    def apply_glb_shadows(
        cls, glb: GlbTarget, *, search_dirs: Sequence[str] = ()
    ) -> Optional[Dict[str, Any]]:
        """Bind a scene's shadow-rig maps into a GLB; publish ``extras.shadow_web``.

        The GLB half of the shadow-rig contract (``mayatk/docs/
        shadow_rig_morphing.md``, *Contracts*). The DCC rigs publish
        ``shadow_metadata`` on the ``data_export`` carrier -- per plane its
        node, source and contact names, the projection model's constants, the
        silhouette texture and, when in use, the atlas rect and the horizon
        map. This reads that channel back out of the file (like the lightmap
        pass), resolves every name to a glTF node index, binds the loose maps
        the FBX does not carry (the horizon PNG, embedded byte for byte under
        a bilinear / no-mip / clamp sampler; a silhouette the material lost)
        and writes the viewer's manifest: the v2 record plus, per plane,
        ``node`` / ``source_node`` / ``contact_node`` (glTF node indices, or
        ``None``), ``material`` (the first primitive's), ``texture_index``
        (the plane's colour map -- the material's base colour, which IS the
        atlas once packed) and ``atlas.texture_index`` /
        ``horizon.texture_index``. Rects in the manifest are glTF top-left
        (:meth:`ImgUtils.flip_rect_v` of the record's bottom-left ones);
        lengths stay DCC units with ``unit_scale`` beside them, as the record
        has them; ``metadata_version`` says which schema the record arrived in
        (a v1 record -- name, texture, intensity -- is filled from
        :attr:`SHADOW_PLANE_DEFAULTS`). The packaged
        ``preview/scripts/shadow_rig.js`` reads the manifest on load,
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
            payload = cls.data_export_channel(gltf, cls.SHADOW_METADATA_KEY)
            if not isinstance(payload, dict):
                return None
            try:
                version = int(payload.get("version", 1))
            except (TypeError, ValueError):
                version = -1
            if not 0 < version <= cls.SHADOW_METADATA_VERSION:
                # Refuse rather than guess, as the lightmap reader does: a
                # newer schema is free to change what a field MEANS.
                logger.warning(
                    "shadow_metadata v%r is newer than this reader (v%s); "
                    "shadow rigs not wired.",
                    payload.get("version"),
                    cls.SHADOW_METADATA_VERSION,
                )
                return None
            try:
                unit_scale = float(payload.get("unit_scale", 1.0))
            except (TypeError, ValueError):
                unit_scale = 0.0
            if not unit_scale > 0.0:
                logger.warning(
                    "shadow_metadata: unit_scale %r is not a positive number; "
                    "read as 1.0.",
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
                for key, value in cls.SHADOW_PLANE_DEFAULTS.items():
                    plane.setdefault(key, copy.deepcopy(value))
                name, kind = plane.get("name"), plane.get("type")
                if not isinstance(name, str) or not name:
                    logger.warning(
                        "shadow_metadata: a plane record has no name; skipped."
                    )
                    skipped += 1
                    continue
                if kind not in ("projected", "horizon"):
                    logger.warning(
                        "Shadow plane %r: unknown type %r; skipped.", name, kind
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
                # whose atlas is nowhere, keeps the old chain below.
                atlas_texture = None
                if atlas:
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
                    material_index, base_texture = cls._shadow_plane_material(
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
                    entry = copy.deepcopy(plane)
                    entry.update(
                        {
                            "node": node_index,
                            "source_node": source_node,
                            "contact_node": contact_node,
                            "material": material_index,
                            "texture_index": texture_index,
                        }
                    )
                    if atlas:
                        packed = dict(atlas)
                        # Whatever the plane ends up sampling: with the atlas
                        # unresolvable, the material's own map IS the atlas
                        # (the DCC points its file node at it), and the rect
                        # still describes the tile inside it -- the viewer
                        # applies the rect only when these two agree.
                        packed["texture_index"] = (
                            atlas_texture
                            if atlas_texture is not None
                            else texture_index
                        )
                        packed["rect"] = _rect(atlas.get("rect"))
                        entry["atlas"] = packed
                    if horizon:
                        block = dict(horizon)
                        block["texture_index"] = horizon_texture
                        block["rect"] = _rect(horizon.get("rect"))
                        entry["horizon"] = block
                    planes.append(entry)

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
                manifest = {
                    "version": 2,
                    "metadata_version": version,
                    "unit_scale": unit_scale,
                    "planes": planes,
                }
                extras[cls.SHADOW_WEB_KEY] = manifest
                edit.dirty = True
                return manifest
            if extras.pop(cls.SHADOW_WEB_KEY, None) is not None:
                edit.dirty = True
            if skipped:
                logger.warning(
                    "Shadow rigs NOT wired: %d plane(s) in this GLB, none bound.",
                    skipped,
                )
            return None
