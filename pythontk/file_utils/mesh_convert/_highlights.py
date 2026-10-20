# !/usr/bin/python
# coding=utf-8
"""The bake's lights in its reflection probe: the highlights a lightmap cannot hold.

A lightmap is the diffuse light the bake's lights left on a surface -- it has
no direction, so it holds no highlight. The reflection probe holds direction,
but it sees the room from one point and the viewer reads it box-projected: a
reflection ray is followed to where it leaves the room's box, and the probe is
read toward that point. A light ON the box (a ceiling panel) reflects where it
hangs. A light hanging over a bench is read as if it hung on the wall behind
it, so its highlight lands nowhere near where the light puts it -- why a moody
rig of bench lights left its metals dark while a bright office's ceiling
panels lit them.

So the probe ships COPIES, each for a group of receivers, with the bake's
lights drawn in where box projection shows them true for that group. For a
light and a focal point F: a probe texel whose ray leaves the box at B shows
what F sees of the light along F -> B. A reflection from F that reads B sees
the light exactly -- direction, size and radiance -- and a surface near F sees
it nearly so. F is chosen per light and per group, and each light is scaled so
the light it gives the group's own surfaces is the light's own:

* the room's shell (surfaces on the probe's box: walls, floor, ceiling) keeps
  the probe as captured -- lights placed for the bench and read from a wall
  glowed where no light shines;
* an object the bake left out (no lightmap, so everything it shows comes from
  the probe) gets a copy of its own: F at the centroid of each light's
  irradiance on it, the irradiance matched over it;
* the rest of the room's interior shares one: F at each light's highlight hot
  spot as the eye sees it, the highlight matched -- a baked surface shows a
  light only through its highlight, its diffuse being the bake's.

The eye is the deliverable's start camera (:attr:`HIGHLIGHT_EYE_NODE`), else
the probe's capture point, which the bake places where the room is seen from.
Measured on a production soldering table (2026-10-19), against the same 15
lights lit live in the viewer: the magnifier (unbaked), the puck, trays and
button box read as the live render did, from three copies at no shader cost --
each material still reads one probe -- where the live lights cost the frame a
fifth.
"""
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pythontk.core_utils.color import Color
from pythontk.math_utils._math_utils import MathUtils

logger = logging.getLogger(__name__)

_LUMA = np.asarray(Color.REC709_LUMINANCE)


class _HighlightsMixin:
    """The bake's lights drawn into per-receiver copies of its reflection probe."""

    #: The GGX roughness a light's highlight is placed and matched with.
    HIGHLIGHT_ROUGHNESS: float = 0.5
    #: The most copies a probe ships (each one is a prefiltered map in GPU memory).
    HIGHLIGHT_MAX_COPIES: int = 12
    #: Surface samples per square metre the placement and the scales read --
    #: and at least this many per group, however small: a puck's 0.06 m^2 at
    #: the density alone was 25 points, and its scale moved with the draw.
    HIGHLIGHT_SAMPLES_PER_M2: float = 400.0
    HIGHLIGHT_MIN_SAMPLES: int = 4000
    #: ... and at most this many: past it a group's focal points and scales
    #: hold still (measured: 50k and 1M samples agree within 5 mm and 0.1%),
    #: while a square rect light's field costs ~30 s per million.
    _HIGHLIGHT_MAX_SAMPLES: int = 50000
    #: The smallest a light is drawn, in degrees as its focal point sees it --
    #: a point light is a delta, which no texel can hold.
    HIGHLIGHT_MIN_ANGLE: float = 0.6
    #: Rays per texel side a light is drawn with.
    HIGHLIGHT_SUPERSAMPLE: int = 3
    #: The node a view starts at -- the eye the highlights are placed for.
    HIGHLIGHT_EYE_NODE: str = "user_pos"
    _HIGHLIGHT_SCALE_RANGE: Tuple[float, float] = (0.02, 4.0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @classmethod
    def highlight_probe_layers(
        cls,
        glb,
        lights: Optional[Sequence[Dict[str, Any]]],
        probe: Optional[Dict[str, Any]] = None,
        width: Optional[int] = None,
        baked: Optional[Sequence[str]] = None,
        eye: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        """The probe's highlight copies: for each receiver group, the lights it is shown.

        Each layer is what its group's copy adds to the probe as captured
        (radiance in the probe's own units, the three.js equirect layout --
        top row +Y): the bake's lights, each drawn where box projection shows
        it true for that group and scaled so the light it gives the group's
        surfaces is the light's own (see the module).

        Parameters:
            glb: ``.glb`` path or an open :class:`GlbEdit`: the geometry the
                groups are read from (model space, metres), and -- unless given
                -- its ``lightmap_web`` manifest's probe and baked materials.
            lights: The bake's lights in metres (:meth:`lights_in_metres`,
                applied here too: a light it leaves out is left out).
            probe: ``{"position", "box"}`` (metres; ``box`` ``{min, max}`` or
                ``None`` for a probe read as distant); default the file's.
            width: The probe's width in texels (the interior's copy is drawn
                at it, an unbaked object's at half); default 1024.
            baked: Material names that carry a lightmap; default the file's.
            eye: Where the room is seen from; default the start camera's node
                (:attr:`HIGHLIGHT_EYE_NODE`), else the probe's position.

        Returns:
            ``[{"materials", "receivers", "image"}]`` -- ``receivers``
            ``"interior"`` or ``"unbaked"``; ``image`` an
            ``(h, w, 3)`` float32 array. Empty when there is nothing to light.
        """
        # The one light validator (idempotent on its own output): a light
        # written wrong is left out, never the pass.
        lights = cls.lights_in_metres(lights)
        if not lights:
            return []
        with cls.open_glb(glb) as edit:
            web = cls._lightmap_web_manifest(edit.gltf) or {}
            probe = probe if probe is not None else web.get("probe")
            if not isinstance(probe, dict) or not probe.get("position"):
                return []
            baked_names = set(
                baked if baked is not None else (web.get("materials") or {})
            )
            meshes = cls._highlight_meshes(edit)
            if eye is None:
                eye = cls._highlight_eye(edit)
        C = np.asarray(probe["position"], float)
        box = probe.get("box")
        lo = np.asarray(box["min"], float) if box else None
        hi = np.asarray(box["max"], float) if box else None
        eye = np.asarray(eye if eye is not None else C, float)
        width = int(width or 1024) // 2 * 2
        groups = cls._highlight_groups(meshes, baked_names, lo, hi)
        layers = []
        for group in groups:
            pts, nrm = group["points"], group["normals"]
            unbaked = group["receivers"] == "unbaked"
            focal, scales = [], []
            for light in lights:
                F, s = cls._highlight_placement(
                    light, pts, nrm, eye, lo, hi, irradiance=unbaked
                )
                focal.append(F)
                scales.append(s)
            w = width if group["receivers"] == "interior" else max(64, width // 2)
            image = cls._highlight_layer(lights, focal, scales, C, lo, hi, w)
            if not np.any(image > 0):
                continue
            layers.append(
                {
                    "materials": sorted(group["materials"]),
                    "receivers": group["receivers"],
                    "image": image,
                }
            )
        return layers

    @classmethod
    def lights_in_metres(
        cls, lights: Optional[Sequence[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """The bake's lights in metres and the viewer's units, or ``None``.

        The lightmap record's ``lights`` (``SceneRecords.LIGHTMAPS``; a host's
        ``specular_lights``, e.g. mayatk ``LightUtils``) arrive in scene units
        with ``unit_scale`` (metres per unit) beside each, as the probe does.
        This carries metres: a point or spot light's intensity in candela (its
        irradiance falls over metres, so it scales by ``unit_scale`` squared),
        a rect's radiance as it is. One without a position, a type or a
        positive intensity is left out -- a field that is no number leaves its
        light out, never the bake that reads them.
        """
        if not isinstance(lights, (list, tuple)):
            return None
        out: List[Dict[str, Any]] = []
        for light in lights:
            if not isinstance(light, dict):
                continue
            kind = light.get("type")
            position = light.get("position")
            try:
                unit = float(light.get("unit_scale", 1.0))
                intensity = float(light.get("intensity", 0.0))
                if (
                    kind not in ("point", "spot", "rect")
                    or not (isinstance(position, (list, tuple)) and len(position) == 3)
                    or unit <= 0.0
                    or intensity <= 0.0
                ):
                    continue
                color = light.get("color") or (1.0, 1.0, 1.0)
                entry: Dict[str, Any] = {
                    "name": str(light.get("name", "")),
                    "type": kind,
                    "position": [round(float(c) * unit, 6) for c in position],
                    "color": [float(c) for c in color][:3],
                    "intensity": (
                        intensity * unit * unit if kind != "rect" else intensity
                    ),
                }
                for key in ("direction", "right"):
                    value = light.get(key)
                    if isinstance(value, (list, tuple)) and len(value) == 3:
                        entry[key] = [float(c) for c in value]
                for key in ("angle", "penumbra"):
                    if light.get(key) is not None:
                        entry[key] = float(light[key])
                for key in ("width", "height", "radius"):
                    if light.get(key) is not None:
                        entry[key] = round(float(light[key]) * unit, 6)
            except (TypeError, ValueError):
                continue
            out.append(entry)
        return out or None

    # ------------------------------------------------------------------
    # Embedding (the lightmap pass's)
    # ------------------------------------------------------------------

    @classmethod
    def _embed_highlight_probes(
        cls,
        edit,
        probe: Dict[str, Any],
        baked: Sequence[str],
        lights: Optional[Sequence[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Embed the probe's highlight copies; the probe entry's ``highlights``, or ``None``.

        *probe* is the embedded probe's ``lightmap_web`` entry (its
        ``bufferView`` is read for the probe's width); *lights* the record's
        (:meth:`lights_in_metres` is applied here). A layer that cannot be
        encoded is warned and left out: its receivers read the probe as
        captured, as a file without highlights does.
        """
        lights = cls.lights_in_metres(lights)
        if not lights:
            return None
        from pythontk.img_utils._img_utils import ImgUtils

        width = cls._radiance_width(edit, probe.get("bufferView")) or 1024
        eye = cls._highlight_eye(edit) or list(probe["position"])
        try:
            layers = cls.highlight_probe_layers(
                edit, lights, probe=probe, width=width, baked=baked, eye=eye
            )
        except Exception as error:  # noqa: BLE001 -- never the bake's cost
            logger.warning("Probe highlights not drawn: %s", error)
            return None
        entries = []
        for layer in layers:
            try:
                raw = ImgUtils.encode_hdr_radiance_pixels(layer["image"])
            except (ImportError, ValueError) as error:
                logger.warning("Probe highlight layer not encoded: %s", error)
                continue
            views = cls._append_bin_views(edit, [raw])
            if not views:
                continue
            height, w = layer["image"].shape[:2]
            entries.append(
                {
                    "bufferView": views[0],
                    "mimeType": "image/vnd.radiance",
                    "width": int(w),
                    "height": int(height),
                    "receivers": layer["receivers"],
                    "materials": layer["materials"],
                }
            )
        if not entries:
            return None
        logger.info(
            "Probe highlights: %d light(s) drawn into %d probe copy(ies) (%s).",
            len(lights),
            len(entries),
            ", ".join(f"{e['receivers']}: {len(e['materials'])}" for e in entries),
        )
        return {
            "eye": [round(float(c), 6) for c in eye],
            "lights": len(lights),
            "copies": entries,
        }

    @staticmethod
    def _radiance_width(edit, view: Optional[int]) -> Optional[int]:
        """The width a Radiance image in bufferView *view* declares (``+X w``), or ``None``."""
        views = edit.gltf.get("bufferViews") or []
        blob = edit.bin_data
        if not isinstance(view, int) or blob is None or not 0 <= view < len(views):
            return None
        entry = views[view]
        start = int(entry.get("byteOffset", 0))
        head = bytes(blob[start : start + min(int(entry["byteLength"]), 512)])
        for line in head.split(b"\n"):
            parts = line.split()
            if len(parts) == 4 and parts[2] in (b"+X", b"-X"):
                try:
                    return int(parts[3])
                except ValueError:
                    return None
        return None

    # ------------------------------------------------------------------
    # The FBX deliverable's copies
    # ------------------------------------------------------------------

    #: The name a highlight copy rides inside an FBX under: the probe's stem
    #: and the copy's place in the record, from 1.
    FBX_HIGHLIGHT_NAME: str = "{stem}_Highlight{index}.hdr"

    @classmethod
    def embed_fbx_highlight_probes(
        cls,
        fbx: str,
        glb: Optional[str] = None,
        *,
        search_dirs: Sequence[str] = (),
        hdr_codec: Any = None,
        logger: Optional[logging.Logger] = None,
    ) -> List[str]:
        """Give an FBX deliverable's probe the highlight copies its GLB ships.

        A GLB's probe carries copies with the bake's lights drawn in
        (:meth:`_embed_highlight_probes`). The FBX carried the probe alone, so
        a reader that shades it as the WebXR viewer does (unitytk's
        deliverable shading) lit its metals with the probe as captured, the
        bench's lights missing from them. The copies are *glb*'s -- the GLB
        this export built from *fbx* -- or, without one, those of a GLB built
        here the same way (:meth:`fbx_to_glb`, whose lightmap pass draws
        them) from *fbx* without its takes (:meth:`_fbx_static_scene`), so
        both deliverables carry the same copies to the byte. Each
        goes inside the FBX as its Radiance bytes, under
        :attr:`FBX_HIGHLIGHT_NAME`, and the record's probe lists them:
        ``highlights: [{"map", "receivers", "objects": [{"name",
        "hierarchy"}]}]``. An engine binds a renderer, not a material, so each
        mesh node is listed under the one copy holding most of its triangles.

        Run on a freshly written FBX: a name the file already embeds keeps its
        bytes (:meth:`FbxMedia.embed`). A no-op on a file whose record has no
        probe or no lights. Never the deliverable's cost: a failure is warned
        and the file stays as written.

        Parameters:
            fbx: The FBX deliverable, rewritten in place.
            glb: A GLB built from *fbx*; ``None`` builds one (FBX2glTF must be
                installed: none is downloaded for this).
            search_dirs: Where the bake's maps and probe are, for that build
                (:meth:`apply_glb_lightmaps`).
            hdr_codec: That build's HDR encoder (:meth:`apply_glb_lightmaps`).
            logger: Where the outcome is said; this module's otherwise.

        Returns:
            The copies' file names, in the record's order; empty when none
            shipped.
        """
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        log = logger or logging.getLogger(__name__)
        key = cls.LIGHTMAP_METADATA_KEY
        try:
            record = FbxMedia.record(fbx, key)
        except (OSError, ValueError) as error:
            log.warning("FBX highlight copies: %s not read (%s).", fbx, error)
            return []
        probe = record.get("probe") if isinstance(record, dict) else None
        if not (
            isinstance(probe, dict)
            and probe.get("map")
            and cls.lights_in_metres(record.get("lights"))
        ):
            return []
        store = TempArtifacts("fbx_highlights", policy="scoped")
        try:
            copies = cls._fbx_highlight_copies(fbx, glb, store, search_dirs, hdr_codec)
            stem = os.path.splitext(os.path.basename(str(probe["map"])))[0]
            entries, files = [], []
            for index, copy in enumerate(copies, 1):
                name = cls.FBX_HIGHLIGHT_NAME.format(stem=stem, index=index)
                path = os.path.join(store.dir_path(), name)
                with open(path, "wb") as fh:
                    fh.write(copy["bytes"])
                files.append((name, path))
                entries.append(
                    {"map": name, "receivers": copy["receivers"], "objects": copy["objects"]}
                )
            if not entries and "highlights" not in probe:
                return []
            record = dict(record, probe=dict(probe, highlights=entries))
            if not entries:
                del record["probe"]["highlights"]
            FbxMedia.set_record(fbx, key, record, files=files)
        except Exception as error:  # noqa: BLE001 -- never the deliverable's cost
            log.warning("FBX highlight copies not shipped: %s", error)
            log.debug("FBX highlight copies failed.", exc_info=True)
            return []
        finally:
            store.cleanup()
        if entries:
            log.info(
                "FBX highlight copies: %d embedded (%s).",
                len(entries),
                ", ".join(f"{e['receivers']}: {len(e['objects'])} object(s)" for e in entries),
            )
        return [entry["map"] for entry in entries]

    @classmethod
    def _fbx_highlight_copies(
        cls, fbx: str, glb: Optional[str], store, search_dirs, hdr_codec
    ) -> List[Dict[str, Any]]:
        """*glb*'s probe copies -- built into *store* when ``None``, from *fbx*'s
        scene at rest (:meth:`_fbx_static_scene`) -- each ``{"bytes",
        "receivers", "objects"}`` (:meth:`_highlight_receivers`)."""
        if glb is None:
            source = cls._fbx_static_scene(fbx, store)
            glb = cls.fbx_to_glb(
                source,
                os.path.join(store.dir_path(), "highlights.glb"),
                overwrite=True,
                auto_install=False,
                prompt=False,
                lightmap_dirs=list(search_dirs),
                hdr_codec=hdr_codec,
                # A copy without its takes has no shots to cut: the clip pass
                # would warn that the export was written with animation off.
                data_export={cls.FBX_TAKES_KEY: None} if source != fbx else None,
            )
        with cls.open_glb(glb) as edit:
            web = cls._lightmap_web_manifest(edit.gltf) or {}
            highlights = (web.get("probe") or {}).get("highlights") or {}
            entries = [
                e
                for e in highlights.get("copies") or []
                if isinstance(e, dict) and isinstance(e.get("bufferView"), int)
            ]
            objects = cls._highlight_receivers(
                edit.gltf, [set(e.get("materials") or ()) for e in entries]
            )
            copies = []
            for entry, receivers in zip(entries, objects):
                data = edit.image_bytes(entry)
                if data and receivers:
                    copies.append(
                        {
                            "bytes": bytes(data),
                            "receivers": entry.get("receivers"),
                            "objects": receivers,
                        }
                    )
        return copies

    @staticmethod
    def _fbx_static_scene(fbx: str, store) -> str:
        """The FBX a copies build converts: *fbx* without its takes
        (:meth:`FbxMedia.drop_takes`), written into *store* under its own
        name -- or *fbx* itself when it carries none.

        The copies read the scene at rest -- FBX2glTF writes each node's
        transform as its properties hold it, which no take moves, and the
        copies read its triangles in an order of their own
        (:meth:`_canonical_triangles`) -- while FBX2glTF bakes every node at
        every frame of every take. Measured on a production assembly (2486
        nodes, 13 takes, 133 MB; 2026-10-19): 1013 s of FBX2glTF, 108 s
        without the takes, the same copies to the byte. The
        media stay: a diffuse map's alpha is the converter's opaque/blend
        verdict, which reorders a mesh's primitives -- and so the copies,
        grouped in the order their materials first appear (measured: a mesh
        whose map went missing swapped two unbaked groups' copies).
        """
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia

        static = os.path.join(store.dir_path(), os.path.basename(fbx))
        return static if FbxMedia.drop_takes(fbx, static)["takes"] else fbx

    #: The root FBX2glTF puts every converted scene under; not the scene's.
    _CONVERTER_ROOT: str = "RootNode"

    @classmethod
    def _highlight_receivers(
        cls, gltf: Dict[str, Any], materials: Sequence[set]
    ) -> List[List[Dict[str, Any]]]:
        """The mesh nodes each copy shades, ``{"name", "hierarchy"}`` -- a node
        under the copy whose *materials* hold most of its triangles (the first
        on a tie), as the lightmap record names an object: its hierarchy root
        first, without the converter's root."""
        nodes = gltf.get("nodes") or []
        meshes = gltf.get("meshes") or []
        mats = gltf.get("materials") or []
        accessors = gltf.get("accessors") or []
        parents = cls._node_parents(gltf)
        out: List[List[Dict[str, Any]]] = [[] for _ in materials]
        for index, node in enumerate(nodes):
            mesh = node.get("mesh")
            if not isinstance(mesh, int) or not 0 <= mesh < len(meshes):
                continue
            votes = [0] * len(materials)
            for prim in meshes[mesh].get("primitives") or []:
                m = prim.get("material")
                name = mats[m].get("name") if isinstance(m, int) and 0 <= m < len(mats) else None
                acc = prim.get("indices", (prim.get("attributes") or {}).get("POSITION"))
                count = (
                    int(accessors[acc].get("count", 0))
                    if isinstance(acc, int) and 0 <= acc < len(accessors)
                    else 0
                )
                for c, wanted in enumerate(materials):
                    if name in wanted:
                        votes[c] += count
            if not votes or max(votes) <= 0:
                continue
            lineage = [
                str(nodes[i].get("name") or "")
                for i in reversed(cls._node_lineage(parents, index))
            ]
            if len(lineage) > 1 and lineage[0] == cls._CONVERTER_ROOT:
                lineage = lineage[1:]
            out[votes.index(max(votes))].append({"name": lineage[-1], "hierarchy": lineage})
        return out

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @classmethod
    def _highlight_eye(cls, edit) -> Optional[List[float]]:
        """The start camera's world position (:attr:`HIGHLIGHT_EYE_NODE`, a
        namespaced one too), or ``None``."""
        from pythontk.file_utils.mesh_convert.glb.reader import GlbReader

        name = cls.HIGHLIGHT_EYE_NODE
        nodes = edit.gltf.get("nodes") or []
        index = next(
            (i for i, n in enumerate(nodes) if n.get("name") == name),
            next(
                (
                    i
                    for i, n in enumerate(nodes)
                    if str(n.get("name") or "").rsplit(":", 1)[-1] == name
                ),
                None,
            ),
        )
        if index is None:
            return None
        position = GlbReader(edit).world_position(index)
        return list(position) if position else None

    @classmethod
    def _highlight_meshes(cls, edit) -> List[Dict[str, Any]]:
        """Every triangle primitive in model space: ``{material, triangles, normals}``.

        ``triangles`` ``(n, 3, 3)`` corners and ``normals`` ``(n, 3, 3)`` their
        vertex normals (or the face's), both through the node's world matrix.
        Read through :class:`GlbReader` -- the one accessor decoder.
        """
        from pythontk.file_utils.mesh_convert.glb.reader import GlbReader

        reader = GlbReader(edit)
        gltf = edit.gltf
        meshes = gltf.get("meshes") or []
        materials = gltf.get("materials") or []
        out = []
        for index, node in enumerate(gltf.get("nodes") or []):
            if "mesh" not in node or not 0 <= node["mesh"] < len(meshes):
                continue
            matrix = reader.world_matrix(index)
            if matrix is None:
                continue
            m = np.asarray(matrix, float)  # row vectors: p' = p @ m
            linear, offset = m[:3, :3], m[3, :3]
            try:
                normal_matrix = np.linalg.inv(linear).T
            except np.linalg.LinAlgError:
                continue
            for prim in meshes[node["mesh"]].get("primitives") or []:
                if prim.get("mode", 4) != 4:
                    continue
                attributes = prim.get("attributes") or {}
                positions = reader.accessor(attributes.get("POSITION"))
                if not positions:
                    continue
                p = np.asarray(positions, float) @ linear + offset
                if "indices" in prim:
                    indices = reader.accessor(prim["indices"])
                    if not indices:
                        continue
                    tri = np.asarray(indices, np.int64).reshape(-1, 3)
                else:
                    tri = np.arange(len(p) // 3 * 3).reshape(-1, 3)
                normals = reader.accessor(attributes.get("NORMAL"))
                corners = p[tri]
                if normals and len(normals) == len(positions):
                    n = np.asarray(normals, float) @ normal_matrix
                    vn = n[tri]
                else:
                    face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
                    vn = np.repeat(face[:, None, :], 3, axis=1)
                vn /= np.maximum(np.linalg.norm(vn, axis=-1, keepdims=True), 1e-12)
                material = prim.get("material")
                name = (
                    materials[material].get("name")
                    if isinstance(material, int) and 0 <= material < len(materials)
                    else None
                )
                out.append({"material": name, "triangles": corners, "normals": vn})
        return out

    @classmethod
    def _highlight_samples(
        cls, meshes: Sequence[Dict[str, Any]], seed: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Area-weighted points ``(n, 3)`` and normals ``(n, 3)`` on *meshes*' triangles
        (:attr:`HIGHLIGHT_SAMPLES_PER_M2`, at least :attr:`HIGHLIGHT_MIN_SAMPLES`
        and at most :attr:`_HIGHLIGHT_MAX_SAMPLES` in all) -- by area alone, so
        a finely cut surface draws no more than a coarse one as large -- and
        drawn over the triangles in :meth:`_canonical_triangles` order, so the
        samples are the geometry's, not its storage's: a build that stores the
        same surfaces another way draws the same copies."""
        if not meshes:
            return np.zeros((0, 3)), np.zeros((0, 3))
        c, vn = cls._canonical_triangles(
            np.concatenate([m["triangles"] for m in meshes]),
            np.concatenate([m["normals"] for m in meshes]),
        )
        area = 0.5 * np.linalg.norm(np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]), axis=1)
        total = float(area.sum())
        wanted = min(
            max(total * cls.HIGHLIGHT_SAMPLES_PER_M2, cls.HIGHLIGHT_MIN_SAMPLES),
            cls._HIGHLIGHT_MAX_SAMPLES,
        )
        rng = np.random.default_rng(seed)
        tri = np.repeat(np.arange(len(c)), rng.poisson(area * wanted / max(total, 1e-9)))
        if not len(tri):
            return np.zeros((0, 3)), np.zeros((0, 3))
        r1, r2 = rng.random(len(tri)), rng.random(len(tri))
        s = np.sqrt(r1)
        bary = np.stack([1 - s, s * (1 - r2), s * r2], 1)[:, :, None]
        n = (vn[tri] * bary).sum(1)
        return (c[tri] * bary).sum(1), n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    @staticmethod
    def _canonical_triangles(
        corners: np.ndarray, normals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """*corners* ``(n, 3, 3)`` and their *normals* in an order of their own:
        each triangle's corners rotated to start at its least (lexicographic,
        to the micrometre), its winding kept, and the triangles sorted by
        their corners. The same surfaces stored in any order, split across any
        meshes, come out alike."""
        key = np.round(corners, 6)

        def before(a, b):  # a < b lexicographically, row by row
            lt, eq = a < b, a == b
            return lt[:, 0] | (eq[:, 0] & (lt[:, 1] | (eq[:, 1] & lt[:, 2])))

        first = np.where(before(key[:, 1], key[:, 0]), 1, 0)
        least = np.where(first[:, None] == 1, key[:, 1], key[:, 0])
        first = np.where(before(key[:, 2], least), 2, first)
        turn = ((first[:, None] + np.arange(3)[None]) % 3)[..., None]
        corners = np.take_along_axis(corners, turn, 1)
        normals = np.take_along_axis(normals, turn, 1)
        order = np.lexsort(np.take_along_axis(key, turn, 1).reshape(len(key), 9).T[::-1])
        return corners[order], normals[order]

    @classmethod
    def _highlight_groups(
        cls,
        meshes: Sequence[Dict[str, Any]],
        baked: set,
        lo: Optional[np.ndarray],
        hi: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """The receiver groups that get a copy, with their surface samples.

        The shell (a baked material whose every surface lies on one of the
        box's faces) gets none. Unbaked materials group by overlapping bounds,
        a copy per group, within :attr:`HIGHLIGHT_MAX_COPIES`; the baked
        interior shares one.
        """
        by_material: Dict[Any, List[Dict[str, Any]]] = {}
        for mesh in meshes:
            if mesh["material"] is not None:  # no material, no probe copy to take
                by_material.setdefault(mesh["material"], []).append(mesh)
        bounds = {
            m: (
                np.min([x["triangles"].reshape(-1, 3).min(0) for x in ms], 0),
                np.max([x["triangles"].reshape(-1, 3).max(0) for x in ms], 0),
            )
            for m, ms in by_material.items()
        }
        margin = 0.05 * float(np.min(hi - lo)) if lo is not None else 0.0

        def on_shell(b_lo, b_hi):
            if lo is None:
                return False
            for axis in range(3):
                for face in (lo[axis], hi[axis]):
                    if abs(b_lo[axis] - face) <= margin and abs(b_hi[axis] - face) <= margin:
                        return True
            return False

        unbaked = [m for m in by_material if m not in baked]
        interior = [m for m in by_material if m in baked and not on_shell(*bounds[m])]
        groups: List[Dict[str, Any]] = []

        # Unbaked: union of overlapping bounds (an asset's parts share a copy).
        pad = 0.05
        parent = {m: m for m in unbaked}

        def find(m):
            while parent[m] != m:
                parent[m] = parent[parent[m]]
                m = parent[m]
            return m

        for i, a in enumerate(unbaked):
            for b in unbaked[i + 1 :]:
                (alo, ahi), (blo, bhi) = bounds[a], bounds[b]
                if np.all(alo - pad <= bhi) and np.all(blo - pad <= ahi):
                    parent[find(a)] = find(b)
        clusters: Dict[Any, List[Any]] = {}
        for m in unbaked:
            clusters.setdefault(find(m), []).append(m)
        for members in clusters.values():
            groups.append({"receivers": "unbaked", "materials": members})

        if len(groups) >= cls.HIGHLIGHT_MAX_COPIES:
            logger.info(
                "Probe highlights: %d unbaked group(s) past the %d copies a probe "
                "ships read the probe as captured.",
                len(groups) - cls.HIGHLIGHT_MAX_COPIES + 1,
                cls.HIGHLIGHT_MAX_COPIES,
            )
            groups = groups[: cls.HIGHLIGHT_MAX_COPIES - 1]
        if interior:
            groups.append({"receivers": "interior", "materials": interior})
        for group in groups:
            group["points"], group["normals"] = cls._highlight_samples(
                [x for m in group["materials"] for x in by_material[m]]
            )
        return [g for g in groups if len(g["points"])]

    # ------------------------------------------------------------------
    # Lights
    # ------------------------------------------------------------------

    @staticmethod
    def _unit(value, default) -> np.ndarray:
        """*value* normalized (*default*, a unit vector, when it is absent or ~zero)."""
        return np.asarray(
            MathUtils.safe_normalize(value if value is not None else default, default), float
        )

    @staticmethod
    def _spot_cone(light) -> Tuple[float, float]:
        """A spot's smoothstep edges, as three.js eases it: ``(cos outer, cos inner)``."""
        angle = float(light.get("angle", math.pi / 4))
        return math.cos(angle), math.cos(angle * (1.0 - float(light.get("penumbra", 0.0))))

    @classmethod
    def _rect_frame(cls, light) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A rect's ``(facing, right, up)``: the way it shines, its width and height axes."""
        facing = cls._unit(light.get("direction"), (0, -1, 0))
        right = cls._unit(light.get("right"), (1, 0, 0))
        up = cls._unit(np.cross(facing, right), (0, 0, 1))
        right = np.cross(up, facing)
        return facing, right, up

    @classmethod
    def _highlight_field(cls, light, points: np.ndarray):
        """The light at each point as three.js lights it: ``(E, direction, luma)``.

        ``E`` ``(n, 3)`` the irradiance on a surface facing it (a point or spot
        light ``I c / max(d^2, 0.01)``, a spot through its smoothstep cone; a
        rect integrated over its area, one-sided), ``direction`` ``(n, 3)``
        toward it (a rect's irradiance-weighted mean), ``luma`` ``(n,)``.
        """
        color = np.asarray(light.get("color") or (1, 1, 1), float)[:3]
        I = float(light["intensity"])
        X = np.asarray(light["position"], float)
        kind = light["type"]
        if kind == "rect":
            facing, right, up = cls._rect_frame(light)
            w, h = float(light.get("width") or 1.0), float(light.get("height") or 1.0)
            n_long = 12
            nu = n_long if w >= h else max(2, int(round(n_long * w / h)))
            nv = n_long if h > w else max(2, int(round(n_long * h / w)))
            area = w * h / (nu * nv)
            emitters = [
                X + right * ((i + 0.5) / nu - 0.5) * w + up * ((j + 0.5) / nv - 0.5) * h
                for i in range(nu)
                for j in range(nv)
            ]
            power = color * I * area
        else:
            emitters = [X]
            power = color * I
            facing = cls._unit(light.get("direction"), (0, -1, 0))
        E = np.zeros(points.shape)
        V = np.zeros(points.shape)
        S = np.zeros(len(points))
        for x in emitters:
            l = x - points
            d2 = np.maximum((l * l).sum(1), 1e-12)
            lh = l / np.sqrt(d2)[:, None]
            f = 1.0 / np.maximum(d2, 0.01)
            if kind == "spot":
                f = f * MathUtils.smoothstep(
                    (-lh) @ facing, *cls._spot_cone(light)
                )
            elif kind == "rect":
                f = f * np.maximum((-lh) @ facing, 0.0)
            e = f[:, None] * power
            y = e @ _LUMA
            E += e
            V += y[:, None] * lh
            S += y
        direction = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        return E, direction, S

    @staticmethod
    def _ggx_d(normals, l, v, rough: float):
        """GGX D of the half vector of *l* and *v* (UE4 alpha = roughness^2)."""
        h = l + v
        h = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
        a2 = (rough * rough) ** 2
        nh = np.clip((normals * h).sum(-1), 0.0, 1.0)
        return a2 / (np.pi * (nh * nh * (a2 - 1.0) + 1.0) ** 2)

    @staticmethod
    def _box_exit(origin, d, lo, hi):
        """Where rays ``origin + t d`` leave the box ``[lo, hi]`` (from inside)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.maximum((hi - origin) / d, (lo - origin) / d)
        t = np.where(np.isfinite(t), t, np.inf).min(-1)
        return origin + d * t[..., None]

    @classmethod
    def _highlight_placement(cls, light, points, normals, eye, lo, hi, irradiance):
        """``(F, scale)``: where the light is drawn from and how bright, for these receivers.

        F is its highlight hot spot as the eye sees it (squared-weight
        centroid), or with *irradiance* (an unbaked group: its light is all
        the probe's) the centroid of its irradiance. The scale makes the light
        the proxy gives the samples -- a source at B, the box point the ray
        from F through the light reaches, as bright at F as the light is --
        the light's own: the same highlight, or the same irradiance.
        """
        X = np.asarray(light["position"], float)
        E, d, S = cls._highlight_field(light, points)
        nl = np.clip((normals * d).sum(1), 0.0, None)
        live = S * nl
        if irradiance:
            weights = live
        else:
            v = eye - points
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            spec = live * cls._ggx_d(normals, d, v, cls.HIGHLIGHT_ROUGHNESS)
            weights = spec * spec
        if weights.sum() <= 0.0:
            return points.mean(0) if len(points) else X, 0.0
        F = (points * weights[:, None]).sum(0) / weights.sum()
        toward = cls._unit(X - F, (0, 1, 0))
        S_F = float(cls._highlight_field(light, F[None])[2][0])
        if lo is not None:
            B = cls._box_exit(F, toward[None], lo, hi)[0]
            b = B - points
            bd2 = np.maximum((b * b).sum(1), 1e-12)
            b /= np.sqrt(bd2)[:, None]
            proxy = S_F * float(((B - F) ** 2).sum()) / bd2
        else:  # read as distant: the proxy is the same light at every point
            b = np.broadcast_to(toward, points.shape)
            proxy = np.full(len(points), S_F)
        proxy = proxy * np.clip((normals * b).sum(1), 0.0, None)
        if not irradiance:
            live = live * cls._ggx_d(normals, d, v, cls.HIGHLIGHT_ROUGHNESS)
            proxy = proxy * cls._ggx_d(normals, b, v, cls.HIGHLIGHT_ROUGHNESS)
        if live.sum() <= 0.0 or proxy.sum() <= 0.0:
            return F, 0.0
        low, high = cls._HIGHLIGHT_SCALE_RANGE
        return F, float(np.clip(live.sum() / proxy.sum(), low, high))

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    @staticmethod
    def _equirect_directions(rows, cols, w, h, ss):
        """Unit directions ``(len(rows)*ss, len(cols)*ss, 3)`` of those texels' subsamples
        (three.js' equirect: ``u = 0.5 + atan2(z, x) / 2pi``, top row +Y)."""
        offsets = (np.arange(ss) + 0.5) / ss
        u = ((np.asarray(cols)[:, None] + offsets[None]).ravel()) / w
        v = ((np.asarray(rows)[:, None] + offsets[None]).ravel()) / h
        phi = (u - 0.5) * 2.0 * np.pi
        lat = (0.5 - v) * np.pi
        cl = np.cos(lat)[:, None]
        return np.stack(
            [
                np.cos(phi)[None, :] * cl,
                np.broadcast_to(np.sin(lat)[:, None], (len(v), len(u))),
                np.sin(phi)[None, :] * cl,
            ],
            -1,
        )

    @classmethod
    def _proxy_radiance(cls, light, F, origin, d, far, scale, r):
        """Radiance ``(..., 3)`` rays ``origin + s d`` (``s < far``) meet on the light, as F sees it
        (a point or spot light as a sphere of radius *r*: :meth:`_proxy_radius`)."""
        X = np.asarray(light["position"], float)
        color = np.asarray(light.get("color") or (1, 1, 1), float)[:3]
        I = float(light["intensity"]) * scale
        out = np.zeros(d.shape[:-1] + (3,))
        if light["type"] == "rect":
            facing, right, up = cls._rect_frame(light)
            if float(np.dot(facing, F - X)) <= 0.0:
                return out  # behind a one-sided rect
            denom = d @ facing
            with np.errstate(divide="ignore", invalid="ignore"):
                s = ((X - origin) @ facing) / denom
            q = origin + d * s[..., None] - X
            hit = (
                (s > 0)
                & (s < far)
                & (denom < 0)
                & (np.abs(q @ right) <= float(light.get("width") or 1.0) / 2)
                & (np.abs(q @ up) <= float(light.get("height") or 1.0) / 2)
            )
            out[hit] = I * color
            return out
        dist = float(np.linalg.norm(X - F))
        omega = 2.0 * math.pi * (1.0 - math.sqrt(1.0 - (r / dist) ** 2))
        att = 1.0
        if light["type"] == "spot":
            axis = cls._unit(light.get("direction"), (0, -1, 0))
            att = MathUtils.smoothstep(
                float(np.dot((F - X) / dist, axis)), *cls._spot_cone(light)
            )
            if att <= 0.0:
                return out
        L = I * att / (max(dist * dist, 0.01) * omega)
        oc = origin - X
        b = (d * oc).sum(-1)
        c = float((oc * oc).sum()) - r * r if oc.ndim == 1 else (oc * oc).sum(-1) - r * r
        disc = b * b - c
        s = -b - np.sqrt(np.maximum(disc, 0.0))
        out[(disc > 0) & (s > 0) & (s < far)] = L * color
        return out

    @classmethod
    def _highlight_layer(cls, lights, focal, scales, C, lo, hi, width):
        """``(width/2, width, 3)``: each light drawn from its focal point, onto the box as C reads it.

        Each light is drawn only over its footprint: a coarse pass with its
        bounds grown by a coarse texel finds the texels it can reach.
        """
        h, w, ss = width // 2, width, cls.HIGHLIGHT_SUPERSAMPLE
        layer = np.zeros((h, w, 3))
        # Coarse texels of f x f fine ones: f divides the height, so both grids
        # tile the same rows and columns.
        f = max(1, h // 64)
        while h % f:
            f -= 1
        cw = w // f
        coarse = cls._equirect_directions(np.arange(h // f), np.arange(cw), cw, h // f, 1)
        coarse_b = cls._box_exit(C, coarse, lo, hi) if lo is not None else None
        for light, F, scale in zip(lights, focal, scales):
            if scale <= 0.0:
                continue
            F = np.asarray(F, float)
            r = cls._proxy_radius(light, F, C, lo, hi, w)
            reach = cls._rect_reach(light) if light["type"] == "rect" else r
            hit = cls._coarse_reach(light, F, C, coarse, coarse_b, reach, cw)
            if not hit.any():
                continue
            # Grown by a coarse texel; columns wrap round, rows stop at the poles.
            rows_c = np.where(hit.any(1))[0]
            cols_c = np.where(hit.any(0))[0]
            rows_c = np.unique(np.clip(np.concatenate([rows_c - 1, rows_c, rows_c + 1]), 0, h // f - 1))
            cols_c = np.unique(np.concatenate([(cols_c + k) % cw for k in (-1, 0, 1)]))
            rows = (rows_c[:, None] * f + np.arange(f)[None]).ravel()
            cols = (cols_c[:, None] * f + np.arange(f)[None]).ravel()
            for start in range(0, len(rows), 64):
                rs = rows[start : start + 64]
                d = cls._equirect_directions(rs, cols, w, h, ss)
                if lo is not None:
                    B = cls._box_exit(C, d, lo, hi)
                    fd = B - F
                    far = np.linalg.norm(fd, axis=-1)
                    fd = fd / np.maximum(far, 1e-12)[..., None]
                else:
                    fd, far = d, np.full(d.shape[:-1], np.inf)
                rad = cls._proxy_radiance(light, F, F, fd, far, scale, r)
                rad = rad.reshape(len(rs), ss, len(cols), ss, 3).mean((1, 3))
                layer[np.ix_(rs, cols)] += rad
        return layer.astype(np.float32)

    @classmethod
    def _proxy_radius(cls, light, F, C, lo, hi, width) -> float:
        """The sphere a point or spot light is drawn as, seen from F: its own
        radius, at least :attr:`HIGHLIGHT_MIN_ANGLE` and at least one texel --
        the layer's texels are laid out from C, and a light smaller than one as
        F sees it fell between every ray and was not drawn at all. Its energy
        is its own whatever the size (radiance x solid angle). At most half
        its distance from F: a light near a wall, seen from C, spans many
        texels, and a texel scaled past 90 degrees (a negative tangent) once
        sized the sphere 0 and the drawing raised on its zero solid angle."""
        X = np.asarray(light["position"], float)
        dist = float(np.linalg.norm(X - F))
        texel = 2.0 * math.pi / width
        if lo is not None:
            toward = cls._unit(X - F, (0, 1, 0))
            B = cls._box_exit(F, toward[None], lo, hi)[0]
            texel *= float(np.linalg.norm(B - C)) / max(float(np.linalg.norm(B - F)), 1e-9)
        angle = min(max(math.radians(cls.HIGHLIGHT_MIN_ANGLE), texel), math.atan(0.5))
        r = max(float(light.get("radius") or 0.0), dist * math.tan(angle))
        return min(r, 0.5 * dist)

    @staticmethod
    def _rect_reach(light) -> float:
        """The radius of a sphere about a rect light holding all of it: its half-diagonal."""
        return 0.5 * math.hypot(float(light.get("width") or 1.0), float(light.get("height") or 1.0))

    @classmethod
    def _coarse_reach(cls, light, F, C, coarse, coarse_b, reach, cw):
        """Coarse texels whose ray from F passes within the light's bounds, grown by a texel."""
        X = np.asarray(light["position"], float)
        dist = float(np.linalg.norm(X - F))
        r = reach
        if dist <= r * 1.05:
            return np.ones(coarse.shape[:-1], bool)  # F is all but inside it
        if coarse_b is not None:
            fd = coarse_b - F
            far = np.linalg.norm(fd, axis=-1)
            fd = fd / np.maximum(far, 1e-12)[..., None]
            # A coarse texel's own footprint at the light's distance from F.
            cell = 2.0 * math.pi / cw * np.linalg.norm(coarse_b - C, axis=-1) * dist / np.maximum(far, 1e-9)
        else:
            fd = coarse
            cell = np.full(coarse.shape[:-1], 2.0 * math.pi / cw * dist)
        oc = F - X
        t = -(fd @ oc)
        closest = np.linalg.norm(oc[None, None, :] + fd * t[..., None], axis=-1)
        return (t > 0) & (closest <= r + 1.5 * cell)
