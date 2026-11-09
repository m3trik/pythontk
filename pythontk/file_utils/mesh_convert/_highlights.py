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

So a delivery's probe holds the bake's lights twice over. The probe itself
is the environment with each light drawn in as its capture point sees it --
what a reader that knows nothing of copies reflects, everywhere. And it ships
COMPLETE COPIES, each for a group of receivers and used in the probe's place:
the environment with the lights drawn where box projection shows them true
for that group. For a light and a focal point F: a probe texel whose ray
leaves the box at B shows what F sees of the light along F -> B. A reflection
from F that reads B sees the light exactly -- direction, size and radiance --
and a surface near F sees it nearly so. F is chosen per light and per group,
and each light is scaled so the light it gives the group's own surfaces is
the light's own:

* the room's shell (surfaces on the probe's box: walls, floor, ceiling) gets
  no copy and reads the probe -- lights placed for the bench and read from a
  wall glowed where no light shines;
* an object the bake left out (no lightmap, so everything it shows comes from
  the probe) gets a copy of its own: F at the centroid of each light's
  irradiance on it, the irradiance matched over it;
* the rest of the room's interior shares one: F at each light's highlight hot
  spot as the eye sees it, the highlight matched -- a baked surface shows a
  light only through its highlight, its diffuse being the bake's.

A receiver is a mesh node: the record's lightmapped nodes are told from the
rest by the rule every reader resolves the record by, so a material baked and
unbaked nodes share puts each where its own bake calls for. The files are the
delivery's (:meth:`_deliverable_probes`): an FBX embeds each under its name.

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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pythontk.core_utils.color import Color
from pythontk.core_utils.engines.scene_export.lightmap_record import LightmapRecord
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.math_utils._math_utils import MathUtils

logger = logging.getLogger(__name__)

_LUMA = np.asarray(Color.REC709_LUMINANCE)


class _Highlights:
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
    #: The narrowest copy drawn: three.js' PMREM prefilters nothing narrower
    #: than 64 texels (a 16-texel cube face), so a smaller copy reflected no map.
    _HIGHLIGHT_MIN_WIDTH: int = 64

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    def highlight_probe_layers(
        self,
        glb,
        lights: Optional[Sequence[Dict[str, Any]]],
        probe: Optional[Dict[str, Any]] = None,
        width: Optional[int] = None,
        baked: Optional[Sequence[str]] = None,
        eye: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        # The one light validator (idempotent on its own output): a light
        # written wrong is left out, never the pass.
        lights = self._mc.lights_in_metres(lights)
        if not lights:
            return []
        with self._mc.open_glb(glb) as edit:
            # The file's own record -- a GLB carries it as it stands -- when
            # the caller gives no probe or baked materials.
            record = SceneRecords.glb_block(edit.gltf, SceneRecords.LIGHTMAPS) or {}
            probe = probe if probe is not None else self._probe_in_metres(record)
            if not isinstance(probe, dict) or not probe.get("position"):
                return []
            if baked is None:
                # Grouped as the delivery's copies are: by receiver, a
                # material baked and unbaked nodes share two of them.
                meshes, keys = self._receiver_meshes(edit, record)
            else:
                meshes, keys = self._mc._highlight_meshes(edit), set(baked)
            if eye is None:
                eye = self._mc._highlight_eye(edit)
            gltf = edit.gltf
        C = np.asarray(probe["position"], float)
        box = probe.get("box")
        lo = np.asarray(box["min"], float) if box else None
        hi = np.asarray(box["max"], float) if box else None
        eye = np.asarray(eye if eye is not None else C, float)
        width = int(width or 1024) // 2 * 2
        groups = self._mc._highlight_groups(meshes, keys, lo, hi)
        layers = self._draw_groups(groups, lights, C, lo, hi, width, eye)
        receivers = self._mc._highlight_receivers(
            gltf,
            [set(layer["materials"]) for layer in layers],
            {(m["node"], m["primitive"]): m["material"] for m in meshes},
        )
        out: List[Dict[str, Any]] = []
        # What the delivery's probe itself adds: each light as the capture
        # point sees it, read by every receiver in no copy.
        n = len(lights)
        seen = self._mc._highlight_layer(lights, [C] * n, [1.0] * n, C, lo, hi, width)
        if np.any(seen > 0):
            out.append({"receivers": "probe", "materials": [], "objects": [], "image": seen})
        for layer, objects in zip(layers, receivers):
            names = {k[1] if isinstance(k, tuple) else k for k in layer["materials"]}
            out.append(
                dict(layer, materials=sorted(n for n in names if n), objects=objects)
            )
        return out

    def _draw_groups(self, groups, lights, C, lo, hi, width, eye) -> List[Dict[str, Any]]:
        """Each group's layer: ``{"materials", "receivers", "image"}`` --
        every light drawn from that group's focal point at its scale
        (:meth:`_highlight_placement`), the interior's at *width*, an
        unbaked group's at :meth:`_copy_width`; a group no light reaches is
        left out."""
        layers = []
        for group in groups:
            pts, nrm = group["points"], group["normals"]
            unbaked = group["receivers"] == "unbaked"
            focal, scales = [], []
            for light in lights:
                F, s = self._mc._highlight_placement(
                    light, pts, nrm, eye, lo, hi, irradiance=unbaked
                )
                focal.append(F)
                scales.append(s)
            w = width if group["receivers"] == "interior" else self._copy_width(width)
            image = self._mc._highlight_layer(lights, focal, scales, C, lo, hi, w)
            if not np.any(image > 0):
                continue
            layers.append(
                {
                    "materials": list(group["materials"]),
                    "receivers": group["receivers"],
                    "image": image,
                }
            )
        return layers

    @classmethod
    def _copy_width(cls, width: int) -> int:
        """An unbaked group's copy width: half the probe's where that is a
        whole factor of its rows as well as its columns and no narrower than
        :attr:`_HIGHLIGHT_MIN_WIDTH`, else the probe's own -- a copy is the
        probe box-filtered down to it."""
        half = width // 2
        return half if width % 4 == 0 and half >= cls._HIGHLIGHT_MIN_WIDTH else width

    def lights_in_metres(
        self,
        lights: Optional[Sequence[Dict[str, Any]]],
        unit_scale: Optional[float] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(lights, (list, tuple)):
            return None
        out: List[Dict[str, Any]] = []
        for light in lights:
            if not isinstance(light, dict):
                continue
            kind = light.get("type")
            position = light.get("position")
            try:
                unit = float(
                    light.get("unit_scale", 1.0 if unit_scale is None else unit_scale)
                )
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
    # The deliverable's probe and copies
    # ------------------------------------------------------------------

    #: The name the probe a delivery ships rides under once the bake's lights
    #: are drawn into it: the probe's stem.
    HIGHLIGHT_PROBE_NAME: str = "{stem}_Lights.hdr"
    #: The name a complete highlight copy rides under: the probe's stem and
    #: the copy's place in the record, from 1.
    HIGHLIGHT_COPY_NAME: str = "{stem}_Highlight{index}.hdr"

    def embed_fbx_highlight_probes(
        self,
        fbx: str,
        glb: Optional[str] = None,
        *,
        search_dirs: Sequence[str] = (),
        logger: Optional[logging.Logger] = None,
    ) -> List[str]:
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia
        from pythontk.file_utils.temp_artifacts import TempArtifacts

        log = logger or logging.getLogger(__name__)
        key = SceneRecords.LIGHTMAPS.key
        try:
            record = FbxMedia.record(fbx, key)
        except (OSError, ValueError) as error:
            log.warning("FBX highlight copies: %s not read (%s).", fbx, error)
            return []
        if not self._wants_highlights(record):
            return []
        dirs = [d for d in search_dirs if d]
        dirs.append(os.path.dirname(os.path.abspath(fbx)))
        store = TempArtifacts("fbx_highlights", policy="scoped")
        try:
            source = glb
            if source is None:
                source = self._fbx_geometry(fbx, store)
            drawn = self._deliverable_probes(source, record, dirs, log)
            if not drawn:
                return []
            files = []
            for name, data in drawn["files"]:
                path = os.path.join(store.dir_path(), name)
                with open(path, "wb") as fh:
                    fh.write(data)
                files.append((name, path))
            FbxMedia.set_record(fbx, key, drawn["record"], files=files)
        except Exception as error:  # noqa: BLE001 -- never the deliverable's cost
            log.warning("FBX highlight copies not shipped: %s", error)
            log.debug("FBX highlight copies failed.", exc_info=True)
            return []
        finally:
            store.cleanup()
        probe = drawn["record"]["probe"]
        copies = probe.get("highlights") or []
        log.info(
            "FBX probe highlights: %d light(s) drawn into %s and %d complete "
            "copy(ies) (%s).",
            drawn["lights"],
            probe["map"],
            len(copies),
            ", ".join(f"{len(c['objects'])} object(s)" for c in copies) or "none",
        )
        return [name for name, _data in drawn["files"]]

    def _fbx_geometry(self, fbx: str, store) -> str:
        """A GLB of *fbx*'s scene at rest (:meth:`_fbx_static_scene`), built
        into *store* the way a GLB export builds one, less its lightmap pass:
        the geometry the copies are drawn for, which the record's files do
        not change."""
        source = self._mc._fbx_static_scene(fbx, store)
        return self._mc.fbx_to_glb(
            source,
            os.path.join(store.dir_path(), "highlights.glb"),
            overwrite=True,
            auto_install=False,
            prompt=False,
            lightmaps=False,
            # A copy without its takes has no shots to cut: the clip pass
            # would warn that the export was written with animation off.
            data_export={SceneRecords.SHOTS.key: None} if source != fbx else None,
        )

    def _wants_highlights(self, record: Any) -> bool:
        """Whether *record* -- a ``lightmaps`` record -- calls for the pass:
        current, with a probe and a light to draw into it, and not drawn
        already (a probe with copies, or whose map is the name the pass gives
        a lit probe, :attr:`HIGHLIGHT_PROBE_NAME`)."""
        if not SceneRecords.LIGHTMAPS.is_current(record):
            return False
        probe = record.get("probe")
        if not (isinstance(probe, dict) and probe.get("map")):
            return False
        lit = self._mc.HIGHLIGHT_PROBE_NAME.format(stem="")
        if probe.get("highlights") or os.path.basename(str(probe["map"])).endswith(lit):
            return False
        return bool(self._mc.lights_in_metres(record.get("lights"), record.get("unit_scale")))

    def _deliverable_probes(
        self,
        glb,
        record: Dict[str, Any],
        search_dirs: Sequence[str],
        log: Optional[logging.Logger] = None,
    ) -> Optional[Dict[str, Any]]:
        """The probe a delivery ships with the bake's lights in it, and its copies.

        *record* is the ``lightmaps`` record the delivery carries
        (:meth:`_wants_highlights`); *glb* the geometry it shades (a path or
        an open :class:`GlbEdit`); *search_dirs* where the bake's probe image
        is. Each file is Radiance bytes in the probe's layout and units:

        - the probe the record names from now on (:attr:`HIGHLIGHT_PROBE_NAME`):
          the environment plus each light as the capture point sees it, at
          its own level, at the probe's own size;
        - a complete copy per receiver group (:attr:`HIGHLIGHT_COPY_NAME`):
          the environment box-filtered to the copy's size plus that group's
          layer (:meth:`_draw_groups`) -- what the viewer once summed from
          the two -- named on the record's ``probe.highlights`` with the mesh
          nodes it shades (:meth:`_highlight_receivers`), each node under the
          copy holding most of its triangles.

        Returns:
            ``{"record", "files", "lights"}`` -- the record naming the files,
            ``[(name, bytes)]`` (the probe first, then each copy in the
            record's order) and how many lights were drawn; ``None`` when
            there is nothing to draw, or the probe cannot be read (warned on
            *log*).
        """
        from pythontk.img_utils._img_utils import ImgUtils

        log = log or logger
        probe = record["probe"]
        lights = self._mc.lights_in_metres(record.get("lights"), record.get("unit_scale"))
        place = self._probe_in_metres(record)
        if not lights or place is None:
            return None
        pixels = self._read_probe(str(probe["map"]), search_dirs, log)
        if pixels is None:
            return None
        h, w = pixels.shape[:2]
        if w != 2 * h or w % 2:
            log.warning(
                "Probe highlights: the probe %r is %d x %d, not an equirectangular "
                "map twice as wide as tall; its lights are not drawn.",
                probe["map"],
                w,
                h,
            )
            return None
        C = np.asarray(place["position"], float)
        box = place["box"]
        lo = np.asarray(box["min"], float) if box else None
        hi = np.asarray(box["max"], float) if box else None
        with self._mc.open_glb(glb) as edit:
            meshes, baked = self._receiver_meshes(edit, record)
            eye = self._mc._highlight_eye(edit)
            gltf = edit.gltf
        eye = np.asarray(eye if eye is not None else C, float)
        groups = self._mc._highlight_groups(meshes, baked, lo, hi)
        layers = self._draw_groups(groups, lights, C, lo, hi, w, eye)
        keys = {(m["node"], m["primitive"]): m["material"] for m in meshes}
        receivers = self._mc._highlight_receivers(
            gltf, [set(layer["materials"]) for layer in layers], keys
        )
        stem = os.path.splitext(os.path.basename(str(probe["map"])))[0]
        encode = ImgUtils.encode_hdr_radiance_pixels
        files: List[Tuple[str, bytes]] = []
        copies: List[Dict[str, Any]] = []
        for layer, objects in zip(layers, receivers):
            if not objects:
                continue
            name = self._mc.HIGHLIGHT_COPY_NAME.format(stem=stem, index=len(copies) + 1)
            files.append((name, encode(self._complete_copy(pixels, layer["image"]))))
            copies.append({"map": name, "objects": objects})
        n = len(lights)
        seen = self._mc._highlight_layer(lights, [C] * n, [1.0] * n, C, lo, hi, w)
        shipped = {k: v for k, v in probe.items() if k != "highlights"}
        if np.any(seen > 0):
            shipped["map"] = self._mc.HIGHLIGHT_PROBE_NAME.format(stem=stem)
            files.insert(0, (shipped["map"], encode(pixels + seen)))
        if copies:
            shipped["highlights"] = copies
        if not files:
            return None
        return {"record": dict(record, probe=shipped), "files": files, "lights": n}

    @staticmethod
    def _complete_copy(probe: np.ndarray, layer: np.ndarray) -> np.ndarray:
        """*probe* box-filtered down to *layer*'s size by a whole factor, plus
        *layer*: a complete copy (the sum the WebXR viewer once made of the
        two, ``kernel/probe.js`` ``addProbe``)."""
        h, w = layer.shape[:2]
        f = probe.shape[1] // w
        boxed = probe.reshape(h, f, w, f, 3).mean((1, 3), dtype=np.float64)
        return (boxed + layer).astype(np.float32)

    @staticmethod
    def _read_probe(
        name: str, search_dirs: Sequence[str], log: logging.Logger
    ) -> Optional[np.ndarray]:
        """The bake's probe image *name* from the first of *search_dirs*
        holding it, as linear RGB floats, rows top-down (non-finite and
        negative texels 0, as Radiance holds them); ``None``, warned, when it
        is nowhere or unreadable."""
        from pythontk.img_utils._img_utils import ImgUtils

        basename = os.path.basename(name)
        path = next(
            (
                os.path.join(d, basename)
                for d in search_dirs
                if d and os.path.isfile(os.path.join(d, basename))
            ),
            None,
        )
        if path is None:
            log.warning(
                "Probe highlights: the probe %r is in none of %s; its lights are "
                "not drawn.",
                basename,
                list(search_dirs),
            )
            return None
        try:
            import cv2

            image = ImgUtils.cv2_read(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        except ImportError as error:
            log.warning("Probe highlights: %r not read (%s).", basename, error)
            return None
        if image is None:
            log.warning("Probe highlights: %r is no image cv2 reads.", basename)
            return None
        pixels = np.asarray(image, np.float32)
        if pixels.ndim == 2:
            pixels = np.repeat(pixels[:, :, None], 3, axis=2)
        pixels = np.ascontiguousarray(pixels[:, :, 2::-1])  # cv2's BGR(A): RGB
        return np.maximum(np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    @staticmethod
    def _probe_in_metres(record: Any) -> Optional[Dict[str, Any]]:
        """``{"position", "box"}`` of *record*'s probe in metres (its lengths
        times the record's ``unit_scale``; ``box`` ``{min, max}`` or ``None``
        for an open scene), or ``None`` without a probe placed in a unit."""
        probe = record.get("probe") if isinstance(record, Mapping) else None
        if not isinstance(probe, Mapping):
            return None
        try:
            unit = float(record.get("unit_scale"))
            position = [float(c) * unit for c in probe.get("position")]
        except (TypeError, ValueError):
            return None
        if unit <= 0.0 or len(position) != 3:
            return None
        box = LightmapRecord.box(probe.get("box"))
        if box:
            box = {k: [c * unit for c in box[k]] for k in ("min", "max")}
        return {"position": position, "box": box}

    def _baked_nodes(
        self, gltf: Dict[str, Any], objects: Sequence[Any]
    ) -> Dict[int, Tuple[Any, Tuple[float, ...]]]:
        """``{node index: (map, scaleOffset)}``: the mesh nodes the record's
        *objects* light, resolved as every reader of the record resolves them
        (``_Media.resolve_entries``: by name and hierarchy,
        namespace-tolerant, a tie left out)."""
        media = self._mc._media
        by_name, leaves, _users, lineage = media.node_index(gltf)
        resolved = media.resolve_entries(objects, by_name, leaves, lineage)
        index = {id(node): i for i, node in enumerate(gltf.get("nodes") or [])}
        baked: Dict[int, Tuple[Any, Tuple[float, ...]]] = {}
        for entry, (nodes, status, _leaves) in zip(objects, resolved):
            if not nodes or status in ("ambiguous", "absent"):
                continue
            try:
                rect = tuple(float(v) for v in entry.get("scaleOffset") or (1, 1, 0, 0))
            except (TypeError, ValueError):
                rect = (1.0, 1.0, 0.0, 0.0)
            for node in nodes:
                baked.setdefault(index[id(node)], (entry.get("map"), rect))
        return baked

    def _receiver_meshes(self, edit, record: Mapping[str, Any]):
        """``(meshes, baked)``: :meth:`_highlight_meshes`, each primitive's
        ``material`` its RECEIVER key -- a lightmapped node's material, map
        and rect, an unbaked node's material alone (``None`` without one) --
        and the keys a lightmap lights. A material baked and unbaked nodes
        share is two receivers, each grouped as its own bake calls for."""
        baked = self._baked_nodes(edit.gltf, record.get("objects") or [])
        meshes, lit = [], set()
        for mesh in self._mc._highlight_meshes(edit):
            material = mesh["material"]
            if material is None:
                key = None
            elif mesh["node"] in baked:
                key = ("baked", material) + baked[mesh["node"]]
                lit.add(key)
            else:
                key = ("unbaked", material)
            meshes.append(dict(mesh, material=key))
        return meshes, lit

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

    def _highlight_receivers(
        self,
        gltf: Dict[str, Any],
        materials: Sequence[set],
        keys: Optional[Mapping[Tuple[int, int], Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """The mesh nodes each copy shades, ``{"name", "hierarchy"}`` -- a node
        under the copy whose *materials* hold most of its triangles (the first
        on a tie), as the lightmap record names an object: its hierarchy root
        first, without the converter's root. A primitive counts by its
        material's name, or by ``keys[(node, primitive)]`` where *keys* are
        given (:meth:`_receiver_meshes`' receiver keys)."""
        nodes = gltf.get("nodes") or []
        meshes = gltf.get("meshes") or []
        mats = gltf.get("materials") or []
        accessors = gltf.get("accessors") or []
        parents = self._mc._media.node_parents(gltf)
        out: List[List[Dict[str, Any]]] = [[] for _ in materials]
        for index, node in enumerate(nodes):
            mesh = node.get("mesh")
            if not isinstance(mesh, int) or not 0 <= mesh < len(meshes):
                continue
            votes = [0] * len(materials)
            for p, prim in enumerate(meshes[mesh].get("primitives") or []):
                m = prim.get("material")
                if keys is not None:
                    name = keys.get((index, p))
                else:
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
                for i in reversed(self._mc._media.node_lineage(parents, index))
            ]
            if len(lineage) > 1 and lineage[0] == self._CONVERTER_ROOT:
                lineage = lineage[1:]
            out[votes.index(max(votes))].append({"name": lineage[-1], "hierarchy": lineage})
        return out

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _highlight_eye(self, edit) -> Optional[List[float]]:
        """The start camera's world position (:attr:`HIGHLIGHT_EYE_NODE`, a
        namespaced one too), or ``None``."""
        from pythontk.file_utils.mesh_convert.glb.reader import GlbReader

        name = self._mc.HIGHLIGHT_EYE_NODE
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

    def _highlight_meshes(self, edit) -> List[Dict[str, Any]]:
        """Every triangle primitive in model space: ``{node, primitive,
        material, triangles, normals}``.

        ``node`` and ``primitive`` its indices in the file, ``material`` its
        material's name; ``triangles`` ``(n, 3, 3)`` corners and ``normals``
        ``(n, 3, 3)`` their vertex normals (or the face's), both through the
        node's world matrix.
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
            for slot, prim in enumerate(meshes[node["mesh"]].get("primitives") or []):
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
                out.append(
                    {
                        "node": index,
                        "primitive": slot,
                        "material": name,
                        "triangles": corners,
                        "normals": vn,
                    }
                )
        return out

    def _highlight_samples(
        self, meshes: Sequence[Dict[str, Any]], seed: int = 0
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
        c, vn = self._canonical_triangles(
            np.concatenate([m["triangles"] for m in meshes]),
            np.concatenate([m["normals"] for m in meshes]),
        )
        area = 0.5 * np.linalg.norm(np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]), axis=1)
        total = float(area.sum())
        wanted = min(
            max(
                total * self._mc.HIGHLIGHT_SAMPLES_PER_M2,
                self._mc.HIGHLIGHT_MIN_SAMPLES,
            ),
            self._mc._HIGHLIGHT_MAX_SAMPLES,
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

    def _highlight_groups(
        self,
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

        if len(groups) >= self._mc.HIGHLIGHT_MAX_COPIES:
            logger.info(
                "Probe highlights: %d unbaked group(s) past the %d copies a probe "
                "ships read the probe as captured.",
                len(groups) - self._mc.HIGHLIGHT_MAX_COPIES + 1,
                self._mc.HIGHLIGHT_MAX_COPIES,
            )
            groups = groups[: self._mc.HIGHLIGHT_MAX_COPIES - 1]
        if interior:
            groups.append({"receivers": "interior", "materials": interior})
        for group in groups:
            group["points"], group["normals"] = self._mc._highlight_samples(
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

    def _rect_frame(self, light) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A rect's ``(facing, right, up)``: the way it shines, its width and height axes."""
        facing = self._unit(light.get("direction"), (0, -1, 0))
        right = self._unit(light.get("right"), (1, 0, 0))
        up = self._unit(np.cross(facing, right), (0, 0, 1))
        right = np.cross(up, facing)
        return facing, right, up

    def _highlight_field(self, light, points: np.ndarray):
        """The light at each point as three.js lights it: ``(E, direction, luma)``.

        ``E`` ``(n, 3)`` the irradiance on a surface facing it (a point or spot
        light ``I c / max(d^2, 0.01)``, a spot through its smoothstep cone; a
        rect integrated over its area, one-sided), ``direction`` ``(n, 3)``
        toward it (a rect's irradiance-weighted mean), ``luma`` ``(n,)``.
        """
        color = np.asarray(light.get("color") or (1, 1, 1), float)[:3]
        intensity = float(light["intensity"])
        X = np.asarray(light["position"], float)
        kind = light["type"]
        if kind == "rect":
            facing, right, up = self._rect_frame(light)
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
            power = color * intensity * area
        else:
            emitters = [X]
            power = color * intensity
            facing = self._unit(light.get("direction"), (0, -1, 0))
        E = np.zeros(points.shape)
        V = np.zeros(points.shape)
        S = np.zeros(len(points))
        for x in emitters:
            to_light = x - points
            d2 = np.maximum((to_light * to_light).sum(1), 1e-12)
            lh = to_light / np.sqrt(d2)[:, None]
            f = 1.0 / np.maximum(d2, 0.01)
            if kind == "spot":
                f = f * MathUtils.smoothstep(
                    (-lh) @ facing, *self._spot_cone(light)
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
    def _ggx_d(normals, to_light, v, rough: float):
        """GGX D of the half vector of *to_light* and *v* (UE4 alpha = roughness^2)."""
        h = to_light + v
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

    def _highlight_placement(self, light, points, normals, eye, lo, hi, irradiance):
        """``(F, scale)``: where the light is drawn from and how bright, for these receivers.

        F is its highlight hot spot as the eye sees it (squared-weight
        centroid), or with *irradiance* (an unbaked group: its light is all
        the probe's) the centroid of its irradiance. The scale makes the light
        the proxy gives the samples -- a source at B, the box point the ray
        from F through the light reaches, as bright at F as the light is --
        the light's own: the same highlight, or the same irradiance.
        """
        X = np.asarray(light["position"], float)
        E, d, S = self._highlight_field(light, points)
        nl = np.clip((normals * d).sum(1), 0.0, None)
        live = S * nl
        if irradiance:
            weights = live
        else:
            v = eye - points
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            spec = live * self._ggx_d(normals, d, v, self._mc.HIGHLIGHT_ROUGHNESS)
            weights = spec * spec
        if weights.sum() <= 0.0:
            return points.mean(0) if len(points) else X, 0.0
        F = (points * weights[:, None]).sum(0) / weights.sum()
        toward = self._unit(X - F, (0, 1, 0))
        S_F = float(self._highlight_field(light, F[None])[2][0])
        if lo is not None:
            B = self._mc._box_exit(F, toward[None], lo, hi)[0]
            b = B - points
            bd2 = np.maximum((b * b).sum(1), 1e-12)
            b /= np.sqrt(bd2)[:, None]
            proxy = S_F * float(((B - F) ** 2).sum()) / bd2
        else:  # read as distant: the proxy is the same light at every point
            b = np.broadcast_to(toward, points.shape)
            proxy = np.full(len(points), S_F)
        proxy = proxy * np.clip((normals * b).sum(1), 0.0, None)
        if not irradiance:
            live = live * self._ggx_d(normals, d, v, self._mc.HIGHLIGHT_ROUGHNESS)
            proxy = proxy * self._ggx_d(normals, b, v, self._mc.HIGHLIGHT_ROUGHNESS)
        if live.sum() <= 0.0 or proxy.sum() <= 0.0:
            return F, 0.0
        low, high = self._HIGHLIGHT_SCALE_RANGE
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

    def _proxy_radiance(self, light, F, origin, d, far, scale, r):
        """Radiance ``(..., 3)`` rays ``origin + s d`` (``s < far``) meet on the light, as F sees it
        (a point or spot light as a sphere of radius *r*: :meth:`_proxy_radius`)."""
        X = np.asarray(light["position"], float)
        color = np.asarray(light.get("color") or (1, 1, 1), float)[:3]
        intensity = float(light["intensity"]) * scale
        out = np.zeros(d.shape[:-1] + (3,))
        if light["type"] == "rect":
            facing, right, up = self._rect_frame(light)
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
            out[hit] = intensity * color
            return out
        dist = float(np.linalg.norm(X - F))
        omega = 2.0 * math.pi * (1.0 - math.sqrt(1.0 - (r / dist) ** 2))
        att = 1.0
        if light["type"] == "spot":
            axis = self._unit(light.get("direction"), (0, -1, 0))
            att = MathUtils.smoothstep(
                float(np.dot((F - X) / dist, axis)), *self._spot_cone(light)
            )
            if att <= 0.0:
                return out
        L = intensity * att / (max(dist * dist, 0.01) * omega)
        oc = origin - X
        b = (d * oc).sum(-1)
        c = float((oc * oc).sum()) - r * r if oc.ndim == 1 else (oc * oc).sum(-1) - r * r
        disc = b * b - c
        s = -b - np.sqrt(np.maximum(disc, 0.0))
        out[(disc > 0) & (s > 0) & (s < far)] = L * color
        return out

    def _highlight_layer(self, lights, focal, scales, C, lo, hi, width):
        """``(width/2, width, 3)``: each light drawn from its focal point, onto the box as C reads it.

        Each light is drawn only over its footprint: a coarse pass with its
        bounds grown by a coarse texel finds the texels it can reach.
        """
        h, w, ss = width // 2, width, self._mc.HIGHLIGHT_SUPERSAMPLE
        layer = np.zeros((h, w, 3))
        # Coarse texels of f x f fine ones: f divides the height, so both grids
        # tile the same rows and columns.
        f = max(1, h // 64)
        while h % f:
            f -= 1
        cw = w // f
        coarse = self._equirect_directions(
            np.arange(h // f), np.arange(cw), cw, h // f, 1
        )
        coarse_b = self._mc._box_exit(C, coarse, lo, hi) if lo is not None else None
        for light, F, scale in zip(lights, focal, scales):
            if scale <= 0.0:
                continue
            F = np.asarray(F, float)
            r = self._mc._proxy_radius(light, F, C, lo, hi, w)
            reach = self._rect_reach(light) if light["type"] == "rect" else r
            hit = self._coarse_reach(light, F, C, coarse, coarse_b, reach, cw)
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
                d = self._equirect_directions(rs, cols, w, h, ss)
                if lo is not None:
                    B = self._mc._box_exit(C, d, lo, hi)
                    fd = B - F
                    far = np.linalg.norm(fd, axis=-1)
                    fd = fd / np.maximum(far, 1e-12)[..., None]
                else:
                    fd, far = d, np.full(d.shape[:-1], np.inf)
                rad = self._proxy_radiance(light, F, F, fd, far, scale, r)
                rad = rad.reshape(len(rs), ss, len(cols), ss, 3).mean((1, 3))
                layer[np.ix_(rs, cols)] += rad
        return layer.astype(np.float32)

    def _proxy_radius(self, light, F, C, lo, hi, width) -> float:
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
            toward = self._unit(X - F, (0, 1, 0))
            B = self._mc._box_exit(F, toward[None], lo, hi)[0]
            texel *= float(np.linalg.norm(B - C)) / max(float(np.linalg.norm(B - F)), 1e-9)
        angle = min(
            max(math.radians(self._mc.HIGHLIGHT_MIN_ANGLE), texel), math.atan(0.5)
        )
        r = max(float(light.get("radius") or 0.0), dist * math.tan(angle))
        return min(r, 0.5 * dist)

    @staticmethod
    def _rect_reach(light) -> float:
        """The radius of a sphere about a rect light holding all of it: its half-diagonal."""
        return 0.5 * math.hypot(float(light.get("width") or 1.0), float(light.get("height") or 1.0))

    def _coarse_reach(self, light, F, C, coarse, coarse_b, reach, cw):
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
