# !/usr/bin/python
# coding=utf-8
"""Lightmaps: carry a host DCC's committed bake into the GLB deliverable.

One layout, two containers: the GLB carries the ``lightmaps`` record as the
FBX's carrier holds it, in its root ``extras``, and every file the record
names -- each object's map, the probe, the probe's highlight copies -- as the
bufferView named by that file (``SceneRecords.GLB_MEDIA``). Nothing is bound
to a material and nothing is cloned: a reader resolves each entry to its node
in the file's own scene by the one rule (``_Media.resolve_entries``)
and binds what it needs there. The maps ride in their web encoding, the
probe and its copies as Radiance; the bake's lights are drawn into the probe
here as the FBX deliverable's pass draws them (``_highlights.py``).

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through the facade it holds (``self._mc``).
"""

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Eager: the signatures name GlbEdit / GlbTarget; SceneRecords is a light
# declaration module every pass reads.
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget

logger = logging.getLogger(__name__)


class _Lightmaps:
    """Lightmaps: resolve, carry, and write the files the record names.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: Per-object bake marker, riding the same FBX user-property channel as the
    #: record (each host's ``LightmapBaker.LIGHTMAP_INFO_ATTR``).
    LIGHTMAP_INFO_KEY = "lightmapInfo"

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    def _lightmap_manifest(self, gltf: dict) -> Optional[Dict[str, Any]]:
        """The ``lightmaps`` record on a parsed glTF's carrier, or ``None``.

        The channel read is :meth:`data_export_channel`; this adds only the
        applier's own expectation that the manifest is an OBJECT, so a channel
        holding anything else reads as absent rather than reaching the walk as
        something without ``.get``.
        """
        data = self._mc.data_export_channel(gltf, SceneRecords.LIGHTMAPS.key)
        return data if isinstance(data, dict) else None

    def _lightmap_record(self, gltf: dict, warn: bool = False) -> Optional[dict]:
        """The ``lightmaps`` record *gltf* carries in its current schema: its
        root ``extras``' (the record as the file ships it), else its carrier's
        (before the carry; :meth:`_current_record`, which says why another
        version is refused when *warn*); ``None`` without one."""
        carried = SceneRecords.glb_block(gltf, SceneRecords.LIGHTMAPS)
        if carried is not None:
            return carried
        if warn:
            return self._mc._current_record(gltf, SceneRecords.LIGHTMAPS)
        record = self._lightmap_manifest(gltf)
        return record if SceneRecords.LIGHTMAPS.is_current(record) else None

    def read_glb_lightmap_manifest(self, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        with self._mc.open_glb(glb) as edit:
            return self._mc._lightmap_manifest(edit.gltf)

    def fix_glb_lightmap_metadata(self, glb: GlbTarget) -> int:
        with self._mc.open_glb(glb) as edit:
            if not SceneRecords.glb_block(edit.gltf, SceneRecords.LIGHTMAPS):
                return 0
            changed = self._mc.strip_fbx_records(edit.gltf)
            if changed:
                edit.dirty = True
            return changed

    def lit_nodes(self, gltf: dict) -> Dict[int, Dict[str, Any]]:
        """``{node index: entry}``: the mesh nodes *gltf*'s ``lightmaps`` record
        lights -- each resolved to its entry (``_Media.resolve_entries``),
        whose map the file carries (``SceneRecords.GLB_MEDIA``). What a reader
        binds, so what the record lights IN THIS FILE: an entry whose node is
        elsewhere, or whose map did not ship, lights nothing here."""
        record = self._lightmap_record(gltf)
        entries = (record or {}).get("objects") or []
        if not isinstance(entries, list) or not entries:
            return {}
        media = SceneRecords.GLB_MEDIA
        held: Dict[str, int] = {}
        for view in gltf.get("bufferViews") or []:
            if isinstance(view, dict) and isinstance(view.get(media.name), str):
                key = media.fold(view[media.name])
                held[key] = held.get(key, 0) + 1
        # A name two views hold rides neither (``GlbMedia.view``).
        riding = {key for key, count in held.items() if count == 1}
        index = {id(node): i for i, node in enumerate(gltf.get("nodes") or [])}
        lit: Dict[int, Dict[str, Any]] = {}
        for entry, nodes, status, _found in self._mc._media.resolved(gltf, entries):
            if not nodes or status in ("ambiguous", "absent"):
                continue
            if media.fold(entry.get("map")) not in riding:
                continue
            for node in nodes:
                lit.setdefault(index[id(node)], entry)
        return lit

    def media_findings(self, gltf: dict) -> Tuple[List[str], List[str]]:
        """``(problems, notes)``: the files *gltf*'s ``lightmaps`` record names,
        held to the media transport (``SceneRecords.GLB_MEDIA``).

        A view a named file rides must state its bytes: a map an 8-bit web
        encoding of an HDR map (``hdr_web``) with a positive ``scale``, the
        probe and each copy Radiance with none -- a reader decodes by what the
        view states. Where the record lights anything here (:meth:`lit_nodes`)
        the probe and its copies ride too, or the file is not what its record
        says; a map an object here reads that rides nothing is a note (the
        build warned of it: that object renders unlit).
        """
        media = SceneRecords.GLB_MEDIA
        record = SceneRecords.glb_block(gltf, SceneRecords.LIGHTMAPS)
        if not record:
            return [], []
        maps = {
            media.file_name(e.get("map"))
            for e in record.get("objects") or []
            if isinstance(e, dict) and e.get("map")
        }
        probe = record.get("probe") if isinstance(record.get("probe"), dict) else {}
        probes = [media.file_name(probe.get("map"))] if probe.get("map") else []
        probes += [
            media.file_name(h.get("map"))
            for h in probe.get("highlights") or []
            if isinstance(h, dict) and h.get("map")
        ]

        def a_map(extras: Dict[str, Any]) -> Optional[str]:
            mime = extras.get(media.mime_type)
            scale = extras.get(media.scale)
            if mime not in media.hdr_web:
                return (
                    f"states {mime!r}, not an 8-bit web encoding of an HDR map "
                    f"({', '.join(media.hdr_web)})"
                )
            if not (isinstance(scale, (int, float)) and scale > 0):
                return f"states no positive extras.{media.scale}"
            return None

        def a_probe(extras: Dict[str, Any]) -> Optional[str]:
            if extras.get(media.mime_type) != media.radiance or media.scale in extras:
                return f"is not Radiance ({media.radiance}, no scale)"
            return None

        checks = {**{name: a_map for name in maps}, **{name: a_probe for name in probes}}
        reading = {
            media.file_name(entry.get("map"))
            for entry, nodes, status, _found in self._mc._media.resolved(
                gltf, record.get("objects") or []
            )
            if nodes and status not in ("ambiguous", "absent")
        }
        problems, notes = self._mc._media.findings(
            gltf,
            SceneRecords.LIGHTMAPS,
            checks,
            reading,
            readers="objects",
            without="render unlit",
        )
        # Where the record lights anything here, its probe and copies ride.
        if self.lit_nodes(gltf):
            where = f"extras.{SceneRecords.LIGHTMAPS.key}"
            problems += [
                f"{where} names {name}, which rides no bufferView"
                for name in probes
                if media.view(gltf, name) is None
            ]
        return problems, notes

    def lightmap_manifest_coverage(self, glb: GlbTarget) -> Dict[str, List[str]]:
        with self._mc.open_glb(glb) as edit:
            manifest = self._lightmap_record(edit.gltf) or {}
            buckets: Dict[str, List[str]] = {
                "present": [],
                "ambiguous": [],
                "absent": [],
            }
            entries = [
                e
                for e in manifest.get("objects") or []
                if isinstance(e, dict) and e.get("name")
            ]
            for entry, _nodes, status, _found in self._mc._media.resolved(edit.gltf, entries):
                bucket = "present" if status in ("exact", "leaf") else status
                buckets[bucket].append(str(entry["name"]))
            return buckets

    @staticmethod
    def lightmap_report(
        coverage: Dict[str, List[str]], bound: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        wanted = list(coverage.get("present", [])) + list(coverage.get("ambiguous", []))
        # Counted in OBJECTS, not names: two objects can share a name -- each
        # its own record entry, told apart by the bound record's ``entry``.
        # Keyed by name alone, one bound ``BODY`` reported its unbound
        # namesake as bound too.
        per_name: Dict[str, Set[Any]] = {}
        for record in bound:
            name = str(record.get("object"))
            per_name.setdefault(name, set()).add(record.get("entry", name))
        left = {name: len(keys) for name, keys in per_name.items()}
        unbound: List[str] = []
        for name in wanted:
            if left.get(name, 0) > 0:
                left[name] -= 1
            else:
                unbound.append(name)
        return {
            "expected": len(wanted),
            "bound": len(wanted) - len(unbound),
            "unbound": unbound,
            "out_of_scope": len(coverage.get("absent", [])),
        }

    def strip_glb_baked_lights(self, glb: GlbTarget) -> List[str]:
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            if not self.lit_nodes(gltf):
                return []
            key = "KHR_lights_punctual"
            carried = []
            for index, node in enumerate(gltf.get("nodes") or []):
                extensions = node.get("extensions")
                if isinstance(extensions, dict) and key in extensions:
                    del extensions[key]
                    if not extensions:
                        del node["extensions"]
                    carried.append(str(node.get("name") or index))
            root = gltf.get("extensions")
            declared = isinstance(root, dict) and key in root
            if declared:
                del root[key]
                if not root:
                    del gltf["extensions"]
            for field in ("extensionsUsed", "extensionsRequired"):
                if key in (gltf.get(field) or []):
                    gltf[field] = [e for e in gltf[field] if e != key]
                    if not gltf[field]:
                        del gltf[field]
            if not (carried or declared):
                return []
            edit.dirty = True
            logger.info(
                "Baked lights: %d light node(s) dropped from the lightmapped GLB "
                "-- their light is already in the lightmaps (%s).",
                len(carried),
                ", ".join(carried[:8]) + (" ..." if len(carried) > 8 else ""),
            )
            return carried

    def apply_glb_lightmaps(
        self,
        glb: GlbTarget,
        search_dirs: Sequence[str] = (),
        *,
        percentile: Optional[float] = None,
        hdr_codec: Any = None,
    ) -> List[Dict[str, Any]]:
        from pythontk.img_utils._img_utils import ImgUtils

        codec = hdr_codec or ImgUtils
        media = SceneRecords.GLB_MEDIA
        identity = [1.0, 1.0, 0.0, 0.0]
        records: List[Dict[str, Any]] = []
        # ONE session for the read and the write: the record is read from the
        # very GLB being edited.
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            # A map bound through a misread record looks like a bad bake
            # rather than a version problem: another schema is refused.
            record = self._lightmap_record(gltf, warn=True)
            if not record:
                return []
            # The record as it stands, under its key in the root extras: one
            # layout, two containers (``RecordSpec.glb_verbatim``). Before
            # the files, which a pass that draws the probe renames on it.
            record = SceneRecords.put_glb_block(gltf, SceneRecords.LIGHTMAPS, record)
            edit.dirty = True
            entries = record.get("objects") or []
            if not entries:
                return []

            # The caller's search paths, in its order -- the host hands the
            # bake's own folders first (LightmapRecords.search_dirs) -- then
            # the file's own. No record names a folder.
            dirs = [d for d in search_dirs if d]
            dirs.append(os.path.dirname(os.path.abspath(edit.path)))

            #: Entries with no node in this GLB. The record is a SCENE record
            #: and every export carries all of it, so on a selection export
            #: these are simply the objects that were not selected -- counted
            #: for one summary line, never warned per object.
            out_of_scope: List[str] = []
            #: Unfindable map -> how many entries wanted it: one line per map,
            #: with the number of objects at stake (an atlas is shared by
            #: every object baked into it).
            missing: Dict[str, int] = {}
            #: map file name -> its scale once it rides, None for a failed
            #: encode: one attempt (and one warning) per map.
            scales: Dict[str, Optional[float]] = {}
            nodes_by_name = None
            meshes = gltf.get("meshes") or []

            def ride(name: str, src: Optional[str]) -> Optional[float]:
                """The record's map *name* as the file holds it: the view
                already riding it (a second pass reads it, wherever the bake's
                folders went), else *src* ridden in its web encoding on first
                use -- never for an entry no primitive of which can read it,
                which would ship dead bytes. ``None`` when it cannot ride."""
                key = media.fold(name)
                if key not in scales:
                    if media.views(gltf, name):
                        index = self._mc._media.held(gltf, name, "Lightmap")
                        stated = (
                            (gltf["bufferViews"][index].get("extras") or {})
                            if index is not None
                            else {}
                        ).get(media.scale)
                        scales[key] = (
                            float(stated)
                            if isinstance(stated, (int, float)) and stated > 0
                            else None
                        )
                        if index is not None and scales[key] is None:
                            logger.warning(
                                "Lightmap %r rides a view stating no scale; not bound.",
                                name,
                            )
                        return scales[key]
                    try:
                        data, scale = codec.encode_hdr_for_web(src, percentile)
                    except (ImportError, ValueError) as error:
                        logger.warning("Lightmap %r not encoded: %s", name, error)
                        scales[key] = None
                    else:
                        index = self._mc._media.put(
                            edit, name, data, "image/png", scale, kind="Lightmap"
                        )
                        scales[key] = None if index is None else scale
                return scales[key]

            for entry_index, (entry, nodes, status, leaves) in enumerate(
                self._mc._media.resolved(gltf, entries)
            ):
                if not isinstance(entry, dict):
                    continue  # malformed record: skipped, as the coverage report does
                name, basename = entry.get("name"), media.file_name(entry.get("map"))
                if status == "ambiguous":
                    # In the file, and genuinely unbindable: still loud.
                    logger.warning(
                        "Lightmap for %r: nothing tells its object apart among "
                        "the GLB node(s) %s and the record entries sharing "
                        "them%s -- ambiguous, not bound.",
                        name,
                        ", ".join(sorted(leaves)),
                        ""
                        if entry.get("hierarchy")
                        else " (the record predates the hierarchy key; "
                        "re-export to publish one)",
                    )
                    continue
                if status == "absent":
                    # Not in this export. Kept at debug because it is also what
                    # a genuine name mismatch looks like, and the node list is
                    # what tells the two apart.
                    out_of_scope.append(str(name))
                    if nodes_by_name is None:
                        nodes_by_name = self._mc._media.node_index(gltf)[0]
                    logger.debug(
                        "Lightmap for %r: no mesh node by that name in the GLB "
                        "(nodes: %s).",
                        name,
                        ", ".join(sorted(nodes_by_name)) or "<none>",
                    )
                    continue
                readable = []
                for node in nodes:
                    prims = meshes[node["mesh"]].get("primitives") or []
                    for prim in prims:
                        if "TEXCOORD_1" not in (prim.get("attributes") or {}):
                            logger.warning(
                                "Lightmap for %r: primitive has no TEXCOORD_1 -- "
                                "the FBX was exported without the lightmap UV "
                                "set; skipped.",
                                name,
                            )
                    if any(
                        "TEXCOORD_1" in (p.get("attributes") or {}) for p in prims
                    ):
                        readable.append(node)
                if not readable:
                    continue
                src = None
                if not media.views(gltf, basename):  # the file's own, else the bake's
                    src = self._mc._media.locate(dirs, basename)
                    if src is None:
                        missing[basename] = missing.get(basename, 0) + 1
                        continue
                    src = os.path.abspath(src)
                scale = ride(basename, src)
                if scale is None:  # encode failed, already logged
                    continue
                for _node in readable:
                    records.append(
                        {
                            "object": name,
                            "map": basename,
                            "scale": scale,
                            "scaleOffset": [
                                float(v) for v in (entry.get("scaleOffset") or identity)
                            ],
                            "entry": entry_index,
                        }
                    )

            for basename, wanted in missing.items():
                # Deliberately does NOT advise passing ``search_dirs``: every
                # shipped caller already does, so the directories listed ARE
                # the full search and the map is in none of them.
                logger.warning(
                    "Lightmap %r not found in %s -- the %d object(s) baked into "
                    "it are not bound and will render unlit.",
                    basename,
                    dirs,
                    wanted,
                )
            if records:
                # The probe lights what the bake lit: a file none of whose
                # objects the record lights ships none (a reader then lights
                # it as a file without a bake).
                self._ride_probe(edit, record, dirs, codec)
            elif len(entries) > len(out_of_scope):
                # "The scene was never baked" (a clean no-op, returned above)
                # and "the bake exists and none of it reached the file" are the
                # same empty list and wildly different deliverables -- the total
                # said once, counted over what this GLB CARRIES.
                logger.warning(
                    "Lightmaps NOT wired: %d object(s) in this GLB are baked "
                    "and none bound -- it ships unlit. Searched %s.",
                    len(entries) - len(out_of_scope),
                    dirs,
                )
            if out_of_scope:
                logger.info(
                    "Lightmap record covers %d object(s) not in this export "
                    "(scene-wide record, exported subset); ignored.",
                    len(out_of_scope),
                )
        return records

    def _ride_probe(
        self, edit: GlbEdit, record: Dict[str, Any], dirs: Sequence[str], codec: Any
    ) -> List[str]:
        """The record's probe and its highlight copies, each riding *edit* as
        Radiance bytes under its file name; the names that ride.

        Where the record calls for its lights to be drawn
        (``_Highlights._wants_highlights``), they are drawn into the probe
        and its complete copies here, on this file's own geometry, as the
        FBX deliverable's pass draws them (``_deliverable_probes``), and the
        record is renamed to name them -- so the GLB and its FBX twin carry
        the same probe. Otherwise (no light, or a probe drawn already: a GLB
        built from a finished FBX) each file the record names is found in
        *dirs* and rides as it is, re-encoded as Radiance by *codec*. A probe
        placed in no unit, found nowhere or unreadable is warned and left
        out: the file still ships its lightmaps.
        """
        media = SceneRecords.GLB_MEDIA
        highlights = self._mc._highlights
        probe = record.get("probe")
        if not isinstance(probe, dict) or not probe.get("map"):
            return []
        basename = media.file_name(probe["map"])
        if highlights._probe_in_metres(record) is None:
            logger.warning(
                "Reflection probe %r has no usable position or unit_scale; "
                "not embedded.",
                basename,
            )
            return []
        files: List[Tuple[str, bytes]] = []
        # Drawn on the file's first pass only: a probe riding already is the
        # file's own (a second pass adds nothing).
        if highlights._wants_highlights(record) and not media.views(edit.gltf, basename):
            try:
                drawn = highlights._deliverable_probes(edit, record, dirs, logger)
            except Exception as error:  # noqa: BLE001 -- never the bake's cost
                logger.warning("Probe highlights not drawn: %s", error)
                logger.debug("Probe highlights failed.", exc_info=True)
                drawn = None
            if drawn:
                # The record names the drawn files from now on (as the FBX's
                # does once its own pass ran): in place, the root extras' own.
                record["probe"] = drawn["record"]["probe"]
                files = list(drawn["files"])
                if media.file_name(record["probe"]["map"]) == basename:
                    # The capture point sees no light: the bake's own map.
                    files = [(basename, None)] + files
        if not files:
            names = [probe["map"]] + [
                h.get("map")
                for h in probe.get("highlights") or []
                if isinstance(h, dict) and h.get("map")
            ]
            files = [(media.file_name(n), None) for n in names]
        shipped: List[str] = []
        for name, data in files:
            if data is None:
                if media.views(edit.gltf, name):
                    # The file's own -- or a tie, which _Media.put names.
                    if self._mc._media.held(edit.gltf, name, "Reflection probe") is not None:
                        shipped.append(name)
                    continue
                path = self._mc._media.locate(dirs, name)
                if path is None:
                    logger.warning(
                        "Reflection probe %r not found in %s; the file ships "
                        "without it.",
                        name,
                        list(dirs),
                    )
                    continue
                try:
                    data = codec.encode_hdr_radiance(path)
                except (ImportError, ValueError, OSError) as error:
                    logger.warning("Reflection probe %r not encoded: %s", name, error)
                    continue
            if self._mc._media.put(edit, name, data, media.radiance, kind="Reflection probe") is None:
                continue
            shipped.append(name)
        return shipped
