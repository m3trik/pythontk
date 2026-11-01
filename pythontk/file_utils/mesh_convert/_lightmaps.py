# !/usr/bin/python
# coding=utf-8
"""Lightmaps: carry a host DCC's committed bake into the GLB deliverable.

Reads the bake manifest the DCC publishes, resolves each entry to the GLB's
mesh nodes, encodes the maps for the web and binds them, and keeps every
lightmap marker and manifest copy saying what the file ships.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through the facade it holds (``self._mc``).
"""

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Eager: the signatures name GlbEdit / GlbTarget; SceneRecords and
# LightmapRecord are light declaration modules every pass reads.
from pythontk.core_utils.engines.scene_export.lightmap_record import LightmapRecord
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit, GlbTarget

logger = logging.getLogger(__name__)


class _Lightmaps:
    """Lightmaps: resolve, encode, bind, and correct the manifest copies.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: Per-object bake marker, riding the same FBX user-property channel as the
    #: record (mayatk/blendertk ``LightmapBaker.LIGHTMAP_INFO_ATTR``).
    LIGHTMAP_INFO_KEY = "lightmapInfo"
    #: Joins a lightmap clone's name to the material it was cloned from. The
    #: lightmap pass makes one material per INSTANCE (each needs its own atlas
    #: rect), so a base material becomes ``<base>~lm<N>``. Named because the
    #: convention has two ends: whoever writes the clone and whoever has to
    #: recognise one -- an envelope section names the BASE materials, and a
    #: check matching clone names against it literally finds nothing and
    #: reports every clone as undescribed (measured: 46 spurious notes on one
    #: production room, from a single base material's clones).
    LIGHTMAP_CLONE_SUFFIX = "~lm"

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    def _lightmap_clone_base(self, name: str) -> str:
        """The material *name* was cloned from, or *name* when it is not a clone.

        Only a ``~lm`` followed by digits is a clone marker: the suffix is legal
        in an authored material name, and stripping at a bare ``~lm`` would
        rewrite one.
        """
        text = str(name)
        base, sep, tail = text.rpartition(self._mc.LIGHTMAP_CLONE_SUFFIX)
        return base if sep and base and tail.isdigit() else text

    def _lightmap_manifest(self, gltf: dict) -> Optional[Dict[str, Any]]:
        """The ``lightmaps`` record on a parsed glTF's carrier, or ``None``.

        The channel read is :meth:`data_export_channel`; this adds only the
        applier's own expectation that the manifest is an OBJECT, so a channel
        holding anything else reads as absent rather than reaching the walk as
        something without ``.get``.
        """
        data = self._mc.data_export_channel(gltf, SceneRecords.LIGHTMAPS.key)
        return data if isinstance(data, dict) else None

    def lightmaps_block(
        self,
        materials: Dict[str, float],
        carrier: str = "occlusion",
        probe: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        block: Dict[str, Any] = {
            "version": SceneRecords.LIGHTMAPS.glb.version,
            "carrier": carrier,
            "materials": {
                name: {"intensity": round(float(level), 6)}
                for name, level in materials.items()
            },
        }
        if probe:
            block["probe"] = dict(probe)
        return block

    def _lightmap_final_values(
        self, gltf: dict, bound: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[int, Dict[str, Any]]:
        """``{node index: {"map"}}``: each node's lightmap in this file.

        Read off the file itself -- the materials a node's primitives wear,
        looked up in *bound* (``{material name: {"map", ...}}``, what the bind
        gave each material) -- because that is what the viewer binds: a copy
        corrected to it says what the deliverable renders. A node wearing two
        different lightmaps is left out: picking one would put the other's
        lighting on record. Keyed by node INDEX, not name: FBX carries leaf
        names only, so two same-named nodes can rightly wear two maps, and
        keyed by name both read as one conflicted entry and kept the bake-time
        ``.exr``. A map's level is the block's (each material's intensity):
        the record states none.
        """
        if not isinstance(bound, dict) or not bound:
            return {}
        materials = gltf.get("materials") or []
        meshes = gltf.get("meshes") or []
        final: Dict[int, Dict[str, Any]] = {}
        for index, node in enumerate(gltf.get("nodes") or []):
            mesh = node.get("mesh")
            if not isinstance(mesh, int) or not 0 <= mesh < len(meshes):
                continue
            worn: List[Dict[str, Any]] = []
            for prim in meshes[mesh].get("primitives") or []:
                mi = prim.get("material")
                if not isinstance(mi, int) or not 0 <= mi < len(materials):
                    continue
                entry = bound.get(materials[mi].get("name"))
                if not isinstance(entry, dict):
                    continue
                pair = {k: entry[k] for k in ("map",) if k in entry}
                if pair and pair not in worn:
                    worn.append(pair)
            if len(worn) == 1:
                final[index] = worn[0]
        return final

    @staticmethod
    def _correct_lightmap_record(record: dict, committed: Optional[dict]) -> bool:
        """Give *record* the ``map`` *committed* carries.

        Returns True when it changed.
        """
        changed = False
        for field in ("map",):
            if not committed or field not in committed:
                continue
            if record.get(field) != committed[field]:
                record[field] = committed[field]
                changed = True
        return changed

    def _correct_manifest_entries(
        self, gltf: dict, manifest: dict, final: Optional[Dict[int, Dict[str, Any]]]
    ) -> bool:
        """Correct a manifest's entries from *final*; True when one changed.

        Each entry is matched to its nodes by the binder's own rule
        (:meth:`_resolve_lightmap_entries`: exact, then namespace-tolerant,
        same-named objects told apart by hierarchy), so an entry is corrected
        where the applier would bind it, and only when every node it resolves
        to ships the same lightmap. A manifest newer than this reader is left
        alone: its fields may no longer mean the same.
        """
        entries = manifest.get("objects")
        if not final or not isinstance(entries, list):
            return False
        if not SceneRecords.LIGHTMAPS.is_current(manifest):
            return False
        nodes_by_name, leaf_index, _users, lineage = self._lightmap_node_index(gltf)
        resolved = self._resolve_lightmap_entries(
            entries, nodes_by_name, leaf_index, lineage
        )
        index_of = {id(node): i for i, node in enumerate(gltf.get("nodes") or [])}
        changed = False
        for entry, (nodes, _status, _leaves) in zip(entries, resolved):
            if not isinstance(entry, dict):
                continue
            shipped = [final.get(index_of.get(id(node))) for node in nodes or ()]
            if shipped and all(v and v == shipped[0] for v in shipped):
                changed |= self._correct_lightmap_record(entry, shipped[0])
        return changed

    def _correct_record(self, gltf: dict, final: Optional[dict]) -> bool:
        """Correct the carrier's ``lightmaps`` record to what this file ships.

        The bake writes the record BEFORE the web encode exists, so its entries
        name an ``.exr`` that ships nowhere. Each entry takes the map of the
        nodes it resolves to (:meth:`_correct_manifest_entries`, from *final*
        -- ``{node index: {"map"}}``, :meth:`_lightmap_final_values`). The
        record so corrected is what the build reports back to the scene's
        record of the export (``fbx_to_glb``'s ``lightmap_manifest``, which
        both hosts' exporters keep); it then leaves the file with the other
        records and the markers (:meth:`strip_fbx_records`).

        Returns:
            Whether the record changed.
        """
        key = SceneRecords.LIGHTMAPS.key
        found = self._mc._sidecar._data_export_carrier(gltf, key)
        record = self._mc.data_export_channel(gltf, key)
        if found is None or not isinstance(record, dict):
            return False
        if not self._correct_manifest_entries(gltf, record, final):
            return False
        self._mc._sidecar._write_data_export(found[0], key, record)
        return True

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

    @staticmethod
    def _node_parents(gltf: dict) -> Dict[int, int]:
        """``{child index: parent index}`` over every node's ``children``.

        The first parent listed wins: glTF allows a node exactly one, so a
        second is a malformed file rather than a choice to honour.
        """
        parents: Dict[int, int] = {}
        for index, node in enumerate(gltf.get("nodes") or []):
            for child in (node or {}).get("children") or []:
                if isinstance(child, int):
                    parents.setdefault(child, index)
        return parents

    @staticmethod
    def _node_lineage(parents: Dict[int, int], index: int) -> List[int]:
        """*index* and its ancestors, nearest first; a cycle ends the walk."""
        chain, seen = [index], {index}
        while chain[-1] in parents and parents[chain[-1]] not in seen:
            chain.append(parents[chain[-1]])
            seen.add(chain[-1])
        return chain

    def _lightmap_node_index(self, gltf: dict):
        """Index a glTF's MESH nodes for manifest lookup.

        Returns ``(nodes_by_name, leaf_index, mesh_users, lineage)``: nodes
        grouped by their exact name, the namespace-stripped leaf of each of
        those names mapped back to the full names carrying it, how many nodes
        reference each mesh (which is what identifies an instanced mesh needing
        its own clone before a per-instance binding can land on it), and each
        mesh node's LINEAGE -- the names from its scene root down to itself,
        keyed by ``id(node)`` -- which is what tells same-named nodes apart
        (:meth:`_resolve_lightmap_entries`).
        """
        nodes = gltf.get("nodes", []) or []
        parents = self._node_parents(gltf)
        nodes_by_name: Dict[str, List[dict]] = {}
        mesh_users: Dict[int, int] = {}  # mesh index -> node reference count
        lineage: Dict[int, Tuple[str, ...]] = {}
        for index, node in enumerate(nodes):
            if "mesh" not in node:
                continue
            nodes_by_name.setdefault(node.get("name", ""), []).append(node)
            mesh_users[node["mesh"]] = mesh_users.get(node["mesh"], 0) + 1
            lineage[id(node)] = tuple(
                str(nodes[i].get("name") or "")
                for i in reversed(self._node_lineage(parents, index))
            )
        leaf_index: Dict[str, List[str]] = {}
        for full in nodes_by_name:
            leaf_index.setdefault(full.rsplit(":", 1)[-1], []).append(full)
        return nodes_by_name, leaf_index, mesh_users, lineage

    @staticmethod
    def _hierarchy_score(hierarchy: Sequence[Any], lineage: Sequence[str]) -> int:
        """How many trailing names *hierarchy* and *lineage* share.

        Compared from the object itself upward, namespace-blind, and stopped at
        the first disagreement. Only the tail can be compared: an export may
        root the file under a node the scene never had (FBX2glTF's
        ``RootNode``) or drop ancestors it was not given, but it cannot rename
        the parents it keeps.
        """
        score = 0
        for mine, theirs in zip(reversed(list(hierarchy)), reversed(list(lineage))):
            if str(mine).rsplit(":", 1)[-1] != str(theirs).rsplit(":", 1)[-1]:
                break
            score += 1
        return score

    def _resolve_lightmap_entries(
        self,
        entries: Sequence[Any],
        nodes_by_name: Dict[str, List[dict]],
        leaf_index: Dict[str, List[str]],
        lineage: Dict[int, Tuple[str, ...]],
    ) -> List[Tuple[Optional[List[dict]], str, List[str]]]:
        """Resolve every manifest entry to its GLB mesh nodes, as ONE assignment.

        One ``(nodes, status, leaves)`` per entry, aligned with *entries* (see
        :meth:`_resolve_lightmap_node` for the statuses; a malformed or
        nameless entry is ``"absent"``). Names alone settle the common case --
        one entry, one node. A scene that reuses a leaf name does not have it:
        FBX carries leaf names only, so two objects named ``BODY`` publish two
        ``BODY`` entries and arrive as two ``BODY`` nodes, and a stale marker
        on a hidden object can put a second entry on a name whose node belongs
        to another. Every set of entries whose candidate nodes OVERLAP is
        settled together -- two such sets settled apart could each bind the
        same node, and the object would wear whichever came last. Each
        (entry, node) pair ranks by the entry's ``hierarchy`` (its scene path,
        root first) against the node's lineage (:meth:`_hierarchy_score`),
        then by whether the NAME matched exactly, so an exact name still beats
        a namespace-stripped one wherever the hierarchy cannot tell them apart
        (the file carries ``PROPS_DA:prop352``, the manifest also lists
        ``PROPS_RF:prop352``). An entry binds to a node only when each is the
        other's UNIQUE best match. An entry whose best candidates all went to
        better-matching entries is ``"absent"`` -- its object is not in this
        file -- and anything still tied, or with nothing to compare, is
        ``"ambiguous"``: never guessed, because a guess puts one object's
        lighting on another. A per-entry match is never widened past the rule
        :meth:`_resolve_lightmap_node` applies.
        """
        resolved: List[Tuple[Optional[List[dict]], str, List[str]]] = []
        pools: Dict[int, List[dict]] = {}  # entry index -> its candidate nodes
        for i, entry in enumerate(entries):
            name = entry.get("name") if isinstance(entry, dict) else None
            if not name or not isinstance(name, str):
                # A malformed record never reaches the name lookup: "" is
                # where the file's NAMELESS mesh nodes are indexed.
                resolved.append((None, "absent", []))
                continue
            nodes, status, leaves = self._resolve_lightmap_node(
                name, nodes_by_name, leaf_index
            )
            resolved.append((nodes, status, leaves))
            pool = nodes or [n for full in leaves for n in nodes_by_name.get(full, [])]
            if pool:
                pools[i] = pool

        # Union-find over node ids: entries sharing any candidate land together.
        parent: Dict[int, int] = {}

        def root(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for pool in pools.values():
            first = root(id(pool[0]))
            for node in pool[1:]:
                parent[root(id(node))] = first
        groups: Dict[int, List[int]] = {}
        for i, pool in pools.items():
            groups.setdefault(root(id(pool[0])), []).append(i)

        def unique_best(pairs: List[Tuple[Any, Tuple[int, int]]]) -> Optional[Any]:
            """The key with the strictly highest rank, or None on a tie / no pairs."""
            if not pairs:
                return None
            top = max(rank for _key, rank in pairs)
            best = [k for k, rank in pairs if rank == top]
            return best[0] if len(best) == 1 else None

        for members in groups.values():
            by_id = {id(node): node for i in members for node in pools[i]}
            if len(members) == 1 and len(by_id) == 1:
                continue  # one entry, one node: the name already said it
            # (hierarchy score, exact name): an entry without a hierarchy
            # scores 0 on every node, which leaves the name to decide.
            ranks: Dict[Tuple[int, int], Tuple[int, int]] = {}
            takers: Dict[int, List[int]] = {}  # node id -> entries it could be
            for i in members:
                hierarchy = entries[i].get("hierarchy")
                usable = isinstance(hierarchy, (list, tuple)) and bool(hierarchy)
                exact = int(resolved[i][1] == "exact")
                for node in pools[i]:
                    score = (
                        self._hierarchy_score(hierarchy, lineage.get(id(node), ()))
                        if usable
                        else 0
                    )
                    ranks[(i, id(node))] = (score, exact)
                    takers.setdefault(id(node), []).append(i)
            best_node = {
                i: unique_best([(id(n), ranks[(i, id(n))]) for n in pools[i]])
                for i in members
            }
            owner = {}  # node id -> the entry it binds to
            for nid, entries_for in takers.items():
                taker = unique_best([(i, ranks[(i, nid)]) for i in entries_for])
                if taker is not None and best_node[taker] == nid:
                    owner[nid] = taker
            for i in members:
                nid = best_node[i]
                if nid is not None and owner.get(nid) == i:
                    # Settled out of a leaf pool is still a leaf match.
                    status = "exact" if resolved[i][1] == "exact" else "leaf"
                    resolved[i] = ([by_id[nid]], status, [])
                    continue
                top = max(ranks[(i, id(n))] for n in pools[i])
                if all(id(n) in owner for n in pools[i] if ranks[(i, id(n))] == top):
                    # Every node this entry could best be went to a better
                    # match: its own object is not in this file.
                    resolved[i] = (None, "absent", [])
                else:
                    # Where each candidate SITS -- the only thing that tells a
                    # reader which two objects collided.
                    paths = sorted("/".join(lineage.get(id(n), ())) for n in pools[i])
                    resolved[i] = (None, "ambiguous", paths)
        return resolved

    def _resolve_lightmap_node(
        self,
        name: Optional[str],
        nodes_by_name: Dict[str, List[dict]],
        leaf_index: Dict[str, List[str]],
    ) -> Tuple[Optional[List[dict]], str, List[str]]:
        """Resolve one manifest object name to the GLB's mesh nodes, by name alone.

        The per-entry stage of :meth:`_resolve_lightmap_entries`, which every
        reader goes through -- two readers of the match rule is how the binder
        and the report came to disagree about what "missing" means -- and which
        settles what a name alone cannot (several objects sharing it).
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
        if not name:
            return None, "absent", []  # "" indexes the file's NAMELESS nodes
        nodes = nodes_by_name.get(name)
        if nodes:
            return nodes, "exact", []
        leaves = leaf_index.get(name.rsplit(":", 1)[-1]) or []
        if len(leaves) == 1:
            return nodes_by_name[leaves[0]], "leaf", leaves
        if len(leaves) > 1:
            return None, "ambiguous", leaves
        return None, "absent", []

    def lightmap_manifest_coverage(self, glb: GlbTarget) -> Dict[str, List[str]]:
        with self._mc.open_glb(glb) as edit:
            manifest = self._mc._lightmap_manifest(edit.gltf) or {}
            nodes_by_name, leaf_index, _, lineage = self._lightmap_node_index(edit.gltf)
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
            resolved = self._resolve_lightmap_entries(
                entries, nodes_by_name, leaf_index, lineage
            )
            for entry, (_, status, _) in zip(entries, resolved):
                bucket = "present" if status in ("exact", "leaf") else status
                buckets[bucket].append(str(entry["name"]))
            return buckets

    @staticmethod
    def lightmap_report(
        coverage: Dict[str, List[str]], bound: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        wanted = list(coverage.get("present", [])) + list(coverage.get("ambiguous", []))
        # Counted in OBJECTS, not names: a record is per (object, material), and
        # two objects can share a name -- each its own manifest entry, told
        # apart by the record's ``entry``. Keyed by name alone, one bound
        # ``BODY`` reported its unbound namesake as bound too.
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

    def embed_glb_lightmap_probe(
        self,
        glb: GlbTarget,
        probe: Optional[Dict[str, Any]],
        search_dirs: Sequence[str],
        hdr_codec: Any = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(probe, dict):
            return None  # the bake captured none: nothing to embed
        with self._mc.open_glb(glb) as edit:
            block = SceneRecords.glb_block(edit.gltf, SceneRecords.LIGHTMAPS)
            if block is None:
                return None
            embedded = block.get("probe")
            if isinstance(embedded, dict) and isinstance(
                embedded.get("bufferView"), int
            ):
                # Embedded already (an export that runs it twice): a second
                # copy would only orphan the first's bytes.
                return embedded
            entry = self._embed_lightmap_probe(
                edit,
                probe,
                probe.get("unit_scale"),
                list(search_dirs),
                hdr_codec=hdr_codec,
            )
            if not entry:
                return None
            block["probe"] = entry  # the block itself: the file's root extras
            edit.dirty = True
            return entry

    def _embed_lightmap_probe(
        self,
        edit: GlbEdit,
        probe: Optional[Dict[str, Any]],
        unit_scale: Optional[float],
        dirs: Sequence[str],
        hdr_codec: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Embed a bake's reflection *probe*, its lengths *unit_scale* metres
        per unit; its ``lightmaps`` entry, or ``None``.

        The probe is the room the bake lit, captured once as an HDR from one
        point (a host's ``LightmapBaker``): what a lightmap cannot hold -- the
        reflections on every baked surface, which are the whole look of a
        metal, and the light on every unbaked one. Embedded whole, as Radiance
        bytes in a bufferView of its own (:meth:`ImgUtils.encode_hdr_radiance`):
        an 8-bit map would clip exactly the lights a reflection shows. The
        manifest names the view by index, which is also what keeps it through
        BIN compaction (:meth:`GlbEdit.compact_bin` keeps every view the JSON
        names). The capture point and the box its reflections project onto
        arrive in scene units with ``unit_scale`` beside them; the entry
        carries them in metres, the GLB's own space.

        A probe that cannot be found or read is warned and left out: the file
        still ships its lightmaps, and a viewer lights it as before. The bytes
        come from *hdr_codec*'s ``encode_hdr_radiance`` (default
        :class:`ImgUtils`; see :meth:`apply_glb_lightmaps`).
        """
        if not isinstance(probe, dict) or not probe.get("map"):
            return None
        basename = os.path.basename(str(probe["map"]))
        try:
            unit = float(unit_scale if unit_scale is not None else 1.0)
        except (TypeError, ValueError):
            unit = 0.0
        position = probe.get("position")
        if not (
            unit > 0.0 and isinstance(position, (list, tuple)) and len(position) == 3
        ):
            logger.warning(
                "Reflection probe %r has no usable position or unit_scale; "
                "not embedded.",
                basename,
            )
            return None
        path = next(
            (
                os.path.join(d, basename)
                for d in dirs
                if os.path.isfile(os.path.join(d, basename))
            ),
            None,
        )
        if path is None:
            logger.warning(
                "Reflection probe %r not found in %s; the file ships without it.",
                basename,
                list(dirs),
            )
            return None
        from pythontk.img_utils._img_utils import ImgUtils

        try:
            raw = (hdr_codec or ImgUtils).encode_hdr_radiance(path)
        except (ImportError, ValueError, OSError) as error:
            logger.warning("Reflection probe %r not encoded: %s", basename, error)
            return None
        views = self._mc._append_bin_views(edit, [raw])
        if not views:
            logger.warning(
                "Reflection probe %r not embedded: the GLB's buffer is external.",
                basename,
            )
            return None
        # Radiance HDR is the one probe encoding: the entry says where its
        # bytes are and where the probe stands, nothing a reader does not read.
        entry: Dict[str, Any] = {
            "bufferView": views[0],
            "position": [round(float(c) * unit, 6) for c in position],
            "box": None,
        }
        box = LightmapRecord.box(probe.get("box"))
        if box:
            entry["box"] = {
                corner: [round(c * unit, 6) for c in box[corner]]
                for corner in ("min", "max")
            }
        return entry

    def strip_glb_baked_lights(self, glb: GlbTarget) -> List[str]:
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            if not SceneRecords.glb_block(gltf, SceneRecords.LIGHTMAPS):
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
        carrier: str = "occlusion",
        percentile: Optional[float] = None,
        replace_authored: bool = True,
        hdr_codec: Any = None,
    ) -> List[Dict[str, Any]]:
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
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            # A map bound through a misread record looks like a bad bake
            # rather than a version problem: another schema is refused.
            manifest = self._mc._current_record(gltf, SceneRecords.LIGHTMAPS)
            if not manifest:
                return []
            entries = manifest.get("objects") or []
            if not entries:
                return []

            # The caller's search paths, in its order -- the host hands the
            # bake's own folders first (LightmapRecords.search_dirs) -- then
            # the file's own. No record names a folder: since 0.11.0 the maps'
            # folders ride the private ``lightmap_dirs``, and a basename the
            # workspace's texture folder also holds (an atlas of an earlier
            # bake) is found in the bake's folder before it.
            dirs = [d for d in search_dirs if d]
            dirs.append(os.path.dirname(os.path.abspath(edit.path)))

            nodes_by_name, leaf_index, mesh_users, lineage = self._lightmap_node_index(
                gltf
            )
            resolutions = self._resolve_lightmap_entries(
                entries, nodes_by_name, leaf_index, lineage
            )
            #: Manifest entries with no node in this GLB. The manifest is a
            #: SCENE record and every export carries all of it, so on a
            #: selection export these are simply the objects that were not
            #: selected -- counted for one summary line, never warned per
            #: object (which turned a correct 3-object push of a 50-object
            #: scene into 47 warnings saying it shipped unlit).
            out_of_scope: List[str] = []

            # exr abspath -> ``(png bytes, scalar)``, ``None`` for a failed encode:
            # one attempt (and one warning) per map, however many objects share it.
            encodes: Dict[str, Optional[tuple]] = {}
            # exr abspath -> encode scalar, recorded as the map is EMBEDDED.
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
            claimed: Dict[int, str] = {}  # material index -> exr abspath
            # Source material indices whose authored carrier map was already
            # reported (displaced, or kept under replace_authored=False) -- so the
            # warning fires once however many instances or primitives share it.
            dropped_authored: Set[int] = set()
            # Each material's carrier slot AS AUTHORED, read the first time the
            # material is touched. A binding in place overwrites that slot, and
            # read afterwards the lightmap itself would pass for an authored map
            # -- warned as displaced on every copy made from the material, or,
            # under replace_authored=False, keeping that copy unlit.
            authored_before: Dict[int, Optional[dict]] = {}

            def _authored(mi: int) -> Optional[dict]:
                if mi not in authored_before:
                    authored_before[mi] = _authored_carrier(gltf["materials"][mi])
                return authored_before[mi]

            # Materials already reported as shared by objects wearing different
            # lightmaps (once each; see the copy site).
            shared_reported: Set[int] = set()
            web_materials: Dict[str, Dict[str, Any]] = {}
            # (material, map, rect) -> the material copy binding it.
            clones: Dict[tuple, int] = {}
            used_transform = False

            located: Dict[Optional[str], Optional[str]] = {}

            def _locate(basename: Optional[str]) -> Optional[str]:
                """*basename*'s first hit in *dirs*, or ``None`` -- once per map."""
                if basename not in located:
                    located[basename] = next(
                        (
                            p
                            for d in dirs
                            for p in [os.path.join(d, basename or "")]
                            if basename and os.path.isfile(p)
                        ),
                        None,
                    )
                return located[basename]

            def _encode(src: str) -> Optional[tuple]:
                """*src* encoded for the web -- once per map."""
                if src not in encodes:
                    try:
                        encodes[src] = (hdr_codec or ImgUtils).encode_hdr_for_web(
                            src, percentile
                        )
                    except (ImportError, ValueError) as error:
                        logger.warning(
                            "Lightmap %r not encoded: %s", os.path.basename(src), error
                        )
                        encodes[src] = None
                return encodes[src]

            def _lights(entry: dict) -> bool:
                """Whether *entry*'s map is found AND encodes -- a map that
                cannot be read lights nothing, like one that cannot be found."""
                found = _locate(entry.get("map"))
                return bool(found) and _encode(os.path.abspath(found)) is not None

            # Material index -> how many primitives wearing it this bind leaves
            # UNLIT: a node no bindable entry resolves to (never baked, or its
            # map is missing or unreadable), or a primitive with no lightmap UV
            # set. A bind
            # IN PLACE lights every primitive wearing the material, and a bake
            # is per object -- so a prop the bake never saw, sharing the wall's
            # material, wore the wall's lighting, while every report counted it
            # unbaked. Such a material is bound through a copy per baked object
            # instead (the clone path below), and stays as authored for the rest.
            lit_nodes = {
                id(node)
                for entry, (nodes, status, _leaves) in zip(entries, resolutions)
                if isinstance(entry, dict)
                and status not in ("ambiguous", "absent")
                and _lights(entry)
                for node in nodes
            }
            unlit_users: Dict[int, int] = {}
            for node in gltf.get("nodes") or []:
                mesh = node.get("mesh") if isinstance(node, dict) else None
                if not isinstance(mesh, int) or not 0 <= mesh < len(gltf["meshes"]):
                    continue
                for prim in gltf["meshes"][mesh].get("primitives", []):
                    mi = prim.get("material")
                    if mi is not None and (
                        id(node) not in lit_nodes
                        or "TEXCOORD_1" not in (prim.get("attributes") or {})
                    ):
                        unlit_users[mi] = unlit_users.get(mi, 0) + 1
            for entry_index, (entry, (nodes, status, leaves)) in enumerate(
                zip(entries, resolutions)
            ):
                if not isinstance(entry, dict):
                    continue  # malformed record: skipped, as the coverage report does
                name, basename = entry.get("name"), entry.get("map")
                rect = [float(v) for v in (entry.get("scaleOffset") or identity)]
                has_rect = rect != identity
                if status == "ambiguous":
                    # In the file, and genuinely unbindable: still loud.
                    logger.warning(
                        "Lightmap for %r: nothing tells its object apart among "
                        "the GLB node(s) %s and the manifest entries sharing "
                        "them%s -- ambiguous, not bound.",
                        name,
                        ", ".join(sorted(leaves)),
                        ""
                        if entry.get("hierarchy")
                        else " (the manifest predates the hierarchy key; "
                        "re-export to publish one)",
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
                src = _locate(basename)
                if src is None:
                    missing[basename] = missing.get(basename, 0) + 1
                    continue
                src = os.path.abspath(src)

                png_name = os.path.splitext(basename)[0] + ".png"

                def _scalar(src=src, png_name=png_name):
                    """Embed *src* on FIRST use (never for an entry whose
                    primitives all fail the guards -- an orphan texture otherwise)."""
                    if src not in scalars:
                        encoded = _encode(src)
                        scalars[src] = None if encoded is None else encoded[1]
                        if encoded is not None:
                            self._mc._images._embed_image_bytes(
                                edit, src, encoded[0], name=png_name, clamp=True
                            )
                            encodes[src] = (None, encoded[1])  # bytes are in
                    return scalars[src]

                for node in nodes:
                    # This object needs a binding of its OWN wherever its map
                    # cannot ride the material in place: a per-instance rect, or
                    # a material another object's DIFFERENT map already claimed
                    # -- a secondary material two baked objects share (one GLASS
                    # on two machine bodies), or a Per-Object bake of instances,
                    # which FBX2glTF keeps as one mesh and one material behind
                    # every instance node. That used to be refused ("atlas
                    # packing prevents this" -- it does not prevent either), and
                    # the object wore the first claimant's lighting. Likewise a
                    # material an object this bind leaves unlit wears too
                    # (``unlit_users``), or that object wears this one's.
                    own = has_rect or any(
                        claimed.get(p.get("material"), src) != src
                        or unlit_users.get(p.get("material"))
                        for p in gltf["meshes"][node["mesh"]].get("primitives", [])
                        if p.get("material") is not None
                        and "TEXCOORD_1" in (p.get("attributes") or {})
                    )
                    if own and mesh_users.get(node["mesh"], 0) > 1:
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

                        if (
                            has_rect
                            or claimed.get(mi, src) != src
                            or unlit_users.get(mi)
                        ):
                            # The binding is this object's alone, and the material
                            # is shared -- so it rides a material CLONE: a rect as
                            # a glTF-standard KHR_texture_transform (any compliant
                            # viewer applies it with no custom code), a map the
                            # shared material cannot carry as the clone's own slot.
                            # The clone is JSON only (same shader inputs, same
                            # embedded texture index).
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
                            # the count are the finding. Read AS AUTHORED: a
                            # material another object bound in place carries a
                            # lightmap in the slot by now, not an authored map.
                            if _authored(mi) is not None:
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
                            if not has_rect and mi not in shared_reported:
                                shared_reported.add(mi)
                                if unlit_users.get(mi):
                                    logger.info(
                                        "Material %r is also worn by %d primitive(s) "
                                        "this bake leaves unlit; the baked objects "
                                        "bind a copy of it (one per map), so they "
                                        "keep it as authored.",
                                        base_name,
                                        unlit_users[mi],
                                    )
                                else:
                                    logger.info(
                                        "Material %r is worn by objects baked into "
                                        "different lightmaps; each map after the "
                                        "first binds its own copy of it.",
                                        base_name,
                                    )
                            scalar = _scalar()
                            if scalar is None:  # encode failed, already logged
                                continue
                            # One copy per (material, map, rect): objects baked
                            # into the same patch of the same map light alike, so
                            # they share it -- a 46-piece room on one material
                            # made 46 identical copies.
                            key = (mi, src, tuple(rect))
                            if key not in clones:
                                clone = copy.deepcopy(base)
                                clone["name"] = (
                                    f"{base_name}{self._mc.LIGHTMAP_CLONE_SUFFIX}"
                                    f"{len(gltf['materials'])}"
                                )
                                binding: Dict[str, Any] = {
                                    "index": edit.embedded[src],
                                    "texCoord": 1,
                                }
                                if has_rect:
                                    g_rect = ImgUtils.flip_rect_v(rect)
                                    binding["extensions"] = {
                                        "KHR_texture_transform": {
                                            "offset": [g_rect[2], g_rect[3]],
                                            "scale": [g_rect[0], g_rect[1]],
                                        }
                                    }
                                    used_transform = True
                                clone[slot] = binding
                                gltf["materials"].append(clone)
                                clones[key] = len(gltf["materials"]) - 1
                                web_materials[clone["name"]] = {
                                    "map": png_name,
                                    "intensity": round(scalar, 6),
                                }
                            prim["material"] = clones[key]
                            records.append(
                                {
                                    "material": gltf["materials"][clones[key]]["name"],
                                    "object": name,
                                    "map": basename,
                                    "intensity": scalar,
                                    "scaleOffset": rect,
                                    "entry": entry_index,
                                }
                            )
                            continue

                        material = gltf["materials"][mi]
                        if mi not in claimed and _authored(mi):
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
                        records.append(
                            {
                                "material": mat_name,
                                "object": name,
                                "map": basename,
                                "intensity": scalar,
                                "scaleOffset": list(identity),
                                "entry": entry_index,
                            }
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
                block = self._mc.lightmaps_block(
                    {name: bound["intensity"] for name, bound in web_materials.items()},
                    carrier,
                )
                unit = manifest.get("unit_scale")
                probe = self._embed_lightmap_probe(
                    edit, manifest.get("probe"), unit, dirs, hdr_codec=hdr_codec
                )
                if probe:
                    # The bake's lights reach the file as highlights drawn into
                    # copies of the probe, never as lights (_highlights.py).
                    highlights = self._mc._highlights._embed_highlight_probes(
                        edit, probe, list(web_materials), manifest.get("lights"), unit
                    )
                    if highlights:
                        probe["highlights"] = highlights
                    block["probe"] = probe
                block = SceneRecords.put_glb_block(
                    edit.gltf, SceneRecords.LIGHTMAPS, block
                )
                # The record's entries this run bound still name the .exr at
                # the pre-normalisation intensity: corrected from what the bind
                # gave each material, read back through each node's materials.
                self._mc._correct_record(
                    edit.gltf, self._mc._lightmap_final_values(edit.gltf, web_materials)
                )
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
