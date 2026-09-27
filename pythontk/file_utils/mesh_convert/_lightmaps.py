# !/usr/bin/python
# coding=utf-8
"""Lightmaps: carry a host DCC's committed bake into the GLB deliverable.

Reads the bake manifest the DCC publishes, resolves each entry to the GLB's
mesh nodes, encodes the maps for the web and binds them, and keeps every
lightmap marker and manifest copy saying what the file ships.

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
from pythontk.file_utils.mesh_convert.glb.edit import GlbTarget

logger = logging.getLogger(__name__)


class _LightmapsMixin:
    """Lightmaps: resolve, encode, bind, and correct the manifest copies.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: FBX user-property key the host DCCs publish their lightmap manifest under
    #: (mayatk/blendertk ``LightmapBaker.LIGHTMAP_METADATA``); arrives in the GLB
    #: as node extras via FBX2glTF's ``--user-properties``.
    LIGHTMAP_METADATA_KEY = SceneRecords.LIGHTMAPS.key
    #: Highest ``lightmap_metadata`` schema this applier knows how to read.
    LIGHTMAP_METADATA_VERSION = SceneRecords.LIGHTMAPS.version
    #: Extras key the web viewer reads (``preview/viewer.html``): the first
    #: scene's extras, then the root's (:meth:`_lightmap_web_manifest`).
    LIGHTMAP_WEB_KEY = "lightmap_web"
    #: Per-object bake marker, riding the same FBX user-property channel as the
    #: manifest (mayatk/blendertk ``LightmapBaker.LIGHTMAP_INFO_ATTR``). Carries
    #: its own copy of the locate hint, once per lightmapped object.
    LIGHTMAP_INFO_KEY = "lightmapInfo"
    #: The publisher's absolute authoring directory for the baked maps -- a
    #: BUILD-TIME hint for :meth:`apply_glb_lightmaps` to find the EXRs, with no
    #: reader once they are embedded. Stripped on the way out (see
    #: :meth:`_reconcile_node_markers`); it is machine-specific, so shipping it leaks
    #: the authoring drive layout and tells the recipient nothing they can use.
    LOCATE_HINT_KEY = "dir"
    #: EVERY build-time locate hint, so a scrub cannot know about one and
    #: miss another. ``dirs`` (plural) joined it when a single folder proved
    #: too weak a hint -- a scene with maps in two places published nothing
    #: and the consumer then found a map by basename alone, binding a stale
    #: atlas. Only a manifest written before 0.11.0 carries them (ABSOLUTE
    #: authoring paths): since then the manifest names no folder at all -- the
    #: maps are embedded in the GLB, and the host hands the build where they
    #: live (``lightmap_dirs``) from its own scene state. Either way no hint
    #: survives into a GLB.
    LOCATE_HINT_KEYS = ("dir", "dirs")
    #: Joins a lightmap clone's name to the material it was cloned from. The
    #: lightmap pass makes one material per INSTANCE (each needs its own atlas
    #: rect), so a base material becomes ``<base>~lm<N>``. Named because the
    #: convention has two ends: whoever writes the clone and whoever has to
    #: recognise one -- an envelope section names the BASE materials, and a
    #: check matching clone names against it literally finds nothing and
    #: reports every clone as undescribed (measured: 46 spurious notes on one
    #: production room, from a single base material's clones).
    LIGHTMAP_CLONE_SUFFIX = "~lm"

    @classmethod
    def _lightmap_clone_base(cls, name: str) -> str:
        """The material *name* was cloned from, or *name* when it is not a clone.

        Only a ``~lm`` followed by digits is a clone marker: the suffix is legal
        in an authored material name, and stripping at a bare ``~lm`` would
        rewrite one.
        """
        text = str(name)
        base, sep, tail = text.rpartition(cls.LIGHTMAP_CLONE_SUFFIX)
        return base if sep and base and tail.isdigit() else text

    @classmethod
    def _lightmap_manifest(cls, gltf: dict) -> Optional[Dict[str, Any]]:
        """The ``lightmap_metadata`` manifest in a parsed glTF, or ``None``.

        The channel read is :meth:`data_export_channel`; this adds only the
        applier's own expectation that the manifest is an OBJECT, so a channel
        holding anything else reads as absent rather than reaching the walk as
        something without ``.get``.
        """
        data = cls.data_export_channel(gltf, cls.LIGHTMAP_METADATA_KEY)
        return data if isinstance(data, dict) else None

    @classmethod
    def _lightmap_web_manifest(cls, gltf: dict) -> Optional[Dict[str, Any]]:
        """The ``lightmap_web`` manifest in a parsed glTF, or ``None``.

        Probed where the viewer probes (``readExtras`` in
        ``preview/viewer.html``), in its order: the first scene's extras --
        where a native DCC export writes it, as a JSON string -- then the
        root's, where the appliers here write an object. Read from the root
        alone, a natively exported deliverable looked unlit to every reader but
        the viewer.
        """
        scenes = gltf.get("scenes") or []
        first = scenes[0] if scenes and isinstance(scenes[0], dict) else {}
        for holder in (first.get("extras"), gltf.get("extras")):
            raw = holder.get(cls.LIGHTMAP_WEB_KEY) if isinstance(holder, dict) else None
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except ValueError:
                    continue  # the viewer keeps looking too
            if isinstance(raw, dict):
                return raw
        return None

    @classmethod
    def _lightmap_final_values(
        cls, gltf: dict, web: Optional[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """``{node index: {"map", "intensity"}}``: each node's lightmap in this file.

        Read off the file itself -- the materials a node's primitives wear,
        looked up in *web* (the ``lightmap_web`` manifest, keyed by material) --
        because that is what the viewer binds: a copy corrected to it says what
        the deliverable renders, whoever produced the file. A node wearing two
        different lightmaps is left out: a marker names one map, and picking one
        would put the other's lighting on record. Keyed by node INDEX, not name:
        FBX carries leaf names only, so two same-named nodes can rightly wear two
        maps, and keyed by name both read as one conflicted entry and kept the
        bake-time ``.exr`` at intensity 1.0.
        """
        published = (web or {}).get("materials")
        if not isinstance(published, dict) or not published:
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
                entry = published.get(materials[mi].get("name"))
                if not isinstance(entry, dict):
                    continue
                pair = {k: entry[k] for k in ("map", "intensity") if k in entry}
                if pair and pair not in worn:
                    worn.append(pair)
            if len(worn) == 1:
                final[index] = worn[0]
        return final

    @staticmethod
    def _correct_lightmap_record(record: dict, committed: Optional[dict]) -> bool:
        """Give *record* the ``map`` and ``intensity`` *committed* carries.

        Returns True when either value changed.
        """
        changed = False
        for field in ("map", "intensity"):
            if not committed or field not in committed:
                continue
            if record.get(field) != committed[field]:
                record[field] = committed[field]
                changed = True
        return changed

    @classmethod
    def _correct_manifest_entries(
        cls, gltf: dict, manifest: dict, final: Optional[Dict[int, Dict[str, Any]]]
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
        try:
            version = int(manifest.get("version", -1))
        except (TypeError, ValueError):
            return False
        if not 0 < version <= cls.LIGHTMAP_METADATA_VERSION:
            return False
        nodes_by_name, leaf_index, _users, lineage = cls._lightmap_node_index(gltf)
        resolved = cls._resolve_lightmap_entries(
            entries, nodes_by_name, leaf_index, lineage
        )
        index_of = {id(node): i for i, node in enumerate(gltf.get("nodes") or [])}
        changed = False
        for entry, (nodes, _status, _leaves) in zip(entries, resolved):
            if not isinstance(entry, dict):
                continue
            shipped = [final.get(index_of.get(id(node))) for node in nodes or ()]
            if shipped and all(v and v == shipped[0] for v in shipped):
                changed |= cls._correct_lightmap_record(entry, shipped[0])
        return changed

    @classmethod
    def without_locate_hints(cls, data_export: Dict[str, Any]) -> Dict[str, Any]:
        """Copy of a ``data_export`` snapshot with build-time locate hints removed.

        The DCC-side counterpart to :meth:`_reconcile_node_markers`, which does the
        same job on a parsed glTF. Both mayatk and blendertk write an export
        sidecar straight from a ``DataNodes.dump(decode=True)`` snapshot, and
        that snapshot carries :attr:`LOCATE_HINT_KEY` -- an absolute authoring
        directory that resolves nowhere but the machine that baked the maps, and
        so discloses that machine's drive layout to whoever receives the file.

        Lives here rather than in either DCC package because it is the same rule
        as the glTF-side scrub applied to a different container: keeping the key
        name, the channel name and the removal in one place is what stops a
        second hint key from being added to two of the three and missed in the
        third.

        Never mutates: the caller's snapshot is a dump it may still be using,
        and a scrub performed for *serialization* has no business editing it.
        A snapshot that actually carries a hint therefore comes back as a new
        dict; one that carries none is returned as-is, since copying a payload
        with nothing to strip would churn every clean export for no benefit.
        Either way the result is meant to be read, not edited in place.
        """
        section = data_export.get(cls.LIGHTMAP_METADATA_KEY)
        # A snapshot taken WITHOUT decode=True leaves the manifest a JSON string;
        # no production caller does that, and re-serializing one here would
        # silently change the shape the sidecar records, so leave it alone.
        if not isinstance(section, dict) or not any(
            k in section for k in cls.LOCATE_HINT_KEYS
        ):
            # Nothing to strip: hand the snapshot straight back rather than
            # churning a copy of a payload that is already clean (pinned by
            # test_without_locate_hints_passes_clean_snapshots_through). Note
            # this is the one path whose result is the caller's own object --
            # safe because the contract is only that this never MUTATES the
            # snapshot, and treating the return as read-only satisfies both.
            return data_export
        scrubbed = dict(data_export)
        scrubbed[cls.LIGHTMAP_METADATA_KEY] = {
            k: v for k, v in section.items() if k not in cls.LOCATE_HINT_KEYS
        }
        return scrubbed

    @classmethod
    def _reconcile_node_markers(cls, gltf: dict, final: dict = None) -> int:
        """Correct the lightmap markers and manifest, and drop their build-time hints.

        Call once the maps are embedded: the hint's only reader is the applier's
        own EXR lookup, so after that it is dead weight that ships an absolute
        authoring path to whoever receives the file. Both carriers are scrubbed --
        the scene-wide manifest and every per-object marker (which holds its own
        copy, so a room with 46 lightmapped objects shipped 46 more).

        Surgical by design: only this one key goes, leaving ``map``/``uv_set``/
        ``intensity``/``scaleOffset`` intact, so a consumer reading the markers
        keeps everything it can actually act on. A basename alone stays resolvable
        against ``search_dirs`` or the GLB's own directory.

        Those KEPT values are also corrected here, from *final* — the map name
        and intensity each node ships in this file, keyed by node index. They are
        written by the DCC bake pass BEFORE the web encode exists, so they name an
        ``.exr`` that ships nowhere and an intensity of 1.0 that predates
        normalisation, while ``extras.lightmap_web`` carries the embedded PNG and
        the scalar restoring the bake range. Measured on a client hand-off: 13.65625
        against 1.0, i.e. a consumer trusting a marker rendered the bake ~13.7x too
        dark, and the markers are what a reader finds FIRST (they sit next to the
        mesh). Correcting beats deleting: the applier reads these markers to locate
        the EXR, and the surviving keys are documented as a consumer contract — the
        defect is that they were stale, not that they exist.

        Both jobs ride the ONE walk because both rewrite the same wrapped-JSON
        markers; splitting them would duplicate the unwrap/re-serialize dance that
        every reader of this structure has to get right.

        Args:
            gltf: Parsed glTF, mutated in place.
            final: Optional ``{node index: {"map": str, "intensity": float}}`` -- what
                each node's lightmap is in this file (:meth:`_lightmap_final_values`).
                A marker takes its node's values, a manifest entry those of the
                nodes it resolves to (:meth:`_correct_manifest_entries`). Omitted,
                the carriers are only stripped.

        Returns:
            int: number of carriers changed (0 when there was nothing to do).
        """
        stripped = 0
        for index, node in enumerate(gltf.get("nodes", []) or []):
            extras = node.get("extras") or {}
            # TWO on-disk shapes, both real. FBX2glTF nests user properties under
            # extras.fromFBX.userProperties (the Maya route); blendertk's native
            # glTF export writes them as TOP-LEVEL node extras, verified on a real
            # deliverable. Walking only the first silently skipped every
            # Blender-authored GLB — and this is public API the preview server
            # points at whatever GLB it is given, so the shape is not knowable
            # from the call site.
            for props in (
                (extras.get("fromFBX") or {}).get("userProperties") or {},
                extras,
            ):
                for key in (cls.LIGHTMAP_METADATA_KEY, cls.LIGHTMAP_INFO_KEY):
                    entry = props.get(key)
                    if entry is None:
                        continue
                    wrapped = isinstance(entry, dict) and "value" in entry
                    raw = entry.get("value") if wrapped else entry
                    # Only the JSON-string form can hide a hint; anything else is
                    # either already a dict or not ours to rewrite.
                    try:
                        data = json.loads(raw) if isinstance(raw, str) else raw
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(data, dict):
                        continue
                    # Correct the stale values first, then drop the hint. A marker
                    # takes its OWN node's values, a manifest entry those of the
                    # nodes it resolves to; either way a record this file does not
                    # bind is left exactly as found rather than guessed at.
                    if key == cls.LIGHTMAP_METADATA_KEY:
                        corrected = cls._correct_manifest_entries(gltf, data, final)
                    else:
                        corrected = cls._correct_lightmap_record(
                            data, (final or {}).get(index)
                        )
                    had_hint = any(k in data for k in cls.LOCATE_HINT_KEYS)
                    if not had_hint and not corrected:
                        continue
                    for hint_key in cls.LOCATE_HINT_KEYS:
                        data.pop(hint_key, None)
                    # Re-serialize in the shape it arrived in, so a reader that does
                    # not know about this scrub sees no structural change.
                    new = json.dumps(data) if isinstance(raw, str) else data
                    if wrapped:
                        entry["value"] = new
                    else:
                        props[key] = new
                    stripped += 1
        return stripped

    @classmethod
    def read_glb_lightmap_manifest(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """The ``lightmap_metadata`` manifest riding a GLB's node extras, or ``None``.

        The manifest travels **in-band**: the host DCC publishes it as a string user
        property on its ``data_export`` carrier node, Maya's FBX exporter writes it as
        an FBX user property, and FBX2glTF's ``--user-properties`` (always passed by
        :meth:`fbx_to_glb`) transcribes it into that node's glTF extras as
        ``extras.fromFBX.userProperties.<key>`` -- probe-verified against v0.13.1.
        So no consumer has to pass anything; the deliverable feeds its own repair.

        A GLB that never made the FBX hop carries the same key as a TOP-LEVEL node
        extra instead (a native glTF export's custom properties), and both shapes
        are read -- the marker walk knows both, and a probe that knew only one
        would make this whole path a silent no-op on half the deliverables.
        Returns ``None`` for a GLB carrying neither, which is a clean no-op for
        every caller.
        """
        with cls.open_glb(glb) as edit:
            return cls._lightmap_manifest(edit.gltf)

    @classmethod
    def fix_glb_lightmap_metadata(cls, glb: GlbTarget) -> int:
        """Make every lightmap copy inside a GLB say what the GLB ships.

        A GLB carries its lightmap facts three times: ``lightmap_web`` -- what
        the viewer binds, and the authority -- the per-node ``lightmapInfo``
        markers, and the ``lightmap_metadata`` manifest on the data_export
        carrier. The bake writes the last two BEFORE the web encode exists, so
        they name an ``.exr`` that ships nowhere at the pre-normalisation
        intensity, with the absolute authoring folder as a locate hint.
        :meth:`apply_glb_lightmaps` corrects them as it binds; this is the same
        repair for a GLB it never touched -- blendertk's native glTF export,
        whose ``floor`` shipped ``.exr`` @ 1.0 beside a ``.png`` @ 3.19
        (measured in Blender 5.1).

        Self-feeding, like the applier: the file names the material each node
        wears and each material's lightmap, so no caller passes anything. A
        record is corrected only where the file says unambiguously what it
        ships (:meth:`_lightmap_final_values`), and the hints leave every
        carrier. A GLB with no ``lightmap_web`` is left as found: nothing is
        embedded, so nothing is authoritative and the hints may still be needed.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.

        Returns:
            How many carriers changed; 0 when the file already agrees.
        """
        with cls.open_glb(glb) as edit:
            web = cls._lightmap_web_manifest(edit.gltf)
            if not web:
                return 0
            changed = cls._reconcile_node_markers(
                edit.gltf, cls._lightmap_final_values(edit.gltf, web)
            )
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

    @classmethod
    def _lightmap_node_index(cls, gltf: dict):
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
        parents = cls._node_parents(gltf)
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
                for i in reversed(cls._node_lineage(parents, index))
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

    @classmethod
    def _resolve_lightmap_entries(
        cls,
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
            nodes, status, leaves = cls._resolve_lightmap_node(
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
                        cls._hierarchy_score(hierarchy, lineage.get(id(node), ()))
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

    @classmethod
    def _resolve_lightmap_node(
        cls,
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

    @classmethod
    def lightmap_manifest_coverage(cls, glb: GlbTarget) -> Dict[str, List[str]]:
        """Split a GLB's bake manifest by whether this GLB actually carries each object.

        The manifest is a **scene** record -- the bake commits to the scene, and
        every export of any subset of it carries the whole thing -- so the
        manifest's length is not the number of objects a given deliverable was
        supposed to light. Reading it as one makes a selection export report
        every unselected object as unlit, which is a false alarm raised
        precisely when the deliverable is correct.

        Returns ``{"present", "ambiguous", "absent"}`` name lists -- one item
        per ENTRY, so a name two objects share is listed once per object --
        where *present* is in scope and bindable, *ambiguous* is in scope but
        matched several nodes nothing tells apart (a real failure), and
        *absent* has no node in this GLB at all (out of scope -- or, if it was
        meant to be exported, a name mismatch this cannot tell apart from a
        scope boundary). Resolved by the binder's own rule
        (:meth:`_resolve_lightmap_entries`).
        """
        with cls.open_glb(glb) as edit:
            manifest = cls._lightmap_manifest(edit.gltf) or {}
            nodes_by_name, leaf_index, _, lineage = cls._lightmap_node_index(edit.gltf)
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
            resolved = cls._resolve_lightmap_entries(
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
        """``{"expected", "bound", "unbound", "out_of_scope"}`` for one bind.

        *coverage* is :meth:`lightmap_manifest_coverage` read BEFORE the bind
        (a miss leaves no record, so only the manifest can say what was
        wanted), *bound* what :meth:`apply_glb_lightmaps` returned. Scoped to
        this GLB: the bake manifest is a SCENE record every export carries
        whole, and counting it against a selection reported every unselected
        object as unlit. An ambiguous leaf stays in scope -- it IS in the file
        and did not bind. ``out_of_scope`` is kept rather than dropped: "3 of 3
        lit" over a 50-object scene is only reassuring once you can see the
        other 47 were never in the export.
        """
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

    @classmethod
    def apply_glb_lightmaps(
        cls,
        glb: GlbTarget,
        search_dirs: Sequence[str] = (),
        carrier: str = "occlusion",
        percentile: Optional[float] = None,
        replace_authored: bool = True,
    ) -> List[Dict[str, Any]]:
        """Wire a host DCC's committed lightmaps into a GLB for the web viewer.

        The GLB consumer half of the scene-state contract: the bake tool commits to
        the *scene* (per-shape markers -> the ``lightmap_metadata`` manifest on the
        FBX ``data_export`` carrier), and this reads that manifest back **out of the
        GLB itself** (:meth:`read_glb_lightmap_manifest`), encodes each referenced HDR
        EXR for the web (:meth:`ImgUtils.encode_hdr_for_web`), embeds it, and binds it
        as the material's ``occlusionTexture`` on ``TEXCOORD_1`` with the
        ``lightmap_web`` root-extras manifest the viewer rebinds from. glTF has no
        lightmap slot; the occlusion carrier is the convention the viewer (and
        blendertk's native exporter) already share -- a naive viewer degrades to grey
        AO rather than to nothing.

        A GLB with no manifest is a clean no-op, which is what makes it safe to run
        unconditionally after every conversion. Name matching is exact first, then
        namespace-tolerant: a name that misses is retried with the ``NS:`` prefix
        stripped from both sides and binds only when that leaf is unambiguous among
        the GLB's nodes (manifests and exports can disagree about namespaces
        without either being wrong -- an older publisher stripped them, some
        exporters flatten them). A name several objects share -- FBX carries leaf
        names only -- is settled by each entry's ``hierarchy`` against where each
        node sits (:meth:`_resolve_lightmap_entries`). Every remaining miss is
        loud: a name nothing tells apart, a name matching no node at all, a
        primitive without ``TEXCOORD_1`` (the FBX was exported without the second
        UV set) and an EXR that cannot be found are each warned and skipped, never
        guessed at. A material worn by objects baked into DIFFERENT maps (a
        secondary material two objects share, or a Per-Object bake of instances)
        binds a copy per object after the first, like a per-instance rect below
        -- it used to be refused, and the object wore the first claimant's
        lighting. The copies are one per (material, map, rect), shared by every
        object baked into that patch of that map.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.
            search_dirs: Directories to resolve the manifest's EXR basenames
                against, in priority order -- the host's own answer to where
                its maps live (a DCC's ``LightmapRecords.search_dirs``: the
                folders the bake markers name first).  Tried after the folders
                a manifest written before 0.11.0 still names, before the GLB's
                directory.
            carrier: ``"occlusion"`` (default) or ``"emissive"`` -- which material
                slot carries the map (mirror of blendertk's ``CARRIERS``).
            percentile: Encode divisor percentile
                (default :attr:`ImgUtils.HDR_WEB_PERCENTILE`).
            replace_authored: Whether a map already sitting in the carrier slot
                gives way to the lightmap. ``True`` (default) displaces it with
                a warning -- a bake IS the deliverable, and its occlusion term
                supersedes a separate AO map. ``False`` keeps the authored map
                and skips the lightmap for that material, warned. Either way, a
                slot holding the material's own ``metallicRoughnessTexture`` is
                NOT authored -- that is the packed-ORM occlusion binding
                (:meth:`set_glb_metallic_roughness`, or FBX2glTF's converted
                packing), whose R channel the bake already contains -- so it is
                displaced silently regardless.

        **Per-instance atlas rects travel as glTF-standard ``KHR_texture_transform``.**
        A manifest record with a non-identity ``scaleOffset`` (an instance's patch of a
        shared atlas over the mesh's shared [0,1] unwrap) cannot bind on the shared
        material -- every sibling would sample the same patch -- so its node gets its
        own MATERIAL clone carrying the rect as a texture transform, and, when the node
        shares its glTF mesh with siblings (FBX2glTF preserves instancing as one mesh
        referenced by many nodes -- probe-measured), its own MESH entry too. Both
        clones are pure JSON referencing the same accessors/bufferViews: zero geometry
        or texture duplication, and any compliant viewer (three.js, model-viewer,
        Babylon, the production WebXR app) renders the rect with no custom code.

        Returns:
            One record per (object, material) binding: ``{"material", "object",
            "map", "intensity", "scaleOffset", "entry"}`` -- several objects
            sharing one atlas material each get a record, and ``entry`` is the
            manifest entry's index, which tells apart objects that share a name.
            Empty when there was no manifest or nothing matched.
        """
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
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            manifest = cls._lightmap_manifest(gltf)
            if not manifest:
                return []
            try:
                version = int(manifest.get("version", -1))
            except (TypeError, ValueError):
                version = -1
            if not 0 < version <= cls.LIGHTMAP_METADATA_VERSION:
                # Refuse rather than guess: a newer schema is free to change what the
                # fields MEAN, and binding a map through a misread manifest looks like
                # a bad bake rather than a version problem.
                logger.warning(
                    "lightmap_metadata v%r is newer than this reader (v%s); "
                    "lightmaps not wired.",
                    manifest.get("version"),
                    cls.LIGHTMAP_METADATA_VERSION,
                )
                return []
            entries = manifest.get("objects") or []
            if not entries:
                return []

            # The manifest's OWN folders first, then the caller's search paths.
            # ``dirs`` (plural) is the publisher's full set: a scene whose maps
            # do not all live in one folder -- the moment ANY object keeps a
            # marker from an earlier bake -- has no single ``dir`` to state, and
            # publishing nothing there dropped the hint for every object that
            # did agree. What followed was silent and severe: the basename
            # search reached the workspace's texture folder and bound a
            # 17-day-old atlas of the same name, so 46 objects sampled a stale
            # map through rects computed for the current bake.
            # ``isinstance`` rather than a bare truthiness test: this reads a
            # manifest out of whatever GLB the caller points at, and a ``dirs``
            # that arrived as a STRING would splat into one bogus directory per
            # character -- a silent, unbounded stat storm rather than an error.
            plural = manifest.get("dirs")
            authored = [
                d
                for d in [
                    manifest.get("dir"),
                    *(plural if isinstance(plural, (list, tuple)) else []),
                ]
                if isinstance(d, str) and d
            ]
            dirs = [d for d in [*authored, *search_dirs] if d]
            dirs.append(os.path.dirname(os.path.abspath(edit.path)))
            authored_norm = {os.path.normcase(os.path.abspath(d)) for d in authored}

            nodes_by_name, leaf_index, mesh_users, lineage = cls._lightmap_node_index(
                gltf
            )
            resolutions = cls._resolve_lightmap_entries(
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
            #: map basename -> the path it resolved to OUTSIDE the manifest's
            #: own folders (found by basename alone; see the bind site).
            fallback_binds: Dict[Optional[str], str] = {}
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
                        encodes[src] = ImgUtils.encode_hdr_for_web(src, percentile)
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
                # A map found OUTSIDE every folder the manifest names was found
                # by basename alone, and a basename is not an identity: the
                # workspace's texture folder routinely holds an atlas from an
                # earlier bake under the same name (and, on a case-insensitive
                # filesystem, under a differently-cased one). Binding it pairs
                # THIS bake's rects with THAT bake's pixels -- every object
                # sampling someone else's lighting, with nothing in the log to
                # say so. Legitimate whenever the maps have simply moved, so it
                # is a warning and not a refusal; named once per map.
                if (
                    authored_norm
                    and os.path.normcase(os.path.dirname(src)) not in authored_norm
                ):
                    fallback_binds.setdefault(basename, src)

                png_name = os.path.splitext(basename)[0] + ".png"

                def _scalar(src=src, png_name=png_name):
                    """Embed *src* on FIRST use (never for an entry whose
                    primitives all fail the guards -- an orphan texture otherwise)."""
                    if src not in scalars:
                        encoded = _encode(src)
                        scalars[src] = None if encoded is None else encoded[1]
                        if encoded is not None:
                            cls._embed_image_bytes(
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
                                    f"{base_name}{cls.LIGHTMAP_CLONE_SUFFIX}"
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

            for basename, resolved in fallback_binds.items():
                logger.warning(
                    "Lightmap %r was not in the folder(s) the manifest names "
                    "(%s); bound %s, found by name alone. If that is an atlas "
                    "from an earlier bake, its rects do not match this one and "
                    "every object on it samples the wrong patch.",
                    basename,
                    ", ".join(authored) or "<none published>",
                    resolved,
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
                # The exact shape the viewer parses (root extras is its 2nd probe).
                web_manifest = {
                    "version": 1,
                    "carrier": carrier,
                    "uv": 1,
                    "encoding": "srgb",
                    "materials": web_materials,
                }
                edit.gltf.setdefault("extras", {})[cls.LIGHTMAP_WEB_KEY] = web_manifest
                # The maps are in the file now, so the authoring-path hints that
                # found them have no reader left -- drop them here, where that is
                # provably true, rather than at export (where the applier still
                # needs them) or never (where they ship). Deliberately gated on a
                # successful embed: a run that bound nothing leaves them intact so
                # a retry -- after fixing a name mismatch, say -- can still locate
                # the EXRs.
                # Same walk corrects the superseded copies: every marker and
                # manifest entry this run bound still names the .exr at the
                # pre-normalisation intensity. Fed the PUBLISHED values, read
                # back through each node's materials (the dicts that went into
                # lightmap_web), so the copies come out identical rather than
                # merely close: a record's "map" is the SOURCE .exr basename --
                # which is what a caller wants to know -- and its "intensity" is
                # unrounded where the published one is round(., 6).
                cls._reconcile_node_markers(
                    edit.gltf, cls._lightmap_final_values(edit.gltf, web_manifest)
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
