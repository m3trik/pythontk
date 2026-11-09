# !/usr/bin/python
# coding=utf-8
"""The media every record whose files ride a GLB shares: its entries
resolved to the file's nodes, and its files carried as bufferViews.

Record-agnostic, so the ``lightmaps`` record, the ``shadows`` record and any
later one go through the one rule and the one writer:

- **Resolution** -- a record's entries (``objects``, ``planes``) resolved to
  the GLB's mesh nodes as ONE assignment (:meth:`_Media.resolve_entries`),
  the rule the WebXR viewer's ``kernel/resolve.js`` mirrors; a record whose
  entries name nodes that are no mesh (the ``articulation`` record's joints)
  resolves them over every node by the same rule, and one whose entry is
  found under another's node (a grabbed part under its joint) over that
  node's subtree alone.
- **Transport** -- each file a record names carried as the bufferView named
  by that file (``SceneRecords.GLB_MEDIA``): found (:meth:`_Media.locate`,
  :meth:`_Media.extract_fbx_files`), written (:meth:`_Media.put`), read back
  (:meth:`_Media.held`) and gated (:meth:`_Media.findings`).

One part of :class:`MeshConvert`, composed in ``_mesh_convert.py``; the
record passes reach it through the facade (``self._mc._media``).
"""

import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Eager: the signatures name GlbEdit / RecordSpec; SceneRecords is the light
# declaration module every pass reads.
from pythontk.core_utils.engines.scene_export.scene_records import (
    NodeNames,
    RecordSpec,
    SceneRecords,
)
from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit

logger = logging.getLogger(__name__)


class _Media:
    """A record's entries resolved to the GLB's nodes, and its files carried.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    # ------------------------------------------------------------ resolution

    @staticmethod
    def node_parents(gltf: dict) -> Dict[int, int]:
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
    def node_lineage(parents: Dict[int, int], index: int) -> List[int]:
        """*index* and its ancestors, nearest first; a cycle ends the walk."""
        chain, seen = [index], {index}
        while chain[-1] in parents and parents[chain[-1]] not in seen:
            chain.append(parents[chain[-1]])
            seen.add(chain[-1])
        return chain

    @staticmethod
    def node_subtree(gltf: dict, index: int) -> set:
        """*index* and every node below it, through ``children``; a cycle
        ends the walk."""
        nodes = gltf.get("nodes") or []
        found, todo = set(), [index]
        while todo:
            at = todo.pop()
            if not isinstance(at, int) or at in found or not 0 <= at < len(nodes):
                continue
            found.add(at)
            todo.extend((nodes[at] or {}).get("children") or [])
        return found

    def node_index(
        self, gltf: dict, *, meshes: bool = True, under: Optional[int] = None
    ):
        """Index a glTF's MESH nodes for a record's lookup -- every node,
        with *meshes* false; with *under*, only that node and the nodes below
        it (:meth:`node_subtree`) -- as ``kernel/resolve.js``'s
        ``indexNodes`` does.

        Returns ``(nodes_by_name, leaf_index, mesh_users, lineage)``: nodes
        grouped by their exact name, the namespace-stripped leaf of each of
        those names mapped back to the full names carrying it, how many nodes
        reference each mesh, and each indexed node's LINEAGE -- the names
        from its scene root down to itself, keyed by ``id(node)`` -- which is
        what tells same-named nodes apart (:meth:`resolve_entries`).
        """
        nodes = gltf.get("nodes", []) or []
        parents = self.node_parents(gltf)
        inside = None if under is None else self.node_subtree(gltf, under)
        nodes_by_name: Dict[str, List[dict]] = {}
        mesh_users: Dict[int, int] = {}  # mesh index -> node reference count
        lineage: Dict[int, Tuple[str, ...]] = {}
        for index, node in enumerate(nodes):
            if not isinstance(node, dict) or (meshes and "mesh" not in node):
                continue
            if inside is not None and index not in inside:
                continue
            nodes_by_name.setdefault(node.get("name", ""), []).append(node)
            if "mesh" in node:
                mesh_users[node["mesh"]] = mesh_users.get(node["mesh"], 0) + 1
            lineage[id(node)] = tuple(
                str(nodes[i].get("name") or "")
                for i in reversed(self.node_lineage(parents, index))
            )
        leaf_index: Dict[str, List[str]] = {}
        for full in nodes_by_name:
            leaf_index.setdefault(full.rsplit(":", 1)[-1], []).append(full)
        return nodes_by_name, leaf_index, mesh_users, lineage

    @staticmethod
    def hierarchy_score(hierarchy: Sequence[Any], lineage: Sequence[str]) -> int:
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

    def resolve_entries(
        self,
        entries: Sequence[Any],
        nodes_by_name: Dict[str, List[dict]],
        leaf_index: Dict[str, List[str]],
        lineage: Dict[int, Tuple[str, ...]],
    ) -> List[Tuple[Optional[List[dict]], str, List[str]]]:
        """Resolve every entry of a record list to its GLB mesh nodes, as ONE
        assignment (the ``lightmaps`` record's ``objects``, the ``shadows``
        record's ``planes``).

        One ``(nodes, status, leaves)`` per entry, aligned with *entries* (see
        :meth:`resolve_node` for the statuses; a malformed or
        nameless entry is ``"absent"``). Names alone settle the common case --
        one entry, one node. A scene that reuses a leaf name does not have it:
        FBX carries leaf names only, so two objects named ``BODY`` publish two
        ``BODY`` entries and arrive as two ``BODY`` nodes, and a stale marker
        on a hidden object can put a second entry on a name whose node belongs
        to another. Every set of entries whose candidate nodes OVERLAP is
        settled together -- two such sets settled apart could each bind the
        same node, and the object would wear whichever came last. Each
        (entry, node) pair ranks by the entry's ``hierarchy`` (its scene path,
        root first) against the node's lineage (:meth:`hierarchy_score`),
        then by whether the NAME matched exactly, so an exact name still beats
        a namespace-stripped one wherever the hierarchy cannot tell them apart
        (the file carries ``PROPS_DA:prop352``, the manifest also lists
        ``PROPS_RF:prop352``). An entry binds to a node only when each is the
        other's UNIQUE best match. An entry whose best candidates all went to
        better-matching entries is ``"absent"`` -- its object is not in this
        file -- and anything still tied, or with nothing to compare, is
        ``"ambiguous"``: never guessed, because a guess puts one object's
        lighting on another. A per-entry match is never widened past the rule
        :meth:`resolve_node` applies.

        The WebXR viewer resolves the record by the same rule
        (``kernel/resolve.js``); ``test_preview_viewer_live`` holds the two to
        the same answers.
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
            nodes, status, leaves = self.resolve_node(
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
                        self.hierarchy_score(hierarchy, lineage.get(id(node), ()))
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

    def resolve_node(
        self,
        name: Optional[str],
        nodes_by_name: Dict[str, List[dict]],
        leaf_index: Dict[str, List[str]],
    ) -> Tuple[Optional[List[dict]], str, List[str]]:
        """Resolve one entry's name to the GLB's mesh nodes, by name alone.

        The per-entry stage of :meth:`resolve_entries`, which every
        reader goes through -- two readers of the match rule is how the binder
        and the report came to disagree about what "missing" means -- and which
        settles what a name alone cannot (several objects sharing it).
        Exact first, then namespace-tolerant: manifests and exports can
        disagree about namespaces without either being wrong (an older
        publisher stripped them, some exporters flatten them), but a leaf
        matching several nodes is never guessed at -- that would put one
        object's lighting on another. The ladder is the one every reader
        binds a name by (``NodeNames.resolve``, the transport's name fallback
        too).

        Returns ``(nodes, status, leaves)`` where *status* is one of
        ``"exact"``, ``"leaf"``, ``"ambiguous"`` (in the file, unbindable) or
        ``"absent"`` (no such node -- on a selection-scoped export, simply not
        part of it). *leaves* carries the candidates behind an ambiguous match.
        """
        return NodeNames.resolve(name, nodes_by_name, leaf_index)

    def resolved(
        self,
        gltf: dict,
        entries: Sequence[Any],
        *,
        meshes: bool = True,
        under: Optional[int] = None,
    ):
        """``[(entry, nodes, status, leaves)]``: each of *entries* (a
        record's ``objects``, ``planes``) resolved to *gltf*'s mesh nodes --
        every node with *meshes* false, *under*'s subtree with *under*
        (:meth:`node_index`) -- as every reader resolves them
        (:meth:`resolve_entries`)."""
        by_name, leaves, _users, lineage = self.node_index(
            gltf, meshes=meshes, under=under
        )
        resolved = self.resolve_entries(entries, by_name, leaves, lineage)
        return [
            (entry, nodes, status, found)
            for entry, (nodes, status, found) in zip(entries, resolved)
        ]

    # ------------------------------------------------------------- transport

    @staticmethod
    def held(gltf: dict, name: str, kind: str = "File") -> Optional[int]:
        """The one bufferView of *gltf* holding the file *name*; ``None`` when
        none does, and -- warned, *kind* naming the file's role -- when several
        do: a tie no reader binds."""
        held = SceneRecords.GLB_MEDIA.views(gltf, name)
        if len(held) > 1:
            logger.warning(
                "%s %r is held by %d bufferViews (%s); no reader binds it.",
                kind,
                name,
                len(held),
                ", ".join(map(str, held)),
            )
            return None
        return held[0] if held else None

    @staticmethod
    def put(
        edit: GlbEdit,
        name: str,
        data: bytes,
        mime_type: str,
        scale: Optional[float] = None,
        kind: str = "File",
    ) -> Optional[int]:
        """The bufferView holding the file *name*: one already there (a
        second pass adds nothing), else *data* appended as the view named by
        the file, its ``extras`` stating its bytes (:class:`GlbMedia`).
        ``None``, warned (*kind* names the file's role), when several views
        hold it -- a tie no reader binds, which another copy would only
        deepen -- or when the GLB's buffer is external."""
        media = SceneRecords.GLB_MEDIA
        if media.views(edit.gltf, name):
            return _Media.held(edit.gltf, name, kind)
        views = GlbEdit.append_bin_views(edit, [data])
        if not views:
            logger.warning("%s %r not embedded: the GLB's buffer is external.", kind, name)
            return None
        view = edit.gltf["bufferViews"][views[0]]
        view[media.name] = media.file_name(name)
        view["extras"] = media.extras(mime_type, scale)
        edit.dirty = True
        return views[0]

    def extract_fbx_files(
        self, gltf: dict, fbx: str, dst_dir: str, spec: Optional[RecordSpec] = None
    ) -> List[str]:
        """Write each file *gltf*'s *spec* record names (the ``lightmaps``
        record's by default; any record whose files ride a GLB, by
        ``FbxMedia.DEPENDENCY_FIELDS``) that the FBX *fbx* embeds into
        *dst_dir*; ``[dst_dir]`` when it wrote any, else ``[]`` -- the search
        path the FBX's own bytes are found on. A GLB built from a finished FBX
        names the drawn probe and its copies, which the FBX alone holds, and
        the shadow maps the export embedded. The record is read as the file
        ships it (its root ``extras``), else off its carrier in the current
        schema. An FBX this reader cannot parse (a stand-in, a text FBX)
        holds none."""
        from pythontk.file_utils.mesh_convert.fbx_media import FbxMedia

        spec = spec or SceneRecords.LIGHTMAPS
        record = SceneRecords.glb_block(gltf, spec)
        if record is None:
            payload = self._mc.data_export_channel(gltf, spec.key)
            current = isinstance(payload, dict) and spec.is_current(payload)
            record = payload if current else None
        if not record:
            return []
        names = FbxMedia.named_files({spec.key: [record]})
        if not names:
            return []
        try:
            written = FbxMedia.extract(fbx, dst_dir, names)
        except (ValueError, OSError) as error:
            logger.debug("%s: %s embeds nothing readable: %s", spec.key, fbx, error)
            return []
        return [dst_dir] if written else []

    @staticmethod
    def locate(dirs: Sequence[str], name: Any) -> Optional[str]:
        """The record's file *name* in the first of *dirs* holding it, else ``None``."""
        basename = SceneRecords.GLB_MEDIA.file_name(name)
        if not basename:
            return None
        return next(
            (
                os.path.join(d, basename)
                for d in dirs
                if os.path.isfile(os.path.join(d, basename))
            ),
            None,
        )

    def findings(
        self,
        gltf: dict,
        spec: RecordSpec,
        checks: Dict[str, Callable[[Dict[str, Any]], Optional[str]]],
        reading: Iterable[str],
        *,
        readers: str,
        without: str,
    ) -> Tuple[List[str], List[str]]:
        """``(problems, notes)``: the files *gltf*'s *spec* record names, held
        to the media transport -- the gate every record whose files ride a
        GLB shares (each record's ``media_findings`` states its own rules).

        A file several bufferViews hold is held by none (a problem); a held
        file's view is put to its check in *checks* (file name -> a callable
        over the view's ``extras`` returning the problem it breaks, or
        ``None``). A file in *reading* -- one an entry this file carries reads
        -- that rides nothing is a note: the build warned of it, and its
        *readers* (``"objects"``, ``"planes"``) go *without* it (``"render
        unlit"``).
        """
        media = SceneRecords.GLB_MEDIA
        where = f"extras.{spec.key}"
        views = gltf.get("bufferViews") or []
        problems: List[str] = []
        for name in sorted(checks):
            held = media.views(gltf, name)
            if len(held) > 1:
                problems.append(
                    f"{where}: {len(held)} bufferViews hold {name} (bufferViews "
                    f"{', '.join(map(str, held))}); a reader binds none of them"
                )
                continue
            if not held:
                continue
            extras = views[held[0]].get("extras")
            problem = checks[name](extras if isinstance(extras, dict) else {})
            if problem:
                problems.append(f"{where}: the view of {name} {problem}")
        unshipped = sorted(
            {n for n in reading if n and media.view(gltf, n) is None}
        )
        notes: List[str] = []
        if unshipped:
            notes.append(
                f"{where}: {len(unshipped)} map(s) {readers} in this file read "
                f"ride no bufferView, so they {without}: {', '.join(unshipped)}"
            )
        return problems, notes
