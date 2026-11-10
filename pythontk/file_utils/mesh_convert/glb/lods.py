# !/usr/bin/python
# coding=utf-8
"""``GlbLods``: a GLB states each LOD group in glTF's own terms, ``MSFT_lod``.

The FBX names a group's levels ``<base>_LOD0..n`` (:class:`pythontk.LodChain`,
the names game engines read) and FBX2glTF writes them as sibling nodes, which every
glTF reader would draw at once. glTF has no ratified LOD extension; the one in
Khronos' registry is the vendor ``MSFT_lod`` (Babylon, engines' and DCCs'
glTF importers, our viewer read it). The pass gives each group's LOD0 node
``extensions.MSFT_lod.ids`` -- the lower levels, taken out of the node tree --
and ``extras.MSFT_screencoverage``, one switch height per level with the cull
last (the spec's companion hint), and lists the extension in
``extensionsUsed`` only: a reader without it draws LOD0 alone, the spec's
fallback, so the file stays plug and play. One self-contained file.

The heights are the FBX's own where its LodGroup attribute states them in
percent (a DCC's FBX writer, with Animation on or off: measured 2026-11-10), else
:meth:`LodChain.default_heights`. Measured 2026-11-10: the Khronos validator
passes such a file with 0 errors and 0 warnings (infos only: an extension it
does not check, and the out-of-tree level nodes "may be unused").
"""

import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from pythontk.file_utils.lod_chain import LodChain

logger = logging.getLogger(__name__)


class GlbLods:
    """``MSFT_lod`` for the LOD groups a GLB's node names state."""

    EXTENSION = "MSFT_lod"
    COVERAGE = "MSFT_screencoverage"

    @classmethod
    def apply(
        cls,
        gltf: Dict[str, Any],
        heights: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> Dict[str, Any]:
        """State every well-formed LOD group of *gltf* in ``MSFT_lod`` (in place).

        Parameters:
            gltf: The GLB's JSON.
            heights: ``{group node name: heights}``, one per level (the cull
                last) -- ``FbxMedia.lod_heights`` of the source; a group it does
                not name, or names with another level count, takes the
                default ladder.

        Returns:
            ``{"groups", "levels", "names"}``: how many groups were stated, how
            many levels they hold and the groups' names. A group with an issue
            (no LOD0, triangles that do not fall ...) is left as written, as
            the Scene Exporter warned; one already stated is left too.
        """
        nodes = gltf.get("nodes") or []
        parent = cls._tree_parents(nodes)
        paths: Dict[int, str] = {}

        def path_of(index: int) -> str:
            # Walked up, not recursed: a deep tree or a malformed file's cycle
            # must not overflow the stack.
            chain, walk, seen = [], index, set()
            while walk is not None and walk not in paths and walk not in seen:
                seen.add(walk)
                chain.append(walk)
                walk = parent.get(walk)
            above = paths.get(walk, "") if walk is not None else ""
            for node_index in reversed(chain):
                above = paths[node_index] = above + "|" + str(nodes[node_index].get("name") or f"node{node_index}")
            return paths[index]

        rows = []
        index_of: Dict[str, int] = {}
        for index, node in enumerate(nodes):
            path = path_of(index)
            index_of.setdefault(path, index)
            above = parent.get(index)
            rows.append(
                {
                    "path": path,
                    "parent": path_of(above) if above is not None else None,
                    "triangles": cls._triangles(gltf, node),
                }
            )
        report: Dict[str, Any] = {"groups": 0, "levels": 0, "names": []}
        for group in LodChain.groups(rows):
            if group["issues"] or group["parent"] is None:
                continue
            levels = [index_of[p] for p in group["levels"]]
            if any(cls.EXTENSION in (nodes[i].get("extensions") or {}) for i in levels):
                continue
            name = group["parent"].rsplit("|", 1)[-1]
            cls._state(gltf, index_of[group["parent"]], levels, (heights or {}).get(name))
            report["groups"] += 1
            report["levels"] += len(levels)
            report["names"].append(name)
        return report

    @classmethod
    def _state(
        cls,
        gltf: Dict[str, Any],
        holder: int,
        levels: Sequence[int],
        heights: Optional[Sequence[float]] = None,
    ) -> None:
        """One group in ``MSFT_lod``: node *holder*'s lower *levels* leave its
        children, LOD0 names them in ``ids`` and carries the switch *heights*
        (the default ladder when they are not valid for the levels), and the
        file lists the extension."""
        nodes = gltf["nodes"]
        lod0, lower = levels[0], list(levels[1:])
        nodes[holder]["children"] = [c for c in nodes[holder].get("children") or [] if c not in lower]
        stated = list(heights or ())
        try:
            LodChain.check_heights(stated)
            if len(stated) != len(levels):
                raise ValueError("another level count")
        except ValueError as error:
            if heights is not None:
                logger.warning(
                    "%s: its heights %s do not fit its %d levels (%s); it switches at the default ladder.",
                    nodes[holder].get("name") or holder, list(heights), len(levels), error,
                )
            stated = LodChain.default_heights(len(levels))
        nodes[lod0].setdefault("extensions", {})[cls.EXTENSION] = {"ids": lower}
        nodes[lod0].setdefault("extras", {})[cls.COVERAGE] = stated
        used = gltf.setdefault("extensionsUsed", [])
        if cls.EXTENSION not in used:
            used.append(cls.EXTENSION)

    @classmethod
    def merge(
        cls,
        sources: Union[str, Sequence[str]],
        dst: Optional[str] = None,
        heights: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> Dict[str, Any]:
        """One GLB of LOD groups from GLBs of one node tree at falling detail,
        finest first: a CAD import's profiles (each a tessellation of the
        exact surfaces, :class:`pythontk.CadImport`), or any writer's levels
        of one scene.

        Each mesh node of the first becomes a group: it keeps its name, its
        transform and its place, and holds its levels, ``<name>_LOD0`` (its
        own mesh) and one ``<name>_LOD<n>`` per later GLB whose matching node
        draws fewer triangles than the level before (a mesher's floor leaves a
        level out for that node), stated in ``MSFT_lod`` (:meth:`apply`'s
        form). Instances keep sharing each level's mesh. A skinned node, or one
        holding other nodes, keeps its one mesh (named in ``skipped``). Nodes
        match by their place in the tree, names checked: trees that differ
        are refused, and a refusal writes nothing (the work is done on a copy
        that replaces *dst* only when it is whole).

        Parameters:
            sources: Two or more GLB paths, finest first; or a CadImport
                record (``.cad.json``), whose assemblies are taken finest
                chord first.
            dst: The GLB written (may be the first GLB, edited in place);
                default ``<stem>_lods.glb`` beside the record (``<stem>`` its
                source's) or beside the first GLB.
            heights: ``{node name: heights}``, one per level kept, the cull
                last; any other group takes :meth:`LodChain.default_heights`.

        Returns:
            ``{"path", "groups", "levels", "names", "skipped"}``: the GLB
            written, the groups made, the levels they hold, their names, and
            the mesh nodes left as they were (with why).
        """
        import json
        import shutil

        if isinstance(sources, (str, os.PathLike)):
            with open(sources, encoding="utf-8") as fh:
                record = json.load(fh)
            where = os.path.dirname(os.path.abspath(sources))
            profiles = sorted(record.get("profiles") or [], key=lambda p: float(p["chord"]))
            glbs = [os.path.join(where, record["assemblies"][p["name"]]) for p in profiles]
            stem = os.path.splitext(record["source"]["file"])[0]
            dst = dst or os.path.join(where, f"{stem}_lods.glb")
        else:
            glbs = [os.fspath(g) for g in sources]
            dst = dst or (os.path.splitext(glbs[0])[0] + "_lods.glb" if glbs else None)
        if len(glbs) < 2:
            raise ValueError("A LOD merge needs two GLBs or more, finest first.")
        if len(glbs) > LodChain.MAX_LEVELS:
            raise ValueError(f"A LOD group holds at most {LodChain.MAX_LEVELS} levels, not {len(glbs)}.")
        os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
        work = f"{dst}.lods-tmp"
        shutil.copyfile(glbs[0], work)
        try:
            report = cls._merge_into(work, glbs, heights)
            os.replace(work, dst)
        finally:
            if os.path.exists(work):
                os.remove(work)
        report["path"] = dst
        logger.info(
            "LOD merge: %d group(s), %d level(s) -> %s%s",
            report["groups"],
            report["levels"],
            os.path.basename(dst),
            f" ({len(report['skipped'])} mesh node(s) kept at one level)" if report["skipped"] else "",
        )
        return report

    @classmethod
    def _merge_into(
        cls, work: str, glbs: Sequence[str], heights: Optional[Mapping[str, Sequence[float]]]
    ) -> Dict[str, Any]:
        """:meth:`merge`'s work on *work*, a copy of the finest GLB."""
        from pythontk.file_utils.mesh_convert.glb.edit import GlbEdit

        report: Dict[str, Any] = {"groups": 0, "levels": 0, "names": [], "skipped": []}
        with GlbEdit.open(work) as edit:
            gltf = edit.gltf
            places = cls._places(gltf)
            mesh_nodes = []
            for i in places:
                node = gltf["nodes"][i]
                if node.get("mesh") is None:
                    continue
                name = str(node.get("name") or f"node{i}")
                if node.get("skin") is not None:
                    report["skipped"].append(f"{name}: skinned; its levels would each need its skin.")
                elif node.get("children"):
                    report["skipped"].append(f"{name}: it holds other nodes; a LOD group's node holds its levels alone.")
                else:
                    mesh_nodes.append(i)
            chains: Dict[int, List[Any]] = {
                i: [(gltf["nodes"][i]["mesh"], cls._triangles(gltf, gltf["nodes"][i]))] for i in mesh_nodes
            }
            for path in glbs[1:]:
                other = GlbEdit.read(path)
                theirs = {place: i for i, place in cls._places(other.gltf).items()}
                matched: Dict[int, Tuple[int, int]] = {}
                for i in mesh_nodes:
                    j = theirs.get(places[i])
                    node = other.gltf["nodes"][j] if j is not None else None
                    if node is None or node.get("name") != gltf["nodes"][i].get("name"):
                        raise ValueError(
                            f"{os.path.basename(path)} is not the same node tree as "
                            f"{os.path.basename(glbs[0])} (node {gltf['nodes'][i].get('name')!r})."
                        )
                    count = cls._triangles(other.gltf, node) if node.get("mesh") is not None else None
                    # Only a level with fewer triangles joins its chain: decided
                    # before the graft, so no mesh is copied that nothing draws.
                    if count is not None and count < chains[i][-1][1]:
                        matched[i] = (j, count)
                grafted = GlbEdit.graft_meshes(
                    edit, other, sorted({other.gltf["nodes"][j]["mesh"] for j, _ in matched.values()})
                )
                for i, (j, count) in matched.items():
                    chains[i].append((grafted[other.gltf["nodes"][j]["mesh"]], count))
            nodes = gltf["nodes"]
            for i in mesh_nodes:
                name = str(nodes[i].get("name") or f"node{i}")
                if len(chains[i]) < 2:
                    report["skipped"].append(f"{name}: no later GLB draws it with fewer triangles.")
                    continue
                levels = []
                for k, (mesh, _count) in enumerate(chains[i]):
                    nodes.append({"name": LodChain.level_name(name, k), "mesh": mesh})
                    levels.append(len(nodes) - 1)
                del nodes[i]["mesh"]
                nodes[i]["children"] = list(levels)
                cls._state(gltf, i, levels, (heights or {}).get(name))
                report["groups"] += 1
                report["levels"] += len(levels)
                report["names"].append(name)
            edit.dirty = True
        return report

    @staticmethod
    def _places(gltf: Mapping[str, Any]) -> Dict[int, tuple]:
        """``{node index: its place in the default scene's tree}``: the child
        positions from a root down, the key two writings of one tree share."""
        nodes = gltf.get("nodes") or []
        scenes = gltf.get("scenes") or []
        roots = (scenes[gltf.get("scene", 0)] if scenes else {}).get("nodes") or []
        places: Dict[int, tuple] = {}
        stack = [(r, (k,)) for k, r in enumerate(roots)]
        while stack:
            index, place = stack.pop()
            if not isinstance(index, int) or not 0 <= index < len(nodes) or index in places:
                continue
            places[index] = place
            for k, child in enumerate(nodes[index].get("children") or []):
                stack.append((child, place + (k,)))
        return places

    @classmethod
    def expand(cls, gltf: Dict[str, Any]) -> Dict[int, List[float]]:
        """The way back from :meth:`apply`, in place: every ``MSFT_lod`` group
        of *gltf* becomes the convention's node tree, ``<base>`` over
        ``<base>_LOD0..n``, for a writer that holds levels as nodes (FBX).

        A group :meth:`apply` stated gets its lower levels back under its
        node, after LOD0. Any other (another writer's names, a parent holding
        other nodes, LOD0 at the scene root) gets a ``<base>`` node at identity
        in LOD0's place, its levels under it, each keeping its own transform
        (each replaces LOD0 where it stands) and renamed ``<base>_LOD<n>``,
        ``<base>`` LOD0's name less any level suffix. ``MSFT_lod`` and its
        coverage leave the nodes, and the extension the file when none is left.

        Returns:
            ``{group node index: heights}``, one height per level with the cull
            last: the coverage the file states where it is valid for the
            levels, else :meth:`LodChain.default_heights`. An id out of range
            or already in the node tree, or more levels than
            :attr:`LodChain.MAX_LEVELS`, leaves its group as it is.
        """
        nodes = gltf.get("nodes") or []
        parent = cls._tree_parents(nodes)
        scenes = gltf.get("scenes") or []
        groups: Dict[int, List[float]] = {}
        for lod0 in range(len(nodes)):
            node = nodes[lod0]
            ids = ((node.get("extensions") or {}).get(cls.EXTENSION) or {}).get("ids")
            if not isinstance(ids, list) or not ids:
                continue
            lower = [i for i in ids if isinstance(i, int) and 0 <= i < len(nodes) and i not in parent]
            if len(lower) != len(ids) or lod0 in lower or len(set(lower)) != len(lower):
                continue
            if len(ids) + 1 > LodChain.MAX_LEVELS:  # MSFT_lod sets no limit; the convention does
                continue
            levels = [lod0] + lower
            names = [str(nodes[i].get("name") or "") for i in levels]
            parsed = LodChain.parse(names[0])
            base = parsed[0] if parsed and parsed[1] == 0 else names[0] or f"node{lod0}"
            holder = parent.get(lod0)
            ours = (
                holder is not None
                and nodes[holder].get("name") == base
                and nodes[holder].get("children") == [lod0]
                and all(LodChain.parse(n) == (base, k) for k, n in enumerate(names))
            )
            if ours:
                group = holder
                nodes[group]["children"] = list(levels)
            else:
                group = len(nodes)
                nodes.append({"name": base, "children": list(levels)})
                containers = (
                    [nodes[holder]["children"]]
                    if holder is not None
                    else [s["nodes"] for s in scenes if isinstance(s.get("nodes"), list)]
                )
                for container in containers:
                    container[:] = [group if c == lod0 else c for c in container]
                if holder is not None:
                    parent[group] = holder
                for k, i in enumerate(levels):
                    nodes[i]["name"] = LodChain.level_name(base, k)
            for i in levels:
                parent[i] = group
            stated = (node.get("extras") or {}).get(cls.COVERAGE)
            try:
                LodChain.check_heights(stated or ())
                if len(stated) != len(levels):
                    raise ValueError("another level count")
                heights = [float(h) for h in stated]
            except (TypeError, ValueError):
                heights = LodChain.default_heights(len(levels))
            del node["extensions"][cls.EXTENSION]
            if not node["extensions"]:
                del node["extensions"]
            if isinstance(node.get("extras"), dict):
                node["extras"].pop(cls.COVERAGE, None)
                if not node["extras"]:
                    del node["extras"]
            groups[group] = heights
        if groups and not any(cls.EXTENSION in (n.get("extensions") or {}) for n in nodes):
            for key in ("extensionsUsed", "extensionsRequired"):
                listed = gltf.get(key)
                if isinstance(listed, list) and cls.EXTENSION in listed:
                    listed.remove(cls.EXTENSION)
                    if not listed:
                        del gltf[key]
        return groups

    @staticmethod
    def _tree_parents(nodes: Sequence[Mapping[str, Any]]) -> Dict[int, int]:
        """``{child: parent}`` over the node tree's own ``children`` lists."""
        parent: Dict[int, int] = {}
        for index, node in enumerate(nodes):
            for child in node.get("children") or []:
                if isinstance(child, int):
                    parent[child] = index
        return parent

    @classmethod
    def parents(cls, nodes: Sequence[Mapping[str, Any]]) -> Dict[int, int]:
        """``{node index: parent index}`` for *nodes*, a group's lower levels
        under the parent of its LOD0 node: MSFT_lod keeps them out of the node
        tree, and each replaces LOD0 where it stands. The one parent map a
        reader of a GLB's hierarchy needs (``GlbReader.parent_of``, the export
        fingerprint's node paths)."""
        parent = cls._tree_parents(nodes)
        for index, node in enumerate(nodes):
            ids = ((node.get("extensions") or {}).get(cls.EXTENSION) or {}).get("ids") or []
            above = parent.get(index)
            if above is None:
                continue
            ancestors, walk = {index}, above  # an id naming one would loop every walk up
            while walk is not None and walk not in ancestors:
                ancestors.add(walk)
                walk = parent.get(walk)
            for level in ids:
                if isinstance(level, int) and level not in parent and level not in ancestors:
                    parent[level] = above
        return parent

    @staticmethod
    def _triangles(gltf: Mapping[str, Any], node: Mapping[str, Any]) -> Optional[int]:
        """The triangles *node*'s mesh draws, or ``None`` when it has none."""
        mesh_index = node.get("mesh")
        meshes = gltf.get("meshes") or []
        if not isinstance(mesh_index, int) or mesh_index >= len(meshes):
            return None
        accessors = gltf.get("accessors") or []
        total = 0
        for primitive in meshes[mesh_index].get("primitives") or []:
            if primitive.get("mode", 4) != 4:
                continue
            source = primitive.get("indices")
            if source is None:
                source = (primitive.get("attributes") or {}).get("POSITION")
            if isinstance(source, int) and source < len(accessors):
                total += int(accessors[source].get("count", 0)) // 3
        return total

    @classmethod
    def renumber(cls, nodes: List[Dict[str, Any]], remap: Mapping[int, int]) -> None:
        """Follow a node renumber (*remap*: old index -> new) in every
        ``MSFT_lod``: a removed level takes its id and its height with it (the
        level kept last still draws down to the group's cull:
        :meth:`LodChain.kept_heights`), and a group left with no lower level
        loses the extension."""
        for node in nodes:
            extension = (node.get("extensions") or {}).get(cls.EXTENSION)
            if not isinstance(extension, dict):
                continue
            ids = list(extension.get("ids") or [])
            coverage = (node.get("extras") or {}).get(cls.COVERAGE)
            kept = [k for k, i in enumerate(ids) if isinstance(i, int) and i in remap]
            extension["ids"] = [remap[ids[k]] for k in kept]
            if isinstance(coverage, list) and len(coverage) == len(ids) + 1:
                node["extras"][cls.COVERAGE] = LodChain.kept_heights(coverage, [0] + [k + 1 for k in kept])
            if not extension["ids"]:
                del node["extensions"][cls.EXTENSION]
                if not node["extensions"]:
                    del node["extensions"]
                extras = node.get("extras") or {}
                extras.pop(cls.COVERAGE, None)
                if "extras" in node and not node["extras"]:
                    del node["extras"]
