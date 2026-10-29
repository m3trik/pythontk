# !/usr/bin/python
# coding=utf-8
"""Articulated rigs: publish the ``articulation`` manifest the viewer poses.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through the facade it holds (``self._mc``).
"""

import copy
import logging
from typing import Any, Dict, List, Optional

# Eager: the channel keys below are SceneRecords' own, read at class
# definition.
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbTarget

logger = logging.getLogger(__name__)


class _Articulation:
    """Articulated rigs: the ``extras.articulation`` block.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    def apply_glb_articulation(self, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            extras = gltf.get("extras")
            if not isinstance(extras, dict):
                extras = gltf["extras"] = {}
            payload = self._mc._current_record(gltf, SceneRecords.ARTICULATION)
            rigs = self._bind_articulated_rigs(gltf, payload) if payload else []
            if rigs:
                edit.dirty = True
                return SceneRecords.put_glb_block(
                    gltf, SceneRecords.ARTICULATION, {"rigs": rigs}
                )
            if extras.pop(SceneRecords.ARTICULATION.key, None) is not None:
                edit.dirty = True
            return None

    def _bind_articulated_rigs(
        self, gltf: dict, payload: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Each rig of *payload* with its joints and grab entries bound to
        node indices; rigs that cannot be bound are left out (see
        :meth:`apply_glb_articulation`)."""
        parents = self._mc._lightmaps._node_parents(gltf)

        def under(node: int, ancestor: int) -> bool:
            # The lineage walk ends at a cycle: a malformed file never hangs.
            return ancestor in self._mc._lightmaps._node_lineage(parents, node)[1:]

        bound: List[Dict[str, Any]] = []
        out_of_scope = 0
        for rig in payload.get("rigs") or []:
            if not isinstance(rig, dict):
                continue
            name = rig.get("name")
            joints = copy.deepcopy(rig.get("joints") or [])
            indices: List[int] = []
            for joint in joints:
                found = self._mc._shadow_rigs._nodes_named(gltf, joint.get("name"))
                if len(found) != 1:
                    break
                indices.append(found[0])
            if len(indices) != len(joints) or not joints:
                missing = (
                    joints[len(indices)]["name"] if len(indices) < len(joints) else None
                )
                found = (
                    self._mc._shadow_rigs._nodes_named(gltf, missing) if missing else []
                )
                if found:
                    logger.warning(
                        "Articulated rig %r: joint %r matches several GLB nodes "
                        "(%s) -- ambiguous, the rig is not wired.",
                        name,
                        missing,
                        ", ".join(str(i) for i in found),
                    )
                else:
                    out_of_scope += 1
                continue
            # A rig is solved in ONE space, its root joints' parent: the page
            # (articulated_rig.js) and unitytk's controller both refuse a rig
            # whose roots hang under different nodes, so the manifest does too.
            spaces = {
                parents.get(index)
                for joint, index in zip(joints, indices)
                if joint.get("parent") is None
            }
            if len(spaces) > 1:
                logger.warning(
                    "Articulated rig %r: its root joints hang under different "
                    "nodes -- the rig is not wired.",
                    name,
                )
                continue
            # Bound by index, the name the node already carries left out.
            names = [joint.pop("name", None) for joint in joints]
            for joint, index in zip(joints, indices):
                joint["node"] = index
            grab = []
            for entry in rig.get("grab") or []:
                try:
                    joint_index = int(entry.get("joint"))
                    joint_node = indices[joint_index]
                except (TypeError, ValueError, IndexError):
                    continue
                candidates = [
                    i
                    for i in self._mc._shadow_rigs._nodes_named(gltf, entry.get("node"))
                    if under(i, joint_node)
                ]
                if len(candidates) != 1:
                    logger.warning(
                        "Articulated rig %r: grab part %r is %s under joint %r; "
                        "it will not be grabbable.",
                        name,
                        entry.get("node"),
                        "ambiguous" if candidates else "not found",
                        names[joint_index],
                    )
                    continue
                grab.append({"node": candidates[0], "joint": joint_index})
            bound.append(dict(rig, joints=joints, grab=grab))
        if out_of_scope:
            logger.info(
                "Articulation record covers %d rig(s) not in this export "
                "(scene-wide channel, exported subset); ignored.",
                out_of_scope,
            )
        return bound
