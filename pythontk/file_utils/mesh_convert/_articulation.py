# !/usr/bin/python
# coding=utf-8
"""Articulated rigs: publish the ``articulation_web`` manifest the viewer poses.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through ``cls``.
"""

import copy
import logging
from typing import Any, Dict, List, Optional

# Eager: the channel keys below are SceneRecords' own, read at class
# definition.
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbTarget

logger = logging.getLogger(__name__)


class _ArticulationMixin:
    """Articulated rigs: the ``extras.articulation_web`` manifest.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    #: ``data_export`` channel the DCC's ``ArticulatedRig`` publishes.
    ARTICULATION_KEY = SceneRecords.ARTICULATION.key
    #: Highest ``articulation`` schema this applier reads.
    ARTICULATION_VERSION = SceneRecords.ARTICULATION.version
    #: Root-extras key the viewer's packaged ``articulated_rig`` script reads
    #: -- the record's declared web projection, shaped by ``ArticulationWeb``.
    ARTICULATION_WEB_KEY = SceneRecords.ARTICULATION.web.key
    #: Where ``articulation_web`` binds by glTF NODE INDEX -- the paths a pass
    #: that renumbers nodes (``strip_glb_curve_proxies``) must follow.
    ARTICULATION_WEB_NODE_FIELDS = (
        ("rigs", "joints", "node"),
        ("rigs", "grab", "node"),
    )

    @classmethod
    def apply_glb_articulation(cls, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        """Publish ``extras.articulation_web``: the scene's articulated rigs,
        bound to this file's nodes.

        The GLB half of the articulated-rig contract. The DCC's
        ``ArticulatedRig`` publishes the ``articulation`` record on the
        ``data_export`` carrier -- per rig its joints (node name, parent,
        rest, rotate order, channels and limits) and the parts a hand grabs.
        This reads it back out of the file and resolves every name to a glTF
        node index: a joint to the one node carrying its name (exactly, else
        by namespace-stripped leaf); a grabbed part to the node of that name
        UNDER its joint's node -- a production assembly repeats part names,
        and the joint says which one is meant. The manifest is the record
        with a ``node`` index on every joint and grab entry; the packaged
        viewer script (``preview/features/articulated_rig/``) builds its
        model from it and poses the joints.

        A rig whose joints are not all in the file (a selection export that
        left it out) is out of scope and skipped, counted once; an ambiguous
        joint name skips its rig with a warning -- a runtime must never pose a
        guess. A grab entry that resolves nowhere is dropped with a warning
        (the rig still plays its clips; that part is just not grabbable).

        Idempotent: the manifest is replaced wholesale on every run, and a run
        that binds nothing removes a stale one. The channel is left in place.

        Parameters:
            glb: ``.glb`` path (modified in place) or an open :class:`GlbEdit`.

        Returns:
            The published manifest, or ``None`` when the file carries no
            channel or no rig in it could be bound.
        """
        with cls.open_glb(glb) as edit:
            gltf = edit.gltf
            extras = gltf.get("extras")
            if not isinstance(extras, dict):
                extras = gltf["extras"] = {}
            payload = cls.data_export_channel(gltf, cls.ARTICULATION_KEY)
            rigs: List[Dict[str, Any]] = []
            if isinstance(payload, dict):
                try:
                    version = int(payload.get("version", 1))
                except (TypeError, ValueError):
                    version = -1
                if not 0 < version <= cls.ARTICULATION_VERSION:
                    # Refuse rather than guess: a newer schema may change what
                    # a field MEANS.
                    logger.warning(
                        "articulation v%r is newer than this reader (v%s); "
                        "articulated rigs not wired.",
                        payload.get("version"),
                        cls.ARTICULATION_VERSION,
                    )
                else:
                    rigs = cls._bind_articulated_rigs(gltf, payload)
            if rigs:
                manifest = {
                    "version": SceneRecords.ARTICULATION.web.version,
                    "metadata_version": version,
                    "rigs": rigs,
                }
                extras[cls.ARTICULATION_WEB_KEY] = manifest
                edit.dirty = True
                return manifest
            if extras.pop(cls.ARTICULATION_WEB_KEY, None) is not None:
                edit.dirty = True
            return None

    @classmethod
    def _bind_articulated_rigs(
        cls, gltf: dict, payload: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Each rig of *payload* with its joints and grab entries bound to
        node indices; rigs that cannot be bound are left out (see
        :meth:`apply_glb_articulation`)."""
        parents = cls._node_parents(gltf)

        def under(node: int, ancestor: int) -> bool:
            # The lineage walk ends at a cycle: a malformed file never hangs.
            return ancestor in cls._node_lineage(parents, node)[1:]

        bound: List[Dict[str, Any]] = []
        out_of_scope = 0
        for rig in payload.get("rigs") or []:
            if not isinstance(rig, dict):
                continue
            name = rig.get("name")
            joints = copy.deepcopy(rig.get("joints") or [])
            indices: List[int] = []
            for joint in joints:
                found = cls._nodes_named(gltf, joint.get("name"))
                if len(found) != 1:
                    break
                indices.append(found[0])
            if len(indices) != len(joints) or not joints:
                missing = (
                    joints[len(indices)]["name"] if len(indices) < len(joints) else None
                )
                found = cls._nodes_named(gltf, missing) if missing else []
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
                    for i in cls._nodes_named(gltf, entry.get("node"))
                    if under(i, joint_node)
                ]
                if len(candidates) != 1:
                    logger.warning(
                        "Articulated rig %r: grab part %r is %s under joint %r; "
                        "it will not be grabbable.",
                        name,
                        entry.get("node"),
                        "ambiguous" if candidates else "not found",
                        joints[joint_index].get("name"),
                    )
                    continue
                grab.append(dict(entry, node=candidates[0], name=entry.get("node")))
            bound.append(dict(rig, joints=joints, grab=grab))
        if out_of_scope:
            logger.info(
                "Articulation record covers %d rig(s) not in this export "
                "(scene-wide channel, exported subset); ignored.",
                out_of_scope,
            )
        return bound
