# !/usr/bin/python
# coding=utf-8
"""Articulated rigs: carry the ``articulation`` record into the GLB.

One layout, two containers: the GLB carries the ``articulation`` record as the
FBX's carrier holds it, in its root ``extras`` (``RecordSpec.glb_verbatim``),
and binds nothing -- no joint or grabbed part to a node index. A reader
resolves each rig's joints by name in the file's own scene (one assignment
over every node, the rule every record shares: ``_media.py``,
``kernel/resolve.js``), each grabbed part by name under its joint's node, and
poses every part from its parent JOINT's pose, whatever node it hangs under.
What a reader cannot pose is said, never guessed (:meth:`_Articulation.findings`):
by this build as it carries the record, by ``MeshConvert.verify_glb`` and by
the WebXR viewer as each reads it.

One job of :class:`MeshConvert`, composed in ``_mesh_convert.py``; it reaches
the other passes through the facade it holds (``self._mc``).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

# Eager: the signatures name GlbTarget; SceneRecords is the light declaration
# module every pass reads.
from pythontk.core_utils.engines.scene_export.scene_records import SceneRecords
from pythontk.file_utils.mesh_convert.glb.edit import GlbTarget

logger = logging.getLogger(__name__)


class _Articulation:
    """Articulated rigs: the ``articulation`` record, carried.

    A private part of :class:`MeshConvert`; call it through the facade.
    """

    def __init__(self, mc: type) -> None:
        self._mc = mc  # the facade class: every name it exposes is read through it

    def _articulation_record(self, gltf: dict, warn: bool = False) -> Optional[dict]:
        """The ``articulation`` record *gltf* carries in its current shape:
        its root ``extras``' (the record as the file ships it), else its
        carrier's (before the carry; :meth:`_current_record`, which says why
        it refuses another version or shape when *warn*); ``None`` without
        one. A reader refuses rather than guesses, and the cure is a
        re-export."""
        spec = SceneRecords.ARTICULATION
        carried = SceneRecords.glb_block(gltf, spec)
        if carried is not None and not SceneRecords.problems(spec, carried):
            return carried
        if warn:
            return self._mc._current_record(gltf, spec)
        record = self._mc.data_export_channel(gltf, spec.key)
        return record if not SceneRecords.problems(spec, record) else None

    def apply_glb_articulation(self, glb: GlbTarget) -> Optional[Dict[str, Any]]:
        # ONE session for the read and the write: the record is read from the
        # very GLB being edited.
        with self._mc.open_glb(glb) as edit:
            gltf = edit.gltf
            record = self._articulation_record(gltf, warn=True)
            if not record:
                return None
            # The record as it stands, under its key in the root extras: one
            # layout, two containers (``RecordSpec.glb_verbatim``).
            record = SceneRecords.put_glb_block(gltf, SceneRecords.ARTICULATION, record)
            edit.dirty = True
            for level, text in self.findings(gltf, record):
                logger.log(level, "Articulation record: %s", text)
            return record

    def findings(
        self, gltf: dict, record: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, str]]:
        """``[(logging level, text)]``: what a reader of *gltf* cannot pose
        of its ``articulation`` record (*record*, else the one it carries),
        each said once, as every reader resolves the record:

        - a rig's joints resolved by name over every node as ONE assignment
          (the record rule, ``_Media.resolve_entries``: exactly, else by the
          name after the last ``:``): a joint two nodes answer leaves its rig
          unposed (a warning: posed from a guess, one part would turn
          another's); a rig none of whose joints is in the file is out of
          scope (an exported subset of a scene-wide record: info); a rig only
          some of whose joints are is unposed (a warning);
        - each grabbed part resolved by name over its joint's node and the
          nodes below it: one that resolves to no single node is not
          grabbable (a warning) -- a production assembly repeats part names,
          and the joint says which one is meant.

        A rig whose root joints hang under different nodes is posed: each
        root's chain is posed and solved in the space its own root hangs
        in, and no channel spans two roots, so nothing needs one space.
        """
        record = record if record is not None else self._articulation_record(gltf)
        if not isinstance(record, dict):
            return []
        media = self._mc._media
        rigs = [rig for rig in record.get("rigs") or [] if isinstance(rig, dict)]
        joints_of = [
            [j for j in rig.get("joints") or [] if isinstance(j, dict)] for rig in rigs
        ]
        resolved = media.resolved(gltf, [j for js in joints_of for j in js], meshes=False)
        index_of = {id(node): i for i, node in enumerate(gltf.get("nodes") or [])}
        found: List[Tuple[int, str]] = []
        out_of_scope = 0
        at = 0
        for rig, joints in zip(rigs, joints_of):
            mine, at = resolved[at : at + len(joints)], at + len(joints)
            name = rig.get("name")
            tied = [(e.get("name"), leaves) for e, _n, status, leaves in mine if status == "ambiguous"]
            if tied:
                joint, leaves = tied[0]
                found.append(
                    (
                        logging.WARNING,
                        f"rig {name!r}: joint {joint!r} names several nodes "
                        f"({', '.join(leaves)}) -- ambiguous, the rig is not posed.",
                    )
                )
                continue
            missing = [e.get("name") for e, nodes, _s, _l in mine if not nodes]
            if not joints or len(missing) == len(joints):
                out_of_scope += 1
                continue
            if missing:
                found.append(
                    (
                        logging.WARNING,
                        f"rig {name!r}: joints not in this file ({', '.join(map(str, missing))})"
                        " -- the rig is not posed.",
                    )
                )
                continue
            bound = [index_of[id(found_nodes[0])] for _e, found_nodes, _s, _l in mine]
            for entry in rig.get("grab") or []:
                part = entry.get("node") if isinstance(entry, dict) else None
                joint = entry.get("joint") if isinstance(entry, dict) else None
                if not (isinstance(joint, int) and 0 <= joint < len(bound)):
                    found.append(
                        (
                            logging.WARNING,
                            f"rig {name!r}: grab part {part!r} rides no joint of the "
                            f"rig ({joint!r}); it is not grabbable.",
                        )
                    )
                    continue
                ((_e, held, status, _l),) = media.resolved(
                    gltf, [{"name": part}], meshes=False, under=bound[joint]
                )
                if not held:
                    found.append(
                        (
                            logging.WARNING,
                            f"rig {name!r}: grab part {part!r} is "
                            f"{'ambiguous' if status == 'ambiguous' else 'not found'} "
                            f"under joint {joints[joint].get('name')!r}; it is not "
                            "grabbable.",
                        )
                    )
        if out_of_scope:
            found.append(
                (
                    logging.INFO,
                    f"{out_of_scope} rig(s) the record covers are not in this "
                    "file (a scene-wide record, an exported subset); a reader "
                    "passes them over.",
                )
            )
        return found
