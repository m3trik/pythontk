# !/usr/bin/python
# coding=utf-8
"""The articulated rigs' web route: ``MeshConvert.apply_glb_articulation``, the
packaged ``articulated_rig`` viewer script and the page's ``viewer.grab``.

Two halves, in one module because they share the fixtures:

* Pure Python -- the pass over hand-authored GLBs (a five-joint arm under a
  scaled rig group, its parts, the ``data_export`` carrier): the record
  carried as it stands (one layout, two containers), and what a reader cannot
  pose said by the build and ``verify_glb``; the server's auto-activation.
* The page -- headless Edge through Playwright, the real ``viewer.html`` and
  three.js, the real ``PreviewServer``: the script's ``ArticulationModel`` port
  against every ``ArticulationConformance`` case, anchors included; joints and
  parts bound by name; a nested rig, an anchored one and a flat model each
  posed from its joint chain; a grab through ``viewer.grab`` against the
  Python solve of the same hold; a real mouse drag (which must move the rig
  and NOT orbit the camera) and its wheel depth; a headset grip on synthetic
  input; the panel's sliders and buttons; a posed rig held over its playing
  clip. Skipped, never failed, without the runtime (``pip install
  playwright`` drives the installed Edge; nothing downloads).
"""

import json
import logging
import math
import os
import struct
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pythontk as ptk  # noqa: E402
from pythontk import (
    ArticulationConformance,
    ArticulationModel,
    MeshConvert,
    PreviewServer,
)  # noqa: E402
from conftest import browser_runtime_available  # noqa: E402

ptk.TestSandbox.activate()

MESH_CONVERT_LOGGER = "pythontk.file_utils.mesh_convert"
#: Metres per unit of the conformance rigs: the record states them in metres
#: (glTF's frame), as the file holds each joint's parent frame.
UNIT = 0.01
#: The rig group's own scale -- the production prop sits under a scaled node.
GROUP_SCALE = 2.0
PARTS = ["pole_geo", "upper_geo", "rod_geo", "inner_geo", "head_geo"]


# ============================================================================
# Fixtures
# ============================================================================


def _metres(rig, unit=UNIT):
    """A conformance *rig* (its lengths in its own units) in metres, as a
    record states it: every rest and anchor translation, and each slide's
    limits."""
    rig = json.loads(json.dumps(rig))
    for joint in rig["joints"]:
        joint["t"] = [v * unit for v in joint["t"]]
        anchor = joint.get("anchor") or {"t": [0.0, 0.0, 0.0], "q": [0.0, 0.0, 0.0, 1.0]}
        joint["anchor"] = {"t": [v * unit for v in anchor["t"]], "q": list(anchor["q"])}
        for channel in joint["channels"]:
            if channel["channel"][0] == "t":
                for key in ("min", "max"):
                    if channel[key] is not None:
                        channel[key] *= unit
    return rig


def _rig_record(name="arm", base="arm", parts=None):
    """A conformance rig as the DCC records it: joint names that are node
    names, and one grabbed part per joint (*parts*, by joint; the arm's
    :data:`PARTS` by default, else ``<joint>_geo``)."""
    rig = _metres(ArticulationConformance.rigs()[base])
    rig["name"] = name
    if parts is None:
        parts = PARTS if base == "arm" else [f"{j['name']}_geo" for j in rig["joints"]]
    for joint in rig["joints"]:
        joint["name"] = f"{name}_{joint['name']}_jnt"
    rig["grab"] = [{"node": part, "joint": i} for i, part in enumerate(parts)]
    return rig


def _payload(*rigs, version=2):
    return {"version": version, "rigs": list(rigs)}


def _data_export_node(payload):
    """The carrier exactly as FBX2glTF transcribes a Maya ``data_export``."""
    return {
        "name": "data_export",
        "extras": {
            "fromFBX": {
                "userProperties": {
                    ptk.SceneRecords.ARTICULATION.key: {
                        "type": "eFbxString",
                        "value": json.dumps(payload),
                    }
                }
            }
        },
    }


def _arm_nodes(record, payload=None, extra_parts=(), rename=None):
    """glTF nodes: the scaled rig group, its joints nested at rest (each
    under its parent joint's node, the record's metres), a cube part under
    each (its grabbed part), and the carrier. *rename* spells a joint's node
    (a namespace an import keeps)."""
    rename = rename or (lambda name: name)
    nodes = [
        {
            "name": f"{record['name']}_RIG",
            "scale": [GROUP_SCALE] * 3,
            "translation": [0.1, 0.0, -0.05],
            "children": [],
        }
    ]
    joint_nodes = []
    for joint in record["joints"]:
        index = len(nodes)
        nodes.append(
            {
                "name": rename(joint["name"]),
                "translation": list(joint["t"]),
                "rotation": list(joint["q"]),
                "children": [],
            }
        )
        joint_nodes.append(index)
        parent = 0 if joint["parent"] is None else joint_nodes[joint["parent"]]
        nodes[parent]["children"].append(index)
    for entry in record["grab"]:
        index = len(nodes)
        nodes.append(
            {
                "name": entry["node"],
                "mesh": index,
                "translation": [0.02, 0.0, 0.0],
                "scale": [0.03, 0.012, 0.012],
            }
        )
        nodes[joint_nodes[entry["joint"]]]["children"].append(index)
    for name, joint in extra_parts:
        index = len(nodes)
        nodes.append({"name": name, "mesh": index, "scale": [0.01] * 3})
        nodes[joint_nodes[joint]]["children"].append(index)
    nodes.append(_data_export_node(payload or _payload(record)))
    return nodes


def _flat_nodes(record):
    """A flat model: every joint's node -- its part -- a child of the scaled
    rig group, unparented from its parent joint (the chain is the record's
    alone), at its rest in that group (the flat rig's rests are all the
    identity), a cube under it; each grabbed part is its joint's own node."""
    nodes = [
        {
            "name": f"{record['name']}_RIG",
            "scale": [GROUP_SCALE] * 3,
            "translation": [0.1, 0.0, -0.05],
            "children": [],
        }
    ]
    for joint in record["joints"]:
        index = len(nodes)
        nodes.append({"name": joint["name"], "children": [index + 1]})
        nodes[0]["children"].append(index)
        nodes.append(
            {
                "name": f"{joint['name']}_mesh",
                "mesh": index + 1,
                "translation": [v for v in joint["anchor"]["t"]],
                "scale": [0.02, 0.02, 0.02],
            }
        )
    nodes.append(_data_export_node(_payload(record)))
    return nodes


def _write_glb(path, nodes, animations=None):
    """A minimal valid binary glTF: a unit cube for every part node."""
    positions = struct.pack(
        "<24f",
        *[
            c
            for x in (-0.5, 0.5)
            for y in (-0.5, 0.5)
            for z in (-0.5, 0.5)
            for c in (x, y, z)
        ],
    )
    quads = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    tris = [i for a, b, c, d in quads for i in (a, b, c, a, c, d)]
    indices = struct.pack(f"<{len(tris)}H", *tris)
    chunks = [positions, indices]
    anim_accessors = []
    if animations:
        for times, values in animations["data"]:
            chunks.append(struct.pack(f"<{len(times)}f", *times))
            chunks.append(struct.pack(f"<{len(values)}f", *values))
    views, blob, offset = [], b"", 0
    for raw in chunks:
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw)})
        pad = (4 - len(raw) % 4) % 4
        blob += raw + b"\0" * pad
        offset += len(raw) + pad
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "count": 8,
            "type": "VEC3",
            "min": [-0.5, -0.5, -0.5],
            "max": [0.5, 0.5, 0.5],
        },
        {"bufferView": 1, "componentType": 5123, "count": len(tris), "type": "SCALAR"},
    ]
    if animations:
        view = 2
        for times, values in animations["data"]:
            accessors.append(
                {
                    "bufferView": view,
                    "componentType": 5126,
                    "count": len(times),
                    "type": "SCALAR",
                    "min": [min(times)],
                    "max": [max(times)],
                }
            )
            accessors.append(
                {
                    "bufferView": view + 1,
                    "componentType": 5126,
                    "count": len(values) // 4,
                    "type": "VEC4",
                }
            )
            anim_accessors.append((len(accessors) - 2, len(accessors) - 1))
            view += 2
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [
            {
                "nodes": [
                    i
                    for i, n in enumerate(nodes)
                    if not any(i in (m.get("children") or []) for m in nodes)
                ]
            }
        ],
        "nodes": nodes,
        # One cube per mesh index the nodes name: a mesh shared by several
        # nodes shares the loader's association, which names one of them.
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}
            for _ in range(1 + max([n.get("mesh", 0) for n in nodes] or [0]))
        ],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    if animations:
        gltf["animations"] = [
            {
                "name": animations["name"],
                "samplers": [
                    {"input": i, "output": o, "interpolation": "LINEAR"}
                    for i, o in anim_accessors
                ],
                "channels": [
                    {"sampler": k, "target": {"node": node, "path": "rotation"}}
                    for k, node in enumerate(animations["nodes"])
                ],
            }
        ]
    json_bytes = json.dumps(gltf).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(blob)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes)
        fh.write(struct.pack("<I4s", len(blob), b"BIN\0") + blob)
    return path


def _extras(path):
    with MeshConvert.open_glb(str(path)) as edit:
        return edit.gltf.get("extras") or {}


def _findings(path):
    """What a reader of the GLB at *path* cannot pose, as the build and
    ``verify_glb`` say it."""
    with MeshConvert.open_glb(str(path)) as edit:
        return MeshConvert._articulation.findings(edit.gltf)


def _said(findings, level=None):
    return "\n".join(text for lvl, text in findings if level is None or lvl == level)


# ============================================================================
# The pass
# ============================================================================


class TestApplyGlbArticulation(unittest.TestCase):
    def setUp(self):
        self.temp = ptk.TempArtifacts("articulation_web", policy="scoped")

    def tearDown(self):
        self.temp.cleanup()

    def _glb(self, nodes):
        return _write_glb(self.temp.path(extension=".glb"), nodes)

    def test_the_glb_carries_the_record_as_it_stands(self):
        """One layout, two containers (articulation v2, 2026-11-08): the GLB
        carries the FBX's record itself in its root extras -- its version,
        its joints by name, no node index anywhere -- and a reader resolves
        it in its own scene."""
        record = _rig_record()
        path = self._glb(_arm_nodes(record))
        carried = MeshConvert.apply_glb_articulation(path)
        self.assertEqual(carried, _payload(record))
        self.assertEqual(_extras(path)[ptk.SceneRecords.ARTICULATION.key], carried)
        rig = carried["rigs"][0]
        self.assertEqual(
            [j["name"] for j in rig["joints"]], [j["name"] for j in record["joints"]]
        )
        self.assertEqual([g["node"] for g in rig["grab"]], PARTS)
        for joint in rig["joints"]:
            self.assertNotIn("node", joint)
        self.assertEqual(_findings(path), [], "every joint and part resolves")

    def test_the_carried_record_is_the_declared_shape(self):
        """The pass's output IS ``ArticulationRecord``, at the record's own
        version: the declaration the runtimes' types are generated from is
        held to what the pipeline really writes."""
        record = _rig_record()
        self.assertEqual(ptk.ArticulationRecord.validate(_payload(record)).errors, [])
        carried = MeshConvert.apply_glb_articulation(self._glb(_arm_nodes(record)))
        res = ptk.ArticulationRecord.validate(carried)
        self.assertEqual((res.errors, res.warnings), ([], []))
        self.assertEqual(carried["version"], ptk.SceneRecords.ARTICULATION.version)

    def test_a_grabbed_part_resolves_under_its_own_joint_when_the_name_repeats(self):
        """A production assembly repeats part names: the joint says which one
        is meant -- the part of that name under its joint's node."""
        record = _rig_record()
        nodes = _arm_nodes(record, extra_parts=[("head_geo", 0)])  # a namesake under joint 0
        path = self._glb(nodes)
        MeshConvert.apply_glb_articulation(path)
        self.assertEqual(_findings(path), [])
        with MeshConvert.open_glb(str(path)) as edit:
            joint = next(i for i, n in enumerate(nodes) if n.get("name") == "arm_head_jnt")
            index = {id(n): i for i, n in enumerate(edit.gltf["nodes"])}
            ((_e, found, status, _l),) = MeshConvert._media.resolved(
                edit.gltf, [{"name": "head_geo"}], meshes=False, under=joint
            )
            self.assertEqual(status, "exact")
            self.assertEqual([index[id(n)] for n in found], [10])  # the head joint's own
            ((_e, both, tied, _l),) = MeshConvert._media.resolved(
                edit.gltf, [{"name": "head_geo"}], meshes=False
            )
            self.assertEqual((both, tied), (None, "ambiguous"), "over the whole file: a tie")

    def test_an_ambiguous_joint_is_carried_and_said_and_its_rig_not_posed(self):
        """A joint two nodes answer: no reader poses its rig (a guess would
        turn one part for another), the build says so -- and the record
        rides as it stands, as every record does."""
        record = _rig_record()
        nodes = _arm_nodes(record)
        nodes.insert(len(nodes) - 1, {"name": record["joints"][2]["name"]})
        path = self._glb(nodes)
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING") as logs:
            carried = MeshConvert.apply_glb_articulation(path)
        self.assertEqual(carried, _payload(record))
        said = "\n".join(logs.output)
        self.assertIn("ambiguous", said)
        self.assertIn(record["joints"][2]["name"], said)
        self.assertIn("not posed", said)

    def test_a_rig_whose_roots_hang_under_different_nodes_is_posed(self):
        """Retired refusal (2026-11-08): a rig poses each part from its joint
        chain, each root's chain in the space its root hangs in, and no
        channel spans two roots -- so a rig whose roots hang under different
        nodes needs no one space, and nothing is said of it (it was refused
        by the build, the page and unitytk's controller since 2026-10-10)."""
        record = _rig_record()
        record["joints"][1]["parent"] = None  # the elbow: a second root
        nodes = _arm_nodes(record)
        nodes[0]["children"].remove(2)  # off the rig group...
        nodes.append({"name": "other_RIG", "children": [2]})  # ...onto another
        path = self._glb(nodes)
        with self.assertNoLogs(MESH_CONVERT_LOGGER, level="WARNING"):
            carried = MeshConvert.apply_glb_articulation(path)
        self.assertEqual(carried["rigs"][0]["joints"][1]["parent"], None)
        self.assertEqual(_findings(path), [])

    def test_a_rig_not_in_the_file_is_carried_and_passed_over(self):
        """A scene-wide record carries a rig the export left out: the record
        rides whole, and the rig is out of scope -- said once, at info."""
        record = _rig_record()
        other = _rig_record("other")
        path = self._glb(_arm_nodes(record, payload=_payload(record, other)))
        with self.assertLogs(MESH_CONVERT_LOGGER, level="INFO") as logs:
            carried = MeshConvert.apply_glb_articulation(path)
        self.assertEqual([r["name"] for r in carried["rigs"]], ["arm", "other"])
        self.assertIn("1 rig(s) the record covers are not in this file", "\n".join(logs.output))
        self.assertEqual(_said(_findings(path), logging.WARNING), "")

    def test_a_rig_only_partly_in_the_file_is_said_and_not_posed(self):
        record = _rig_record()
        nodes = _arm_nodes(record)
        nodes[3]["name"] = "renamed"  # the collar joint's node
        path = self._glb(nodes)
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING") as logs:
            MeshConvert.apply_glb_articulation(path)
        said = "\n".join(logs.output)
        self.assertIn("joints not in this file (arm_collar_jnt)", said)

    def test_a_newer_schema_is_refused_not_guessed(self):
        record = _rig_record()
        path = self._glb(_arm_nodes(record, payload=_payload(record, version=99)))
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING"):
            self.assertIsNone(MeshConvert.apply_glb_articulation(path))

    def test_a_v1_record_is_refused_with_its_fix(self):
        """A version 1 record -- no anchors, the file's own units, bound by
        index on the GLB -- is no record a reader reads: refused and said,
        the cure a re-export."""
        record = _rig_record()
        for joint in record["joints"]:
            del joint["anchor"]
        path = self._glb(_arm_nodes(record, payload=_payload(record, version=1)))
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING") as logs:
            self.assertIsNone(MeshConvert.apply_glb_articulation(path))
        said = "\n".join(logs.output)
        self.assertIn("v1 is not the v2", said)
        self.assertIn("re-export", said)
        self.assertNotIn(ptk.SceneRecords.ARTICULATION.key, _extras(path))

    def test_a_joint_that_states_no_parent_is_refused_and_said(self):
        """Every joint states its ``parent`` -- ``null`` for a root -- in the
        record's declared shape (``ArticulationRecord``). A record leaving it
        out is none a reader reads: refused whole and said, where it was read
        as a root until 2026-10-20 -- the cure is a re-export."""
        record = _rig_record()
        written = json.loads(json.dumps(record))
        del written["joints"][0]["parent"]
        path = self._glb(_arm_nodes(record, payload=_payload(written)))
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING") as logs:
            self.assertIsNone(MeshConvert.apply_glb_articulation(path))
        said = "\n".join(logs.output)
        self.assertIn(f"{ptk.SceneRecords.ARTICULATION.key} record not applied", said)
        self.assertIn("missing required key 'parent'", said)

    def test_rerunning_is_idempotent_and_the_carried_record_stands(self):
        """The record is written whole on every run, its root copy read
        first: a second run changes nothing, and a file whose carrier left
        (a finished GLB) still carries its record."""
        record = _rig_record()
        path = self._glb(_arm_nodes(record))
        first = MeshConvert.apply_glb_articulation(path)
        self.assertEqual(MeshConvert.apply_glb_articulation(path), first)
        with MeshConvert.open_glb(str(path)) as edit:
            edit.gltf["nodes"] = [
                n for n in edit.gltf["nodes"] if n.get("name") != "data_export"
            ]
            edit.dirty = True
        self.assertEqual(MeshConvert.apply_glb_articulation(path), first)
        self.assertEqual(_extras(path)[ptk.SceneRecords.ARTICULATION.key], first)

    def test_a_grab_part_missing_under_its_joint_is_said(self):
        record = _rig_record()
        written = json.loads(json.dumps(record))
        written["grab"].append({"node": "ghost_geo", "joint": 1})
        path = self._glb(_arm_nodes(record, payload=_payload(written)))
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING") as logs:
            carried = MeshConvert.apply_glb_articulation(path)
        said = "\n".join(logs.output)
        self.assertIn("grab part 'ghost_geo' is not found under joint 'arm_elbow_jnt'", said)
        self.assertEqual(
            [g["node"] for g in carried["rigs"][0]["grab"]], PARTS + ["ghost_geo"]
        )

    def test_a_flat_models_parts_are_their_joints_and_grabbable(self):
        """A flat model: every part under the root, chained by the record's
        parents alone, each grabbed part its joint's own node -- found under
        that node, itself included."""
        record = _rig_record("flat", base="flat", parts=[])
        record["grab"] = [{"node": j["name"], "joint": i} for i, j in enumerate(record["joints"])]
        path = _write_glb(self.temp.path(extension=".glb"), _flat_nodes(record))
        MeshConvert.apply_glb_articulation(path)
        self.assertEqual(_findings(path), [])

    def test_a_cyclic_hierarchy_elsewhere_does_not_hang_the_pass(self):
        """A malformed file whose ``children`` loop must not trap the walks
        that find a grabbed part under its joint and a node's lineage."""
        record = _rig_record()
        nodes = _arm_nodes(record)
        a, b = len(nodes), len(nodes) + 1
        nodes += [
            {"name": "head_geo", "children": [b]},
            {"name": "loop", "children": [a]},
        ]
        path = self._glb(nodes)
        result = {}

        def run():
            result["carried"] = MeshConvert.apply_glb_articulation(path)
            result["findings"] = _findings(path)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(30)
        self.assertFalse(worker.is_alive(), "the pass looped on the cycle")
        self.assertEqual(result["findings"], [], "the head part is its joint's own")

    def test_a_node_strip_leaves_the_record_naming_its_nodes(self):
        """The conversion strips render-effect curve proxies AFTER this pass,
        renumbering every node behind one. The record names its nodes, so
        nothing follows the renumber: it stands as carried, and every joint
        and part still resolves (the index-bound block's renumber pass,
        ``GlbBlock.node_fields``, retired with it)."""
        record = _rig_record()
        nodes = _arm_nodes(record)
        for node in nodes:
            if "children" in node:
                node["children"] = [i + 1 for i in node["children"]]
        proxy = {
            "name": "Fx__highlight",
            "extras": {
                "fromFBX": {
                    "userProperties": {
                        MeshConvert.CURVE_PROXY_MARKER: {
                            "type": "eFbxBool",
                            "value": True,
                        }
                    }
                }
            },
        }
        path = self._glb([proxy] + nodes)
        carried = MeshConvert.apply_glb_articulation(path)
        MeshConvert.strip_glb_curve_proxies(path)
        self.assertEqual(_extras(path)[ptk.SceneRecords.ARTICULATION.key], carried)
        self.assertEqual(_findings(path), [])
        self.assertFalse(hasattr(MeshConvert, "_renumber_block_nodes"))

    def test_verify_glb_notes_what_a_reader_cannot_pose(self):
        """A recipient's check says what the rigs' readers refuse -- an
        ambiguous joint, a rig not wholly in the file, a part not grabbable --
        as notes under the record's address: the file is readable, those
        rigs or parts inert. Added: 2026-11-08"""
        record = _rig_record()
        written = json.loads(json.dumps(record))
        written["grab"].append({"node": "ghost_geo", "joint": 1})
        tied = _rig_record("tied")
        nodes = _arm_nodes(record, payload=_payload(written, tied, _rig_record("gone")))
        tied_nodes = _arm_nodes(tied)[:-1]
        shift = len(nodes) - 1
        for node in tied_nodes:
            if "children" in node:
                node["children"] = [i + shift for i in node["children"]]
        nodes[-1:-1] = tied_nodes + [{"name": tied["joints"][0]["name"]}]
        path = self._glb(nodes)
        MeshConvert.apply_glb_articulation(path)
        with MeshConvert.open_glb(path) as edit:
            MeshConvert.embed_glb_handoff(
                edit, ptk.HandoffRecord.block("glb", [], {"application": "maya"})
            )
        report = MeshConvert.verify_glb(path)
        where = f"extras.{ptk.SceneRecords.ARTICULATION.key}: "
        notes = [n for n in report["notes"] if n.startswith(where)]
        said = "\n".join(notes)
        self.assertEqual(len(notes), 3, report["notes"])
        self.assertIn("rig 'tied': joint 'tied_base_jnt' names several nodes", said)
        self.assertIn("rig 'arm': grab part 'ghost_geo' is not found", said)
        self.assertIn("1 rig(s) the record covers are not in this file", said)
        self.assertFalse(
            [p for p in report["problems"] if p.startswith(where)], report["problems"]
        )

    def test_the_handoff_names_the_record(self):
        """A standalone reader is told the record is there, in the artifact:
        the handoff of a file carrying it names it -- its ``assembly`` row,
        addressed at the record under its key, saying what it means in the
        record's own terms, as the FBX's handoff does."""
        path = self._glb(_arm_nodes(_rig_record()))
        carried = MeshConvert.apply_glb_articulation(path)
        self.assertIsNotNone(carried)
        with MeshConvert.open_glb(path) as edit:
            MeshConvert.embed_glb_handoff(
                edit, ptk.HandoffRecord.block("glb", [], {"application": "maya"})
            )
        handoff = MeshConvert.read_glb_handoff(path)
        record = ptk.SceneRecords.ARTICULATION
        (row,) = [r for r in handoff["assembly"] if r["data"] == [f"extras.{record.key}"]]
        self.assertEqual(row["part"], record.key)
        self.assertEqual(row["spec"], record.assembly_for(carried))


class TestServerAutoActivation(unittest.TestCase):
    def setUp(self):
        self.temp = ptk.TempArtifacts("articulation_server", policy="scoped")
        self.server = PreviewServer(root=self.temp.dir_path(), port=0, viewer=False)

    def tearDown(self):
        self.server.stop()
        self.temp.cleanup()

    def test_a_rig_deliverable_activates_the_script(self):
        path = _write_glb(self.temp.path(extension=".glb"), _arm_nodes(_rig_record()))
        MeshConvert.apply_glb_articulation(path)
        self.server.publish(path)
        self.assertIn("articulated_rig", self.server.scripts)

    def test_a_plain_deliverable_activates_nothing_of_it(self):
        path = _write_glb(self.temp.path(extension=".glb"), _arm_nodes(_rig_record()))
        self.server.publish(path)  # the channel alone is not the record carried
        self.assertNotIn("articulated_rig", self.server.scripts)


# ============================================================================
# The page
# ============================================================================


#: The probe: imports the feature's model port by its own served path (as an
#: app vendoring it would), waits for the script's session on load, runs the
#: test's action, and leaves helpers on `window.__probe` for the Python side to
#: call between real input events.
PROBE_JS = """
import { ArticulationModel } from '../features/articulated_rig/model.js';

export default function probe(viewer) {
  const report = { ready: false, errors: [], loads: 0 };
  window.__probe = report;
  const THREE = viewer.THREE;
  const waiters = [];
  viewer.on('frame', () => { for (const fn of waiters.splice(0)) fn(); });
  const frames = (n) => new Promise((resolve) => {
    let left = n;
    const tick = () => { left -= 1; if (left <= 0) resolve(); else waiters.push(tick); };
    waiters.push(tick);
  });
  report.frames = frames;
  // The held point of a rig's hold, in WORLD space: its chain's space.
  report.worldOf = (rig, joint, local) => {
    const p = rig.model.point(rig.state, joint, local);
    const space = rig.spaces[joint];
    space.updateWorldMatrix(true, false);
    return space.localToWorld(new THREE.Vector3(...p)).toArray();
  };
  // Each joint node's world pose against the one its joint chain gives
  // (the chain's space times the model's world pose): [distance, quaternion
  // distance] per joint, scale left out (a part keeps its own).
  report.chainErrors = (rig) => rig.nodes.map((node, i) => {
    const [p, q] = rig.model.world(rig.state)[i];
    const space = rig.spaces[i];
    space.updateWorldMatrix(true, false);
    node.updateWorldMatrix(true, false);
    const want = new THREE.Matrix4().compose(new THREE.Vector3(...p), new THREE.Quaternion(...q), new THREE.Vector3(1, 1, 1))
      .premultiply(space.matrixWorld);
    const wp = new THREE.Vector3(); const wq = new THREE.Quaternion(); const ws = new THREE.Vector3();
    const gp = new THREE.Vector3(); const gq = new THREE.Quaternion(); const gs = new THREE.Vector3();
    want.decompose(wp, wq, ws);
    node.matrixWorld.decompose(gp, gq, gs);
    const a = wq.toArray(); const b = gq.toArray();
    const qd = Math.min(Math.hypot(...a.map((v, k) => v - b[k])), Math.hypot(...a.map((v, k) => v + b[k])));
    return [wp.distanceTo(gp), qd];
  });
  report.setSliders = async (rig, values) => {
    const inputs = [...rig.panel.element.querySelectorAll('.slider input')];
    inputs.forEach((input, i) => {
      if (values[i] === undefined) return;
      input.value = String(values[i]);
      input.dispatchEvent(new Event('input'));
    });
    await frames(2);
  };
  // What the script holds on the page: grabbable objects, joint axes in the
  // scene, and its sections of the Rig window.
  report.counts = () => {
    let axes = 0;
    viewer.model?.traverse((o) => { if (o.type === 'AxesHelper') axes += 1; });
    const sections = [...document.querySelectorAll('*')]
      .filter((e) => e.childElementCount === 0 && e.textContent.startsWith('Articulated rig: ')).length;
    return { grabs: viewer.grab.count, axes, sections, loads: report.loads };
  };
  report.snapshot = () => {
    const session = viewer.model?.userData.articulatedRig;
    const rig = session?.rigs[0];
    return {
      state: rig ? rig.state.slice() : null,
      posed: rig ? rig.posed : null,
      holding: viewer.grab.holding,
      camera: viewer.camera.position.toArray(),
      nodes: rig ? rig.nodes.map((n) => [n.position.toArray(), n.quaternion.toArray()]) : null,
    };
  };

  viewer.on('load', async ({ model }) => {
    report.loads += 1;
    try {
      await frames(2);
      const session = model.userData.articulatedRig;
      report.hasSession = !!session;
      if (!session) { report.ready = true; return; }
      const ctx = { viewer, THREE, model, session, report, frames, ArticulationModel };
      __ACTION__
      report.ready = true;
    } catch (error) {
      report.errors.push(String(error && error.stack || error));
      report.ready = true;
    }
  });
}
"""

#: Every conformance case through the JS port; the worst difference per
#: quantity (quaternions sign-insensitive), reported for Python to judge.
CONFORMANCE_JS = """
{
  const doc = __DOC__;
  const worst = { pose: 0, world: 0, read: 0, solve: 0 };
  const qd = (a, b) => Math.min(
    Math.hypot(...a.map((v, i) => v - b[i])), Math.hypot(...a.map((v, i) => v + b[i])));
  const vd = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
  const rigs = new Set();
  for (const c of doc.cases) {
    rigs.add(c.rig);
    const m = new ArticulationModel(doc.rigs[c.rig]);
    m.pose(c.state).forEach(([t, q], i) => {
      worst.pose = Math.max(worst.pose, vd(t, c.pose[i][0]), qd(q, c.pose[i][1]));
    });
    m.world(c.state).forEach(([p, q], i) => {
      worst.world = Math.max(worst.world, vd(p, c.world[i][0]), qd(q, c.world[i][1]));
    });
    const read = m.read(c.read.locals, c.read.hint);
    worst.read = Math.max(worst.read, vd(read, c.read.expect));
    for (const s of c.solve) {
      const got = m.solve(s.from, s.joint, s.local, s.target, s.rotation);
      worst.solve = Math.max(worst.solve, vd(got, s.expect));
    }
  }
  report.worst = worst;
  report.count = doc.cases.length;
  report.rigs = [...rigs];
  report.scaleOf = typeof ArticulationModel.prototype.scaleOf;
}
"""

#: A grab through `viewer.grab`: aim at the head part's centre, take it, move
#: the point, and report what the Python side needs to solve the same hold.
API_GRAB_JS = """
{
  const rig = session.rigs[0];
  const entry = rig.grab.find((g) => g.name === '__PART__');
  const centre = new THREE.Box3().setFromObject(entry.node).getCenter(new THREE.Vector3());
  const origin = centre.clone().add(new THREE.Vector3(0, 0, 3));
  const direction = centre.clone().sub(origin).normalize();
  const hit = viewer.grab.pick(origin, direction);
  report.picked = hit ? hit.object.name : null;
  report.start = rig.state.slice();
  report.pressed = viewer.grab.press('mouse', origin, direction);
  const hold = rig.holds.get('mouse');
  report.joint = hold.joint;
  report.local = hold.local;
  report.rotation = hold.rotation;
  const target = hit.point.clone().add(new THREE.Vector3(0.03, 0.02, -0.02));
  viewer.grab.drag('mouse', target, null);
  report.target = target.toArray();
  // the same target in its chain's space, metres as the file holds it: what
  // Python solves to
  const space = rig.spaces[hold.joint];
  space.updateWorldMatrix(true, false);
  report.targetRig = space.worldToLocal(target.clone()).toArray();
  report.after = rig.state.slice();
  report.held = report.worldOf(rig, hold.joint, hold.local);
  report.chain = report.chainErrors(rig);
  viewer.grab.release('mouse');
  await frames(3);
  report.kept = rig.state.slice();
  report.posed = rig.posed;
  report.hasUnit = 'unit' in rig;
  report.nodeRotation = rig.nodes[1].quaternion.toArray();
}
"""

#: A headset grip on synthetic input: squeeze with the hand inside the head
#: part, move the hand, let go.
XR_GRAB_JS = """
{
  const rig = session.rigs[0];
  const entry = rig.grab.find((g) => g.name === 'head_geo');
  const centre = new THREE.Box3().setFromObject(entry.node).getCenter(new THREE.Vector3());
  const pose = (p) => ({ position: p.toArray(), orientation: [0, 0, 0, 1] });
  // Outside a session the rig's place is the origin: tracked == scene.
  viewer.grab.step({ grips: { right: pose(centre) }, squeeze: { right: true } });
  report.xrHolding = viewer.grab.holding;
  const moved = centre.clone().add(new THREE.Vector3(-0.02, 0.03, 0.01));
  viewer.grab.step({ grips: { right: pose(moved) }, squeeze: { right: true } });
  const hold = rig.holds.get('right');
  report.xrHeld = report.worldOf(rig, hold.joint, hold.local);
  report.xrTarget = moved.toArray();
  viewer.grab.step({ grips: { right: pose(moved) }, squeeze: { right: false } });
  report.xrReleased = viewer.grab.holding;
}
"""

#: The mouse path: where the head part is on screen, for Playwright to press.
SCREEN_JS = """
{
  const rig = session.rigs[0];
  const entry = rig.grab.find((g) => g.name === 'head_geo');
  const centre = new THREE.Box3().setFromObject(entry.node).getCenter(new THREE.Vector3());
  const ndc = centre.clone().project(viewer.camera);
  const rect = viewer.renderer.domElement.getBoundingClientRect();
  report.screen = [rect.left + (ndc.x + 1) / 2 * rect.width, rect.top + (1 - ndc.y) / 2 * rect.height];
  report.before = report.snapshot();
  // Where the held point is, in world space, while a hold lasts.
  report.heldPoint = () => {
    const hold = rig.holds.get('mouse');
    return hold ? report.worldOf(rig, hold.joint, hold.local) : null;
  };
}
"""

#: A slider: the first channel's slider set by an input event, as a user
#: dragging it would.
SLIDER_JS = """
{
  const rig = session.rigs[0];
  const input = rig.panel.element.querySelector('.slider input');
  report.panelShown = rig.panel.shown;
  report.sliderCount = rig.panel.element.querySelectorAll('.slider input').length;
  report.sliderLabels = [...rig.panel.element.querySelectorAll('.controls .key')].map((n) => n.textContent);
  report.grabNames = rig.grab.map((g) => g.name);
  report.grabParents = rig.grab.map((g) => g.node.parent?.name ?? null);
  report.sliderTexts = [...rig.panel.element.querySelectorAll('.slider output')].map((n) => n.textContent);
  input.value = '37';
  input.dispatchEvent(new Event('input'));
  await frames(2);
  report.sliderState = rig.state.slice();
  report.sliderPosed = rig.posed;
  // A slide off 0: its text in centimetres, off the record's metres.
  const slide = report.sliderLabels.indexOf('extend tx');
  if (slide >= 0) {
    const inputs = rig.panel.element.querySelectorAll('.slider input');
    inputs[slide].value = '0.021';
    inputs[slide].dispatchEvent(new Event('input'));
    await frames(8);
    report.slideValue = rig.state[slide];
    report.slideText = rig.panel.element.querySelectorAll('.slider output')[slide].textContent;
  }
}
"""

#: The panel's buttons and rows: Reset, Return to animation, Joint axes, and
#: the grab / mode rows they leave.
PANEL_JS = """
{
  const rig = session.rigs[0];
  const buttons = Object.fromEntries([...rig.panel.element.querySelectorAll('button')].map((b) => [b.textContent, b]));
  report.buttons = Object.keys(buttons);
  const rows = () => {
    const kids = [...rig.panel.element.querySelector('.rows:not(.controls)').children];
    const out = {};
    for (let i = 0; i + 1 < kids.length; i += 2) out[kids[i].textContent] = kids[i + 1].textContent;
    return out;
  };
  await report.setSliders(rig, [40]);
  await frames(40);
  report.posedRows = rows();
  buttons['Reset'].click();
  await frames(2);
  report.resetState = rig.state.slice();
  report.resetPosed = rig.posed;
  report.resetChain = report.chainErrors(rig);
  buttons['Return to animation'].click();
  await frames(40);
  report.returnedPosed = rig.posed;
  report.returnedRows = rows();
  report.axesBefore = rig.helpers.map((h) => h.visible);
  buttons['Joint axes'].click();
  report.axesOn = rig.helpers.map((h) => h.visible);
  report.axesParents = rig.helpers.map((h) => h.parent === rig.nodes[rig.helpers.indexOf(h)]);
  report.axesAt = rig.helpers.map((h) => [h.position.toArray(), Math.max(...h.geometry.attributes.position.array.map(Math.abs))]);
  buttons['Joint axes'].click();
  report.axesOff = rig.helpers.map((h) => h.visible);
}
"""

#: A posed rig over a playing clip: the pose holds after the mixer, frame
#: after frame, until play is pressed again, which hands the rig back.
CLIP_JS = """
{
  const rig = session.rigs[0];
  const quat = () => rig.nodes[0].quaternion.toArray();
  const moved = (a, b) => a.some((v, j) => Math.abs(v - b[j]) > 1e-6);
  viewer.setPlaying(true);
  await frames(3);
  const a = quat(); await frames(12); report.clipMoves = moved(a, quat());
  await report.setSliders(rig, [-30]);
  const posed = rig.nodes.map((n) => n.quaternion.toArray());
  await frames(12);
  report.playing = viewer.playing;
  report.heldOverClip = rig.nodes.every((n, i) => !moved(n.quaternion.toArray(), posed[i]));
  report.posedWhilePlaying = rig.posed;
  viewer.setPlaying(false);
  await frames(2);
  viewer.setPlaying(true);
  await frames(2);
  report.handedBack = !rig.posed;
  const b = quat(); await frames(12); report.clipMovesAgain = moved(b, quat());
}
"""

#: A rig posed from its joint chain: random channel values through the
#: sliders, then each joint node's world pose against its chain's.
CHAIN_JS = """
{
  const rig = session.rigs[0];
  await report.setSliders(rig, __VALUES__);
  report.state = rig.state.slice();
  report.chain = report.chainErrors(rig);
  report.nested = rig.nested.slice();
  report.locals = rig.nodes.map((n) => [n.position.toArray(), n.quaternion.toArray()]);
  report.spaces = rig.spaces.map((s) => s.name);
  report.scales = rig.nodes.map((n) => n.scale.toArray());
  // Each joint's axes: where on its node, turned how, and how long.
  report.axes = rig.helpers.map((h) => {
    const at = h.geometry.attributes.position.array;
    return [h.position.toArray(), h.quaternion.toArray(), Math.max(...at.map(Math.abs))];
  });
}
"""

#: A hand turning as it moves: the head taken through ``viewer.grab`` and
#: dragged with a world turn, and what Python needs to solve the same move --
#: the held joint's chain space's world rotation, which the turn is seen from.
TURN_GRAB_JS = """
{
  const rig = session.rigs[0];
  const entry = rig.grab.find((g) => g.name === 'head_geo');
  const centre = new THREE.Box3().setFromObject(entry.node).getCenter(new THREE.Vector3());
  const origin = centre.clone().add(new THREE.Vector3(0, 0, 3));
  const direction = centre.clone().sub(origin).normalize();
  const hit = viewer.grab.pick(origin, direction);
  report.start = rig.state.slice();
  report.pressed = viewer.grab.press('mouse', origin, direction);
  const hold = rig.holds.get('mouse');
  report.joint = hold.joint;
  report.local = hold.local;
  report.rotation = hold.rotation;
  const turn = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0.3, 0.9, 0.2).normalize(), 0.5);
  const target = hit.point.clone().add(new THREE.Vector3(0.01, 0.015, -0.01));
  viewer.grab.drag('mouse', target, turn);
  report.turn = turn.toArray();
  const space = rig.spaces[hold.joint];
  space.updateWorldMatrix(true, false);
  report.targetRig = space.worldToLocal(target.clone()).toArray();
  const q = new THREE.Quaternion();
  space.matrixWorld.decompose(new THREE.Vector3(), q, new THREE.Vector3());
  report.spaceTurn = q.toArray();
  const first = new THREE.Quaternion();
  rig.spaces[0].updateWorldMatrix(true, false);
  rig.spaces[0].matrixWorld.decompose(new THREE.Vector3(), first, new THREE.Vector3());
  report.firstSpaceTurn = first.toArray();
  report.after = rig.state.slice();
  viewer.grab.release('mouse');
}
"""

#: What the script built: its rigs, their spaces and grabs, nothing else.
BUILT_JS = """
{
  report.rigs = session.rigs.map((rig) => ({
    name: rig.record.name,
    joints: rig.nodes.map((n) => n.name),
    grab: rig.grab.map((g) => g.name),
    spaces: rig.spaces.map((s) => s.name),
  }));
}
"""


@unittest.skipUnless(
    browser_runtime_available(), "needs playwright + an installed Edge channel"
)
class TestArticulatedRigLive(unittest.TestCase):
    """The script and the page's grab, in the real page."""

    @classmethod
    def setUpClass(cls):
        cls.temp = ptk.TempArtifacts("articulation_web_live", policy="scoped")
        cls.record = _rig_record()
        cls.glb = cls._built(_arm_nodes(cls.record))

    @classmethod
    def _built(cls, nodes, animations=None):
        path = _write_glb(cls.temp.path(extension=".glb"), nodes, animations)
        MeshConvert.apply_glb_articulation(path)
        return path

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _load(self, action, gestures=None, glb=None, session=True):
        """Load *glb* (the class's fixture by default) with *action* in the
        probe; *gestures*, if given, is called with the Playwright page once
        the probe is ready, and its return joins the report. *session*: that
        the script builds one."""
        from playwright.sync_api import sync_playwright

        probe = self.temp.path(extension=".js")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write(PROBE_JS.replace("__ACTION__", action))
        server = ptk.PreviewServer(viewer=True, title="articulation-test", port=0)
        server.start()
        server.add_script("articulated_rig")
        server.add_script("probe", probe)
        console = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    channel="msedge",
                    headless=True,
                    args=["--enable-unsafe-swiftshader"],
                )
                page = browser.new_page(viewport={"width": 1100, "height": 800})
                page.on(
                    "console",
                    lambda m: (
                        console.append(f"[{m.type}] {m.text}")
                        if m.type in ("error", "warning", "info")
                        else None
                    ),
                )
                page.on("pageerror", lambda e: console.append(f"[pageerror] {e}"))
                page.goto(server.url, wait_until="domcontentloaded", timeout=120_000)
                page.wait_for_function("() => !!window.__probe", timeout=120_000)
                server.publish(glb or self.glb)
                page.wait_for_function(
                    "() => window.__probe.ready === true", timeout=180_000
                )
                found = page.evaluate(
                    "() => JSON.parse(JSON.stringify(window.__probe, (k, v) => typeof v === 'function' ? undefined : v))"
                )
                if gestures:
                    self._server = server
                    found["gestures"] = gestures(page)
                browser.close()
        finally:
            server.stop()
        found["console"] = console
        self.assertEqual(found["errors"], [], found)
        self.assertEqual(
            bool(found.get("hasSession")), session, f"the script's session: {found}"
        )
        return found

    def _warned(self, found):
        return "\n".join(line for line in found["console"] if "articulated_rig" in line)

    def test_the_js_model_matches_every_conformance_case(self):
        """Every case, the anchored rigs of articulation v2 included (an
        anchored arm, a flat model, a branching tree under anchors); the port
        measures no unit (``scaleOf`` retired: the record is in metres)."""
        doc = ptk.Conformance.cases("articulation", seed=2, per_rig=3)
        found = self._load(CONFORMANCE_JS.replace("__DOC__", json.dumps(doc)))
        self.assertEqual(found["count"], len(doc["cases"]))
        self.assertTrue({"anchored", "flat", "anchored_tree"} <= set(found["rigs"]))
        for quantity, tolerance in doc["tolerance"].items():
            self.assertLessEqual(
                found["worst"][quantity], tolerance, (quantity, found["worst"])
            )
        self.assertEqual(found["scaleOf"], "undefined")

    def test_a_grab_through_the_page_solves_what_python_solves(self):
        found = self._load(API_GRAB_JS.replace("__PART__", "head_geo"))
        self.assertEqual(found["picked"], "head_geo")
        self.assertTrue(found["pressed"])
        model = ArticulationModel(self.record)
        # The mouse keeps the held link's rotation as taken.
        want = model.solve(
            found["start"],
            found["joint"],
            found["local"],
            found["targetRig"],
            found["rotation"],
        )
        for got, exp in zip(found["after"], want):
            self.assertAlmostEqual(got, exp, delta=1e-6)
        self.assertLess(math.dist(found["held"], found["target"]), 1e-3)
        # Let go and the pose holds, frame after frame.
        self.assertTrue(found["posed"])
        self.assertEqual(found["kept"], found["after"])
        # The record is in metres, the file's own: no unit is measured.
        self.assertFalse(found["hasUnit"])
        for distance, turn in found["chain"]:
            self.assertLess(distance, 1e-7)
            self.assertLess(turn, 1e-7)

    def test_a_headset_grip_holds_the_part_and_carries_it(self):
        found = self._load(XR_GRAB_JS)
        self.assertEqual(found["xrHolding"], ["right"])
        self.assertLess(math.dist(found["xrHeld"], found["xrTarget"]), 2e-3)
        self.assertEqual(found["xrReleased"], [])

    def test_a_mouse_drag_moves_the_rig_and_not_the_camera(self):
        def drag(page):
            x, y = page.evaluate("() => window.__probe.screen")
            page.mouse.move(x, y)
            page.mouse.down()
            for k in range(1, 9):
                page.mouse.move(x + 6 * k, y - 4 * k)
            page.wait_for_timeout(100)
            during = page.evaluate("() => window.__probe.snapshot()")
            page.mouse.up()
            page.wait_for_timeout(200)
            return {
                "during": during,
                "after": page.evaluate("() => window.__probe.snapshot()"),
            }

        found = self._load(SCREEN_JS, gestures=drag)
        before, during, after = (
            found["before"],
            found["gestures"]["during"],
            found["gestures"]["after"],
        )
        self.assertEqual(during["holding"], ["mouse"])
        self.assertEqual(after["holding"], [])
        self.assertNotEqual(after["state"], before["state"])
        self.assertTrue(after["posed"])
        for a, b in zip(after["camera"], before["camera"]):
            self.assertAlmostEqual(a, b, places=6)  # the orbit never saw the press

    def test_the_wheel_pushes_the_held_point_along_the_view(self):
        """While a mouse holds a part, the wheel moves the drag plane along
        the view -- the held point goes deeper or nearer -- and the camera
        does not zoom."""

        def wheel(page):
            x, y = page.evaluate("() => window.__probe.screen")
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + 4, y - 3)
            page.wait_for_timeout(100)
            held = page.evaluate("() => window.__probe.heldPoint()")
            camera = page.evaluate("() => window.__probe.snapshot().camera")
            for _ in range(3):
                page.mouse.wheel(0, 120)
                page.wait_for_timeout(50)
            page.mouse.move(x + 4, y - 3)
            page.wait_for_timeout(100)
            pushed = page.evaluate("() => window.__probe.heldPoint()")
            after = page.evaluate("() => window.__probe.snapshot()")
            page.mouse.up()
            return {"held": held, "pushed": pushed, "camera": camera, "after": after}

        found = self._load(SCREEN_JS, gestures=wheel)
        g = found["gestures"]
        self.assertEqual(g["after"]["holding"], ["mouse"])
        self.assertGreater(math.dist(g["held"], g["pushed"]), 1e-4, "the point moved in depth")
        for a, b in zip(g["after"]["camera"], g["camera"]):
            self.assertAlmostEqual(a, b, places=6)

    def test_a_slider_poses_its_channel(self):
        found = self._load(SLIDER_JS)
        self.assertTrue(found["panelShown"])
        self.assertEqual(
            found["sliderCount"], len(ArticulationModel(self.record).channels)
        )
        self.assertAlmostEqual(found["sliderState"][0], 37.0, places=6)
        self.assertTrue(found["sliderPosed"])
        # A slide reads in centimetres, off the record's metres.
        (extend,) = [
            t for label, t in zip(found["sliderLabels"], found["sliderTexts"])
            if label == "extend tx"
        ]
        self.assertEqual(extend, "0.0 cm")
        # Off 0 too: the slide's metres read x 100 (pinned 2026-11-08: a
        # text that dropped the x 100 read 0.0 cm at rest as well).
        self.assertAlmostEqual(found["slideValue"], 0.021, delta=1e-4)
        self.assertEqual(found["slideText"], f"{found['slideValue'] * 100:.1f} cm")
        self.assertEqual(found["slideText"], "2.1 cm")

    def test_the_panel_names_each_joint_and_part_by_its_node(self):
        """The page names each joint and grabbed part by its node -- the name
        the record joined it by: a slider reads ``<joint> <channel>`` for the
        joint the DCC named (``arm_elbow_jnt`` read as ``elbow``), never
        ``joint1``. Added: 2026-10-20"""
        found = self._load(SLIDER_JS)
        conformance = ArticulationConformance.rigs()["arm"]
        self.assertEqual(
            found["sliderLabels"],
            [
                f"{joint['name']} {channel['channel']}"
                for joint in conformance["joints"]
                for channel in joint["channels"]
            ],
        )
        self.assertEqual(found["grabNames"], PARTS)

    def test_the_panels_buttons_reset_return_and_show_the_axes(self):
        """Reset poses the rest pose and holds it; Return to animation hands
        the rig back (the mode row says which); Joint axes toggles a helper
        on every joint's node."""
        found = self._load(PANEL_JS)
        self.assertEqual(
            set(found["buttons"]) & {"Reset", "Return to animation", "Joint axes"},
            {"Reset", "Return to animation", "Joint axes"},
        )
        self.assertEqual(found["posedRows"].get("mode"), "posed (holding)")
        self.assertEqual(found["posedRows"].get("grab"), ", ".join(PARTS))
        self.assertEqual(found["resetState"], [0.0] * len(found["resetState"]))
        self.assertTrue(found["resetPosed"])
        for distance, turn in found["resetChain"]:
            self.assertLess(distance, 1e-7)
            self.assertLess(turn, 1e-7)
        self.assertFalse(found["returnedPosed"])
        self.assertEqual(found["returnedRows"].get("mode"), "animation")
        n = len(self.record["joints"])
        self.assertEqual(found["axesBefore"], [False] * n)
        self.assertEqual(found["axesOn"], [True] * n)
        self.assertEqual(found["axesParents"], [True] * n)
        reach = sum(math.dist(j["t"], (0.0, 0.0, 0.0)) for j in self.record["joints"])
        for at, size in found["axesAt"]:
            self.assertEqual(at, [0.0, 0.0, 0.0], "an identity anchor: the node's own origin")
            self.assertAlmostEqual(size, 0.12 * reach, delta=1e-6)
        self.assertEqual(found["axesOff"], [False] * n)

    def test_a_posed_rig_holds_over_its_playing_clip_until_play_is_pressed(self):
        """After the mixer: a rig a slider posed holds that pose while its
        clip plays on, frame after frame; pressing play again hands it back
        to the clip."""
        model = ArticulationModel(self.record)
        quats = []
        for degrees in (0.0, 60.0, 0.0):
            state = model.rest_state()
            state[1] = degrees  # the base's rz
            quats += list(model.local(state, 0)[1])
        nodes = _arm_nodes(self.record)
        glb = self._built(
            nodes,
            animations={"name": "clip", "data": [([0.0, 0.5, 1.0], quats)], "nodes": [1]},
        )
        found = self._load(CLIP_JS, glb=glb)
        self.assertTrue(found["clipMoves"], "the clip plays the base")
        self.assertTrue(found["playing"])
        self.assertTrue(found["posedWhilePlaying"])
        self.assertTrue(found["heldOverClip"], "the clip overwrote the posed rig")
        self.assertTrue(found["handedBack"])
        self.assertTrue(found["clipMovesAgain"])

    def test_joints_bind_by_name_through_a_namespace(self):
        """A joint is bound by its name, not an index: a node an import kept
        a namespace on (``NS:arm_base_jnt``) answers the record's
        ``arm_base_jnt`` by the name after its last ':'. Added: 2026-11-08"""
        glb = self._built(_arm_nodes(self.record, rename=lambda name: f"NS:{name}"))
        found = self._load(BUILT_JS, glb=glb)
        (rig,) = found["rigs"]
        self.assertEqual(rig["name"], "arm")
        self.assertEqual(len(rig["joints"]), len(self.record["joints"]))
        self.assertEqual(rig["grab"], PARTS)

    def test_every_rigs_joints_bind_as_one_assignment(self):
        """Every rig's joints are ONE assignment over the file's nodes: a
        second rig naming ``B:arm_base_jnt`` and the rest, none of them in the
        file, never takes by its leaf the nodes the first rig's exact names
        took -- two rigs never bind one node, and the second is a rig not in
        this file. Pinned 2026-11-08 (each rig its own assignment bound both
        rigs to the one arm). Added: 2026-11-08"""
        second = _rig_record("copy")
        for a, b in zip(second["joints"], self.record["joints"]):
            a["name"] = f"B:{b['name']}"
        glb = self._built(
            _arm_nodes(self.record, payload=_payload(self.record, second))
        )
        found = self._load(BUILT_JS, glb=glb)
        self.assertEqual([r["name"] for r in found["rigs"]], ["arm"], self._warned(found))
        self.assertEqual(
            found["rigs"][0]["joints"], [j["name"] for j in self.record["joints"]]
        )
        self.assertIn(
            "1 rig(s) the record covers are not in this file", self._warned(found)
        )

    def test_an_ambiguous_joint_leaves_its_rig_unposed_and_said(self):
        nodes = _arm_nodes(self.record)
        nodes.insert(len(nodes) - 1, {"name": self.record["joints"][2]["name"]})
        found = self._load(BUILT_JS, glb=self._built(nodes), session=False)
        warned = self._warned(found)
        self.assertIn("arm_collar_jnt names several nodes", warned)
        self.assertIn("ambiguous, the rig is not posed", warned)

    def test_a_rig_only_partly_in_the_file_is_said_and_not_posed(self):
        nodes = _arm_nodes(self.record)
        nodes[3]["name"] = "renamed"  # the collar joint's node
        found = self._load(BUILT_JS, glb=self._built(nodes), session=False)
        self.assertIn("arm: joints not in the scene: arm_collar_jnt", self._warned(found))

    def test_the_next_load_tears_the_last_rig_down(self):
        """A second push of the file leaves one rig on the page: the last
        load's grabs unregistered, its joint axes and its Rig window section
        gone."""

        def again(page):
            first = page.evaluate("() => window.__probe.counts()")
            page.evaluate("() => { window.__probe.ready = false; }")
            self._server.publish(self.glb)
            page.wait_for_function("() => window.__probe.ready === true", timeout=180_000)
            return {"first": first, "second": page.evaluate("() => window.__probe.counts()")}

        found = self._load(BUILT_JS, gestures=again)
        first, second = found["gestures"]["first"], found["gestures"]["second"]
        self.assertEqual(second["loads"], first["loads"] + 1)
        self.assertEqual(first["grabs"], len(PARTS))
        self.assertEqual(second, dict(first, loads=second["loads"]))
        self.assertEqual((second["axes"], second["sections"]), (len(self.record["joints"]), 1))

    def test_a_rig_not_in_the_file_is_passed_over(self):
        other = _rig_record("other")
        glb = self._built(_arm_nodes(self.record, payload=_payload(self.record, other)))
        found = self._load(BUILT_JS, glb=glb)
        self.assertEqual([r["name"] for r in found["rigs"]], ["arm"])
        self.assertIn("1 rig(s) the record covers are not in this file", self._warned(found))

    def test_a_grabbed_part_binds_under_its_own_joint(self):
        """A part name the assembly repeats binds the one under its joint's
        node; one found under no joint's node is said and left ungrabbable."""
        record = json.loads(json.dumps(self.record))
        record["grab"].append({"node": "ghost_geo", "joint": 1})
        nodes = _arm_nodes(self.record, payload=_payload(record), extra_parts=[("head_geo", 0)])
        found = self._load(SLIDER_JS, glb=self._built(nodes))
        self.assertEqual(found["grabNames"], PARTS)
        self.assertEqual(
            found["grabParents"], [j["name"] for j in self.record["joints"]]
        )
        self.assertIn("grab part ghost_geo is not found under joint arm_elbow_jnt", self._warned(found))

    def test_a_rig_whose_roots_hang_under_different_nodes_poses_each_in_its_space(self):
        """Each root's chain is posed and solved in the space its own root
        hangs in: a rig whose second root hangs under another, differently
        placed node grabs and lands there (HEAD refused the rig). Added:
        2026-11-08"""
        record = json.loads(json.dumps(self.record))
        record["joints"][1]["parent"] = None  # the elbow: a second root
        nodes = _arm_nodes(record)
        nodes[0]["children"].remove(2)
        nodes.insert(
            len(nodes) - 1,
            {"name": "other_RIG", "translation": [0.3, 0.1, 0.0], "scale": [1.5] * 3, "children": [2]},
        )
        glb = self._built(nodes)
        built = self._load(BUILT_JS, glb=glb)
        (rig,) = built["rigs"]
        self.assertEqual(rig["spaces"], ["arm_RIG"] + ["other_RIG"] * 4)
        found = self._load(API_GRAB_JS.replace("__PART__", "head_geo"), glb=glb)
        self.assertTrue(found["pressed"])
        want = ArticulationModel(record).solve(
            found["start"], found["joint"], found["local"], found["targetRig"], found["rotation"]
        )
        for got, exp in zip(found["after"], want):
            self.assertAlmostEqual(got, exp, delta=1e-6)
        self.assertLess(math.dist(found["held"], found["target"]), 1e-3)
        for distance, turn in found["chain"]:
            self.assertLess(distance, 1e-7)
            self.assertLess(turn, 1e-7)

    def test_a_hands_turn_is_seen_from_its_joints_own_space(self):
        """A hand's turn reaches the held link seen from the space ITS chain
        is solved in: on a rig whose roots hang under differently turned
        nodes, a head held through the second root's chain and dragged with
        a world turn solves what Python solves with that turn read in that
        root's space. Pinned 2026-11-08 (a turn read in the first root's
        space passed while both spaces were unturned). Added: 2026-11-08"""
        record = json.loads(json.dumps(self.record))
        record["joints"][1]["parent"] = None  # the elbow: a second root
        nodes = _arm_nodes(record)
        nodes[0]["children"].remove(2)
        nodes[0]["rotation"] = [0.0, math.sin(0.2), 0.0, math.cos(0.2)]
        half = math.radians(35.0)
        nodes.insert(
            len(nodes) - 1,
            {
                "name": "other_RIG",
                "translation": [0.3, 0.1, 0.0],
                "rotation": [math.sin(half), 0.0, 0.0, math.cos(half)],
                "scale": [1.5] * 3,
                "children": [2],
            },
        )
        found = self._load(TURN_GRAB_JS, glb=self._built(nodes))
        self.assertTrue(found["pressed"])
        s, f = found["spaceTurn"], found["firstSpaceTurn"]
        self.assertGreater(
            min(math.dist(s, f), math.dist(s, [-v for v in f])),
            0.1,
            "the two roots' spaces are turned apart",
        )
        q = ptk.Quaternion
        seen = q.multiply(q.multiply(q.conjugate(s), found["turn"]), s)
        wanted = q.multiply(seen, found["rotation"])
        want = ArticulationModel(record).solve(
            found["start"], found["joint"], found["local"], found["targetRig"], wanted
        )
        for got, exp in zip(found["after"], want):
            self.assertAlmostEqual(got, exp, delta=1e-6)

    def test_a_flat_models_parts_keep_their_own_scale(self):
        """A flat model's parts posed from their joint chain keep their own
        scale, and a scaled part's scale never reaches the parts posed from
        it (a joint's frame is its node's place and turn). Pinned
        2026-11-08: no fixture scaled a joint's node. Added: 2026-11-08"""
        record = _rig_record("flat", base="flat", parts=[])
        record["grab"] = [
            {"node": j["name"], "joint": i} for i, j in enumerate(record["joints"])
        ]
        nodes = _flat_nodes(record)
        scaled = {"flat_upper_jnt": 1.5, "flat_lower_jnt": 0.75}
        for node in nodes:
            if node.get("name") in scaled:
                node["scale"] = [scaled[node["name"]]] * 3
        values = self._values(record, 11)
        found = self._load(
            CHAIN_JS.replace("__VALUES__", json.dumps(values)), glb=self._built(nodes)
        )
        self.assertEqual(found["nested"], [True, False, False, False])
        self.assertTrue([v for v in found["state"] if abs(v) > 1e-3], "posed")
        for joint, scale in zip(record["joints"], found["scales"]):
            for v in scale:
                self.assertAlmostEqual(
                    v, scaled.get(joint["name"], 1.0), places=6, msg=joint["name"]
                )
        for distance, turn in found["chain"]:
            self.assertLess(distance, 1e-7)
            self.assertLess(turn, 1e-7)

    def _values(self, record, seed):
        """Random channel values inside the limits, one per slider."""
        import random

        rng = random.Random(seed)
        model = ArticulationModel(record)
        out = []
        for slot, (_joint, channel) in enumerate(model.channels):
            lo, hi = model.limits(slot)
            span = 120.0 if channel[0] == "r" else 0.02
            out.append(rng.uniform(span * -0.8 if lo is None else lo, span * 0.8 if hi is None else hi))
        return out

    def test_an_anchored_rig_poses_about_its_anchors(self):
        """Articulation v2: a nested rig whose hinges, slides and ball turn
        about anchors off their pivots poses as the model does -- each node's
        local the model's, each world pose its chain's -- and a grab solves
        what Python solves. Added: 2026-11-08"""
        record = _rig_record("anchored", base="anchored")
        glb = self._built(_arm_nodes(record))
        values = self._values(record, 5)
        found = self._load(CHAIN_JS.replace("__VALUES__", json.dumps(values)), glb=glb)
        self.assertEqual(found["nested"], [True] * len(record["joints"]))
        model = ArticulationModel(record)
        for (t, q), (want_t, want_q) in zip(found["locals"], model.pose(found["state"])):
            self.assertLess(math.dist(t, want_t), 1e-7)
            self.assertLess(min(math.dist(q, want_q), math.dist(q, [-v for v in want_q])), 1e-7)
        for distance, turn in found["chain"]:
            self.assertLess(distance, 1e-7)
            self.assertLess(turn, 1e-7)
        grab = self._load(API_GRAB_JS.replace("__PART__", "head_geo"), glb=glb)
        want = model.solve(grab["start"], grab["joint"], grab["local"], grab["targetRig"], grab["rotation"])
        for got, exp in zip(grab["after"], want):
            self.assertAlmostEqual(got, exp, delta=1e-6)

    def test_a_flat_model_is_posed_from_its_joint_chain(self):
        """A flat model -- every part under the root, chained by the record's
        parents alone, its hinges in its anchors -- poses each part where its
        joint chain puts it, not its scene parent; its parts (each its
        joint's own node) are grabbed and land. Added: 2026-11-08"""
        record = _rig_record("flat", base="flat", parts=[])
        record["grab"] = [{"node": j["name"], "joint": i} for i, j in enumerate(record["joints"])]
        glb = self._built(_flat_nodes(record))
        values = self._values(record, 9)
        found = self._load(CHAIN_JS.replace("__VALUES__", json.dumps(values)), glb=glb)
        self.assertEqual(found["nested"], [True, False, False, False])
        self.assertEqual(found["spaces"], ["flat_RIG"] * 4)
        moved = [s for s in found["state"] if abs(s) > 1e-3]
        self.assertTrue(moved, "the sliders posed it")
        for distance, turn in found["chain"]:
            self.assertLess(distance, 1e-7)
            self.assertLess(turn, 1e-7)
        # Joint axes on each anchor, its axes the anchor's, sized off the
        # rig's bones pivot to pivot: a flat model's rest translates are all
        # 0, which, summed, sized every helper 0.12 m whatever the rig (and
        # an unlimited slide +-1 m). Pinned 2026-11-08.
        model = ArticulationModel(record)
        reach = sum(
            math.dist(model._bone(i), (0.0, 0.0, 0.0)) for i in range(len(model.joints))
        )
        self.assertGreater(reach, 0.05)
        for joint, (at, turn, size) in zip(record["joints"], found["axes"]):
            self.assertLess(math.dist(at, joint["anchor"]["t"]), 1e-7)
            want = joint["anchor"]["q"]
            self.assertLess(
                min(math.dist(turn, want), math.dist(turn, [-v for v in want])), 1e-7
            )
            self.assertAlmostEqual(size, 0.12 * reach, delta=1e-6)
        grab = self._load(API_GRAB_JS.replace("__PART__", "flat_slider_jnt"), glb=glb)
        self.assertTrue(grab["pressed"])
        want = ArticulationModel(record).solve(
            grab["start"], grab["joint"], grab["local"], grab["targetRig"], grab["rotation"]
        )
        for got, exp in zip(grab["after"], want):
            self.assertAlmostEqual(got, exp, delta=1e-6)
        for distance, turn in grab["chain"]:
            self.assertLess(distance, 1e-7)
            self.assertLess(turn, 1e-7)


#: A production export: every rig in the file built, the named part grabbed
#: through the API, and the clip left playing underneath the grab.
PRODUCTION_JS = """
{
  report.rigs = session.rigs.map((rig) => ({
    name: rig.record.name,
    joints: rig.nodes.length,
    grab: rig.grab.map((g) => g.name),
  }));
  const rig = session.rigs[0];
  const entry = rig.grab.find((g) => g.name === '__PART__') || rig.grab[rig.grab.length - 1];
  const box = new THREE.Box3().setFromObject(entry.node);
  const centre = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  report.hasMixer = !!viewer.mixer;
  viewer.setPlaying(true);
  const origin = centre.clone().add(new THREE.Vector3(0, 0, 5 * size + 1));
  const direction = centre.clone().sub(origin).normalize();
  const hit = viewer.grab.pick(origin, direction);
  report.picked = hit ? hit.object.name : null;
  report.pressed = viewer.grab.press('mouse', origin, direction);
  const hold = rig.holds.get('mouse');
  const target = hit.point.clone().add(new THREE.Vector3(-0.2, 0.15, 0.1).multiplyScalar(size));
  for (let k = 1; k <= 20; k += 1) {
    viewer.grab.drag('mouse', hit.point.clone().lerp(target, k / 20), null);
    await frames(1);
  }
  report.miss = new THREE.Vector3(...report.worldOf(rig, hold.joint, hold.local)).distanceTo(target);
  report.size = size;
  viewer.grab.release('mouse');
  const posed = rig.nodes.map((n) => n.quaternion.toArray());
  await frames(10);  // the clip plays on; the released pose must hold
  report.held = rig.nodes.every((n, i) => n.quaternion.toArray().every((v, j) => Math.abs(v - posed[i][j]) < 1e-9));
}
"""


@unittest.skipUnless(
    os.environ.get("ARTICULATED_RIG_LIVE_GLB") and browser_runtime_available(),
    "set ARTICULATED_RIG_LIVE_GLB to a GLB mayatk's articulated_rig_live_check.py wrote",
)
class TestArticulatedRigProduction(TestArticulatedRigLive):
    """A real export (the magnifier, through the Maya pipeline) in the real
    page: its rig binds, its head is grabbed and lands, and the released pose
    holds while its clip plays on."""

    @classmethod
    def setUpClass(cls):
        cls.temp = ptk.TempArtifacts("articulation_web_production", policy="scoped")
        cls.glb = os.environ["ARTICULATED_RIG_LIVE_GLB"]

    def test_the_production_rig_binds_and_its_head_is_grabbed(self):
        part = os.environ.get("ARTICULATED_RIG_LIVE_PART", "MAG_GLASS")
        found = self._load(PRODUCTION_JS.replace("__PART__", part))
        rig = found["rigs"][0]
        print(f"\nproduction rig: {rig}")
        self.assertGreaterEqual(rig["joints"], 1)
        self.assertIn(part, rig["grab"])
        self.assertEqual(found["picked"], part)
        self.assertTrue(found["pressed"])
        self.assertLess(found["miss"], 0.01 * found["size"], found)
        self.assertTrue(found["hasMixer"])
        self.assertTrue(found["held"], "the playing clip overwrote the released pose")

    # The fixture tests of the parent class are not re-run against the
    # production file.
    test_the_js_model_matches_every_conformance_case = None
    test_a_grab_through_the_page_solves_what_python_solves = None
    test_a_headset_grip_holds_the_part_and_carries_it = None
    test_a_mouse_drag_moves_the_rig_and_not_the_camera = None
    test_the_wheel_pushes_the_held_point_along_the_view = None
    test_a_slider_poses_its_channel = None
    test_the_panel_names_each_joint_and_part_by_its_node = None
    test_the_panels_buttons_reset_return_and_show_the_axes = None
    test_a_posed_rig_holds_over_its_playing_clip_until_play_is_pressed = None
    test_joints_bind_by_name_through_a_namespace = None
    test_an_ambiguous_joint_leaves_its_rig_unposed_and_said = None
    test_a_rig_not_in_the_file_is_passed_over = None
    test_a_rig_only_partly_in_the_file_is_said_and_not_posed = None
    test_the_next_load_tears_the_last_rig_down = None
    test_a_grabbed_part_binds_under_its_own_joint = None
    test_a_rig_whose_roots_hang_under_different_nodes_poses_each_in_its_space = None
    test_an_anchored_rig_poses_about_its_anchors = None
    test_a_flat_model_is_posed_from_its_joint_chain = None


if __name__ == "__main__":
    unittest.main()
