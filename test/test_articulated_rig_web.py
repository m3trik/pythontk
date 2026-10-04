# !/usr/bin/python
# coding=utf-8
"""The articulated rigs' web route: ``MeshConvert.apply_glb_articulation``, the
packaged ``articulated_rig`` viewer script and the page's ``viewer.grab``.

Two halves, in one module because they share the fixtures:

* Pure Python -- the pass over hand-authored GLBs (a five-joint arm under a
  scaled rig group, its parts, the ``data_export`` carrier) and the server's
  auto-activation.
* The page -- headless Edge through Playwright, the real ``viewer.html`` and
  three.js, the real ``PreviewServer``: the script's ``ArticulationModel`` port
  against every ``ArticulationConformance`` case; a grab through
  ``viewer.grab`` against the Python solve of the same hold; a real mouse drag
  (which must move the rig and NOT orbit the camera); a headset grip on
  synthetic input; a slider. Skipped, never failed, without the runtime
  (``pip install playwright`` drives the installed Edge; nothing downloads).
"""

import json
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
#: The fixture's unit: the record is in centimetres, the glTF in metres.
UNIT = 0.01
#: The rig group's own scale -- the production prop sits under a scaled node.
GROUP_SCALE = 2.0
PARTS = ["pole_geo", "upper_geo", "rod_geo", "inner_geo", "head_geo"]


# ============================================================================
# Fixtures
# ============================================================================


def _rig_record(name="arm"):
    """The conformance arm as the DCC records it: joint names that are node
    names, and one grabbed part per joint."""
    rig = json.loads(json.dumps(ArticulationConformance.rigs()["arm"]))
    rig["name"] = name
    for joint in rig["joints"]:
        joint["name"] = f"{name}_{joint['name']}_jnt"
    rig["grab"] = [{"node": part, "joint": i} for i, part in enumerate(PARTS)]
    return rig


def _payload(*rigs, version=1):
    return {"version": version, "rigs": list(rigs)}


def _data_export_node(payload):
    """The carrier exactly as FBX2glTF transcribes a Maya ``data_export``."""
    return {
        "name": "data_export",
        "extras": {
            "fromFBX": {
                "userProperties": {
                    MeshConvert.ARTICULATION_KEY: {
                        "type": "eFbxString",
                        "value": json.dumps(payload),
                    }
                }
            }
        },
    }


def _arm_nodes(record, payload=None, extra_parts=()):
    """glTF nodes: the scaled rig group, its joints at rest (record units times
    UNIT), a cube part under each, and the carrier."""
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
                "name": joint["name"],
                "translation": [v * UNIT for v in joint["t"]],
                "rotation": list(joint["q"]),
                "children": [],
            }
        )
        joint_nodes.append(index)
        parent = 0 if joint["parent"] is None else joint_nodes[joint["parent"]]
        nodes[parent]["children"].append(index)
    for i, part in enumerate(PARTS):
        index = len(nodes)
        nodes.append(
            {
                "name": part,
                "mesh": 0,
                "translation": [0.02, 0.0, 0.0],
                "scale": [0.03, 0.012, 0.012],
            }
        )
        nodes[joint_nodes[i]]["children"].append(index)
    for name, joint in extra_parts:
        index = len(nodes)
        nodes.append({"name": name, "mesh": 0, "scale": [0.01] * 3})
        nodes[joint_nodes[joint]]["children"].append(index)
    nodes.append(_data_export_node(payload or _payload(record)))
    return nodes


def _write_glb(path, nodes, animations=None):
    """A minimal valid binary glTF: one unit cube every part node shares."""
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
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
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

    def test_the_manifest_binds_every_joint_and_grabbed_part_to_its_node(self):
        record = _rig_record()
        path = self._glb(_arm_nodes(record))
        manifest = MeshConvert.apply_glb_articulation(path)
        self.assertIsNotNone(manifest)
        self.assertEqual(_extras(path)[MeshConvert.ARTICULATION_WEB_KEY], manifest)
        rig = manifest["rigs"][0]
        self.assertEqual([j["node"] for j in rig["joints"]], [1, 2, 3, 4, 5])
        self.assertEqual([g["node"] for g in rig["grab"]], [6, 7, 8, 9, 10])
        self.assertEqual([g["name"] for g in rig["grab"]], PARTS)
        # the record rides through unchanged beside the indices
        self.assertEqual(rig["joints"][0]["rotate_order"], "zyx")
        self.assertEqual(manifest["metadata_version"], 1)

    def test_the_manifest_and_its_record_are_the_declared_shapes(self):
        """The applier's output IS ``ArticulationWeb`` and its input
        ``ArticulationRecord``: the declarations the runtimes' types are
        generated from are held to what the pipeline really writes."""
        record = _rig_record()
        self.assertEqual(ptk.ArticulationRecord.validate(_payload(record)).errors, [])
        manifest = MeshConvert.apply_glb_articulation(self._glb(_arm_nodes(record)))
        res = ptk.ArticulationWeb.validate(manifest)
        self.assertEqual((res.errors, res.warnings), ([], []))
        self.assertEqual(manifest["version"], ptk.SceneRecords.ARTICULATION.web.version)

    def test_a_grabbed_part_binds_under_its_own_joint_when_the_name_repeats(self):
        record = _rig_record()
        nodes = _arm_nodes(
            record, extra_parts=[("head_geo", 0)]
        )  # a second head_geo under joint 0
        path = self._glb(nodes)
        rig = MeshConvert.apply_glb_articulation(path)["rigs"][0]
        head = next(g for g in rig["grab"] if g["name"] == "head_geo")
        self.assertEqual(head["node"], 10)  # the one under the head joint

    def test_an_ambiguous_joint_leaves_its_rig_unwired_with_a_warning(self):
        record = _rig_record()
        nodes = _arm_nodes(record)
        nodes.insert(len(nodes) - 1, {"name": record["joints"][2]["name"]})
        path = self._glb(nodes)
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING") as logs:
            self.assertIsNone(MeshConvert.apply_glb_articulation(path))
        self.assertIn("ambiguous", "\n".join(logs.output))

    def test_a_rig_not_in_the_export_is_out_of_scope(self):
        record = _rig_record()
        other = _rig_record("other")
        path = self._glb(_arm_nodes(record, payload=_payload(record, other)))
        manifest = MeshConvert.apply_glb_articulation(path)
        self.assertEqual([r["name"] for r in manifest["rigs"]], ["arm"])

    def test_a_newer_schema_is_refused_not_guessed(self):
        record = _rig_record()
        path = self._glb(_arm_nodes(record, payload=_payload(record, version=99)))
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING"):
            self.assertIsNone(MeshConvert.apply_glb_articulation(path))

    def test_rerunning_is_idempotent_and_a_run_that_binds_nothing_clears(self):
        record = _rig_record()
        path = self._glb(_arm_nodes(record))
        first = MeshConvert.apply_glb_articulation(path)
        self.assertEqual(MeshConvert.apply_glb_articulation(path), first)
        # the same file without its channel: the stale manifest goes
        with MeshConvert.open_glb(str(path)) as edit:
            edit.gltf["nodes"] = [
                n for n in edit.gltf["nodes"] if n.get("name") != "data_export"
            ]
            edit.dirty = True
        self.assertIsNone(MeshConvert.apply_glb_articulation(path))
        self.assertNotIn(MeshConvert.ARTICULATION_WEB_KEY, _extras(path))

    def test_a_grab_part_missing_under_its_joint_is_dropped_with_a_warning(self):
        record = _rig_record()
        record["grab"].append({"node": "ghost_geo", "joint": 1})
        path = self._glb(_arm_nodes(record))
        with self.assertLogs(MESH_CONVERT_LOGGER, level="WARNING"):
            rig = MeshConvert.apply_glb_articulation(path)["rigs"][0]
        self.assertNotIn("ghost_geo", [g["name"] for g in rig["grab"]])

    def test_a_cyclic_hierarchy_elsewhere_does_not_hang_the_pass(self):
        """A malformed file whose ``children`` loop must not trap the walk
        that checks a grabbed part is under its joint."""
        record = _rig_record()
        nodes = _arm_nodes(record)
        a, b = len(nodes), len(nodes) + 1
        nodes += [
            {"name": "head_geo", "children": [b]},
            {"name": "loop", "children": [a]},
        ]
        path = self._glb(nodes)
        result = {}
        worker = threading.Thread(
            target=lambda: result.update(
                manifest=MeshConvert.apply_glb_articulation(path)
            ),
            daemon=True,
        )
        worker.start()
        worker.join(30)
        self.assertFalse(worker.is_alive(), "the pass looped on the cycle")
        head = next(
            g for g in result["manifest"]["rigs"][0]["grab"] if g["name"] == "head_geo"
        )
        self.assertEqual(head["node"], 10)  # the one under the head joint

    def test_a_curve_proxy_strip_keeps_the_manifest_on_its_nodes(self):
        """The conversion strips render-effect curve proxies AFTER this pass,
        renumbering every node behind one; the manifest's indices must follow
        or the runtime poses the wrong nodes."""
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
        MeshConvert.apply_glb_articulation(path)
        MeshConvert.strip_glb_curve_proxies(path)
        with MeshConvert.open_glb(str(path)) as edit:
            names = [n.get("name") for n in edit.gltf["nodes"]]
            rig = edit.gltf["extras"][MeshConvert.ARTICULATION_WEB_KEY]["rigs"][0]
        for joint in rig["joints"]:
            self.assertEqual(names[joint["node"]], joint["name"])
        for entry in rig["grab"]:
            self.assertEqual(names[entry["node"]], entry["name"])

    def test_a_manifest_index_to_a_stripped_node_reads_as_absent(self):
        """The renumber keeps a manifest's form (a dict, or its JSON text) and
        turns an index to a removed node into None, never a neighbour's."""
        manifest = {
            "rigs": [{"joints": [{"node": 0}, {"node": 1}], "grab": [{"node": 2}]}]
        }
        remap = {0: 0, 2: 1}  # node 1 was stripped
        fields = MeshConvert.ARTICULATION_WEB_NODE_FIELDS
        got = MeshConvert._renumber_manifest_nodes(
            json.loads(json.dumps(manifest)), fields, remap
        )
        self.assertEqual([j["node"] for j in got["rigs"][0]["joints"]], [0, None])
        self.assertEqual(got["rigs"][0]["grab"][0]["node"], 1)
        text = MeshConvert._renumber_manifest_nodes(json.dumps(manifest), fields, remap)
        self.assertEqual(json.loads(text), got)

    def test_the_handoff_names_the_manifest(self):
        envelope = MeshConvert.build_scene_sidecar({}, source={"application": "maya"})
        self.assertIn(
            f"extras.{MeshConvert.ARTICULATION_WEB_KEY}", envelope["handoff"]["reads"]
        )


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
        self.server.publish(path)  # the channel alone is not the manifest
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
  // The held point of a rig's hold, in WORLD space.
  report.worldOf = (rig, joint, local) => {
    const p = rig.model.point(rig.state, joint, local).map((v) => v * rig.unit);
    rig.space.updateWorldMatrix(true, false);
    return rig.space.localToWorld(new THREE.Vector3(...p)).toArray();
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
  const worst = { pose: 0, world: 0, read: 0, solve: 0, scale: 0 };
  const qd = (a, b) => Math.min(
    Math.hypot(...a.map((v, i) => v - b[i])), Math.hypot(...a.map((v, i) => v + b[i])));
  const vd = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
  for (const c of doc.cases) {
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
    worst.scale = Math.max(worst.scale, Math.abs(m.scaleOf(c.scale.locals) - c.scale.expect));
  }
  report.worst = worst;
  report.count = doc.cases.length;
}
"""

#: A grab through `viewer.grab`: aim at the head part's centre, take it, move
#: the point, and report what the Python side needs to solve the same hold.
API_GRAB_JS = """
{
  const rig = session.rigs[0];
  const entry = rig.grab.find((g) => g.name === 'head_geo');
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
  // the same target in the rig's space, record units: what Python solves to
  rig.space.updateWorldMatrix(true, false);
  report.targetRig = rig.space.worldToLocal(target.clone()).toArray().map((v) => v / rig.unit);
  report.after = rig.state.slice();
  report.held = report.worldOf(rig, hold.joint, hold.local);
  viewer.grab.release('mouse');
  await frames(3);
  report.kept = rig.state.slice();
  report.posed = rig.posed;
  report.unit = rig.unit;
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
  input.value = '37';
  input.dispatchEvent(new Event('input'));
  await frames(2);
  report.sliderState = rig.state.slice();
  report.sliderPosed = rig.posed;
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
        path = _write_glb(cls.temp.path(extension=".glb"), _arm_nodes(cls.record))
        MeshConvert.apply_glb_articulation(path)
        cls.glb = path

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _load(self, action, gestures=None):
        """Load the fixture with *action* in the probe; *gestures*, if given, is
        called with the Playwright page once the probe is ready, and its return
        joins the report."""
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
                        if m.type in ("error", "warning")
                        else None
                    ),
                )
                page.on("pageerror", lambda e: console.append(f"[pageerror] {e}"))
                page.goto(server.url, wait_until="domcontentloaded", timeout=120_000)
                page.wait_for_function("() => !!window.__probe", timeout=120_000)
                server.publish(self.glb)
                page.wait_for_function(
                    "() => window.__probe.ready === true", timeout=180_000
                )
                found = page.evaluate(
                    "() => JSON.parse(JSON.stringify(window.__probe, (k, v) => typeof v === 'function' ? undefined : v))"
                )
                if gestures:
                    found["gestures"] = gestures(page)
                browser.close()
        finally:
            server.stop()
        found["console"] = console
        self.assertEqual(found["errors"], [], found)
        self.assertTrue(
            found.get("hasSession"), f"the script built no session: {found}"
        )
        return found

    def test_the_js_model_matches_every_conformance_case(self):
        doc = ptk.Conformance.cases("articulation", seed=2, per_rig=3)
        found = self._load(CONFORMANCE_JS.replace("__DOC__", json.dumps(doc)))
        self.assertEqual(found["count"], len(doc["cases"]))
        for quantity, tolerance in doc["tolerance"].items():
            self.assertLessEqual(
                found["worst"][quantity], tolerance, (quantity, found["worst"])
            )

    def test_a_grab_through_the_page_solves_what_python_solves(self):
        found = self._load(API_GRAB_JS)
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
        # The unit was measured, not assumed: centimetre record, metre file.
        self.assertAlmostEqual(found["unit"], UNIT, places=9)

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

    def test_a_slider_poses_its_channel(self):
        found = self._load(SLIDER_JS)
        self.assertTrue(found["panelShown"])
        self.assertEqual(
            found["sliderCount"], len(ArticulationModel(self.record).channels)
        )
        self.assertAlmostEqual(found["sliderState"][0], 37.0, places=6)
        self.assertTrue(found["sliderPosed"])

    def test_a_root_joint_without_a_parent_key_binds(self):
        """The model reads a missing ``parent`` as a root; the script's binding
        must find its rig space the same way, not skip the rig."""
        record = _rig_record()
        written = json.loads(json.dumps(record))
        del written["joints"][0]["parent"]
        self.glb = _write_glb(
            self.temp.path(extension=".glb"),
            _arm_nodes(record, payload=_payload(written)),
        )
        MeshConvert.apply_glb_articulation(self.glb)
        found = self._load("report.joints = session.rigs[0].nodes.length;")
        self.assertEqual(found["joints"], len(record["joints"]))


#: A production export: every rig in the file built, the named part grabbed
#: through the API, and the clip left playing underneath the grab.
PRODUCTION_JS = """
{
  report.rigs = session.rigs.map((rig) => ({
    name: rig.record.name,
    joints: rig.nodes.length,
    grab: rig.grab.map((g) => g.name),
    unit: rig.unit,
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
    test_a_slider_poses_its_channel = None
    test_a_root_joint_without_a_parent_key_binds = None


if __name__ == "__main__":
    unittest.main()
