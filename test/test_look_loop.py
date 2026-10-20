# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.core_utils.engines.lighting.look_match.loop: a round of look matching in one call.

The bake worker and the capture service are stand-ins: what is under test is
the round -- what it bakes, in what order it asks, what it writes.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from pythontk.core_utils.engines.lighting.light_layers import LightLayers, LightMix
from pythontk.core_utils.engines.lighting.light_predictor import LightPredictor
from pythontk.core_utils.engines.lighting.look_match.target import LookRegion, LookTarget
from pythontk.core_utils.engines.lighting.look_match.loop import LookLoop

KEY = {
    "lights": [
        {"name": "L_KEY", "type": "spot", "pos": [0, 200, 100], "intensity": 2.0}
    ]
}
FILL = {
    "lights": [
        {"name": "L_FILL", "type": "area", "pos": [0, 300, 0], "rot": [-90, 0, 0]}
    ]
}


class TestLookLoop(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="look_loop_")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        h, w = 24, 32
        xx = np.mgrid[0:h, 0:w][1] / w
        self.store = LightLayers(os.path.join(self.dir, "layers"))
        key = np.repeat(np.exp(-((xx - 0.3) ** 2) / 0.03)[..., None], 3, 2)
        self.store.add("key", key.astype(np.float32), spec=KEY)
        self.store.add("fill", np.full((h, w, 3), 0.2, np.float32), spec=FILL)
        truth = LightMix({"key": {"weight": 1.0}, "fill": {"weight": 0.5}})
        ref = (LightPredictor(self.store).frame(truth) * 255 + 0.5).astype(np.uint8)
        Image.fromarray(ref).save(os.path.join(self.dir, "ref.png"))
        LookTarget(
            os.path.join(self.dir, "ref.png"),
            [
                LookRegion(
                    "left", "area", {"rect": [0.1, 0.2, 0.5, 0.8]}, measures=["mean"]
                ),
                LookRegion(
                    "right", "area", {"rect": [0.6, 0.2, 0.95, 0.8]}, measures=["mean"]
                ),
            ],
        ).save(os.path.join(self.dir, "target.json"))
        truth.save(os.path.join(self.dir, "mix.json"))
        self.session = os.path.join(self.dir, "session.json")
        with open(self.session, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "layers": "layers",
                    "target": "target.json",
                    "mix": "mix.json",
                    "loop": {"queue": "queue"},
                },
                f,
            )
        self.asked = []

    def invoke(self, op, timeout=None, **request):
        """The capture service's client, standing in: an exact frame is the
        prediction; a capture adds the baked roles to the store."""
        self.asked.append(op)
        if op == "exact":
            match = LookLoop(self.session, client=self).match
            frame = match.predict(LightMix.from_dict(request["mix"]), (32, 24))
            Image.fromarray((frame * 255 + 0.5).astype(np.uint8)).save(request["out"])
        if op == "capture":
            for name, payload in self.baked.items():
                self.store.add(
                    name,
                    np.full((24, 32, 3), 0.05, np.float32),
                    spec=payload,
                    replace=True,
                )
        return {}

    def rig(self, **roles):
        path = os.path.join(self.dir, "rig.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "roles": roles,
                    "fit": {"tolerance": 0.0, "regularize": 0.0},
                },
                f,
            )
        return path

    def test_a_roles_specular_scale_reaches_the_mix(self):
        loop = LookLoop(self.session, client=self)
        mix = loop.mix_of({"roles": {"key": dict(KEY, weight=1.0, specular=0.3)}})
        self.assertEqual(mix.specular("key"), 0.3)

    def test_only_missing_or_changed_roles_bake(self):
        moved = {"lights": [dict(FILL["lights"][0], pos=[0, 250, 0])]}
        loop = LookLoop(self.session, client=self)
        rig = LookLoop.load_rig(
            self.rig(key=dict(KEY, weight=1.0), fill=moved, rim=KEY)
        )
        self.assertEqual(sorted(loop.stale(rig)), ["fill", "rim"])

    def test_a_layer_baked_from_an_older_save_of_the_scene_bakes_again(self):
        """The worker bakes the scene as saved: a layer from a save before the
        last is stale even when its payload is not. One with no scene record (a
        store from before records) is taken as current, not re-baked wholesale."""
        from pythontk.core_utils.engines.lighting.light_layers import LightLayerScene

        scene = os.path.join(self.dir, "room.ma")
        with open(scene, "w", encoding="utf-8") as f:
            f.write("//Maya ASCII")
        with open(self.session, encoding="utf-8") as f:
            data = json.load(f)
        data["loop"]["worker"] = {
            "module": "mayatk.light_utils.light_layer_baker",
            "scene": "room.ma",
        }
        with open(self.session, "w", encoding="utf-8") as f:
            json.dump(data, f)
        now = LightLayerScene.of(scene)
        older = dict(now, mtime=now["mtime"] - 60)
        frame = np.full((24, 32, 3), 0.1, np.float32)
        self.store.add("key", frame, spec=KEY, replace=True, scene=now)
        self.store.add("fill", frame, spec=FILL, replace=True, scene=older)
        self.store.add("rim", frame, spec=KEY)  # no record
        loop = LookLoop(self.session, client=self)
        rig = LookLoop.load_rig(self.rig(key=KEY, fill=FILL, rim=KEY))
        self.assertEqual(loop.stale(rig), ["fill"])

    def _with_worker_scene(self):
        from pythontk.core_utils.engines.lighting.light_layers import LightLayerScene

        scene = os.path.join(self.dir, "room.ma")
        with open(scene, "w", encoding="utf-8") as f:
            f.write("//Maya ASCII")
        with open(self.session, encoding="utf-8") as f:
            data = json.load(f)
        data["loop"]["worker"] = {"module": "mayatk.light_utils.light_layer_baker", "scene": "room.ma"}
        with open(self.session, "w", encoding="utf-8") as f:
            json.dump(data, f)
        now = LightLayerScene.of(scene)
        return scene, now, dict(now, mtime=now["mtime"] - 60)

    def test_the_layers_an_older_save_baked_bake_again_from_their_own_payloads(self):
        """Update layers: what the scene's last save made stale, re-baked from
        the payloads the store recorded, then captured -- no rig file needed."""
        scene, now, older = self._with_worker_scene()
        frame = np.full((24, 32, 3), 0.1, np.float32)
        settings = {"resolution": 256}
        self.store.add("key", frame, spec=KEY, replace=True, scene=now, settings=settings)
        self.store.add("fill", frame, spec=FILL, replace=True, scene=older, settings=settings)
        self.store.add("rim", frame, spec=KEY, scene=older, settings={"resolution": 512})
        self.store.set_stamp(existing={"OFFICE": {"intensity": 0.0}})
        loop = LookLoop(self.session, client=self)
        self.assertEqual(loop.stale_layers(), ["fill", "rim"])
        jobs = []

        def submit(job):
            jobs.append(job)
            return str(len(jobs))

        loop.queue = mock.Mock(submit=submit, wait=lambda job_id, timeout=None, stale=None: {"layers": {}})
        self.baked = {"fill": FILL, "rim": KEY}
        folder = os.path.join(self.dir, "rebake")
        loop.rebake(["fill", "rim"], folder)
        # One job per baker setting the layers were baked with, each from its own spec.
        self.assertEqual(sorted(len(j["layers"]) for j in jobs), [1, 1])
        by_layer = {name: job for job in jobs for name in job["layers"]}
        self.assertEqual(by_layer["fill"]["layers"]["fill"], LookLoop.payload(FILL))
        self.assertEqual(by_layer["rim"]["settings"], {"resolution": 512})
        self.assertEqual(by_layer["fill"]["base"], {"existing": {"OFFICE": {"intensity": 0.0}}})
        self.assertEqual(self.asked, ["capture", "reload", "capture", "reload"])

    def test_a_loop_runs_over_an_open_session_and_waits_out_a_worker_starting(self):
        """A panel's own session (its mix, its layers' memory), and a worker it
        just launched -- a GUI Maya takes minutes to come up -- is waited for."""
        from pythontk.core_utils.engines.lighting.look_match._look_match import LookMatch

        match = LookMatch.open(self.session)
        loop = LookLoop(self.session, client=self, match=match)
        self.assertIs(loop.match, match)
        waited = {}
        loop.queue = mock.Mock(
            submit=lambda job: "1",
            wait=lambda job_id, timeout=None, stale=None: waited.update(stale=stale) or {},
        )
        loop.worker_stale = 900.0
        loop.rebake([], os.path.join(self.dir, "none"))  # nothing to bake: no job
        self.assertEqual(waited, {})
        self.store.add("fill", np.full((24, 32, 3), 0.1, np.float32), spec=FILL, replace=True)
        loop.match = LookMatch.open(self.session)
        self.baked = {"fill": FILL}  # what the capture stand-in adds back
        loop.rebake(["fill"], os.path.join(self.dir, "one"))
        self.assertEqual(waited["stale"], 900.0)

    def test_a_rebake_the_worker_refused_says_so(self):
        _scene, _now, older = self._with_worker_scene()
        self.store.add("fill", np.full((24, 32, 3), 0.1, np.float32), spec=FILL, replace=True, scene=older)
        loop = LookLoop(self.session, client=self)
        loop.queue = mock.Mock(
            submit=lambda job: "1",
            wait=lambda job_id, timeout=None, stale=None: {"layers": {"fill": {"refused": "no light"}}},
        )
        with self.assertRaises(RuntimeError) as ctx:
            loop.rebake(["fill"], os.path.join(self.dir, "rebake"))
        self.assertIn("no light", str(ctx.exception))

    def test_a_round_with_nothing_to_bake_fits_and_judges(self):
        loop = LookLoop(self.session, client=self)
        loop.bake = lambda *a, **k: self.fail("nothing changed, nothing to bake")
        out = os.path.join(self.dir, "round")
        report = loop.iterate(
            self.rig(key=dict(KEY, weight=0.3, free=True), fill=dict(FILL, weight=0.5)),
            out,
        )
        self.assertEqual(report["baked"], [])
        self.assertEqual(self.asked, ["exact"])
        for name in (
            "mix.json",
            "rig.json",
            "sheet.png",
            "exact.png",
            "sheet_exact.png",
            "compare.png",
            "report.json",
        ):
            self.assertTrue(os.path.isfile(os.path.join(out, name)), name)
        with open(os.path.join(out, "rig.json"), encoding="utf-8") as f:
            fitted = json.load(f)
        self.assertAlmostEqual(fitted["roles"]["key"]["weight"], 1.0, delta=0.15)
        self.assertLess(report["fit"]["cost"][1], report["fit"]["cost"][0])
        self.assertTrue(report["exact"])

    def test_a_round_records_the_state_it_baked_under(self):
        """What the rig built from these layers must set again (``to_rig``)."""
        loop = LookLoop(self.session, client=self)

        def bake(rig, names, folder, timeout=None):
            self.baked = {n: LookLoop.payload(rig["roles"][n]) for n in names}
            return {}

        loop.bake = bake
        path = os.path.join(self.dir, "rig_base.json")
        base = {
            "existing": {"OFFICE": {"intensity": 0.0}},
            "materials": {"MAT_ROOM_ai": {"emission": 0.0}},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"version": 1, "roles": {"rim": dict(KEY, weight=0.2)}, "base": base}, f
            )
        report = loop.iterate(
            path, os.path.join(self.dir, "round"), fit=False, exact=False
        )
        stamp = LookLoop(self.session, client=self).match.layers.stamp
        self.assertEqual(stamp["existing"], base["existing"])
        self.assertEqual(stamp["materials"], base["materials"])
        self.assertNotIn("base_changed", report)
        # A later round baked under another state: the store now mixes two.
        base["materials"] = {"MAT_ROOM_ai": {"emission": 1.0}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "roles": {
                        "rim": dict(
                            KEY,
                            weight=0.3,
                            lights=[dict(KEY["lights"][0], pos=[1, 2, 3])],
                        )
                    },
                    "base": base,
                },
                f,
            )
        loop = LookLoop(self.session, client=self)
        loop.bake = bake
        report = loop.iterate(
            path, os.path.join(self.dir, "round2"), fit=False, exact=False
        )
        self.assertEqual(report["base_changed"], ["materials"])

    def test_a_changed_role_bakes_then_captures_before_the_fit(self):
        loop = LookLoop(self.session, client=self)
        baked_names = []

        def bake(rig, names, folder, timeout=None):
            baked_names.extend(names)
            self.baked = {n: LookLoop.payload(rig["roles"][n]) for n in names}
            return {}

        loop.bake = bake
        report = loop.iterate(
            self.rig(
                key=dict(KEY, weight=1.0),
                fill=dict(FILL, weight=0.5),
                rim=dict(KEY, weight=0.2),
            ),
            os.path.join(self.dir, "round"),
            exact=False,
        )
        self.assertEqual(baked_names, ["rim"])
        self.assertEqual(report["baked"], ["rim"])
        self.assertEqual(self.asked, ["capture", "reload"])
        self.assertIn("rim", loop.match.layers)  # the session reopened on the new store

    def test_a_bake_with_no_worker_says_where_to_name_one(self):
        with open(self.session, encoding="utf-8") as f:
            data = json.load(f)
        data.pop("loop")
        with open(self.session, "w", encoding="utf-8") as f:
            json.dump(data, f)
        loop = LookLoop(self.session, client=self)
        with self.assertRaises(RuntimeError) as ctx:
            loop.bake({"roles": {"rim": KEY}}, ["rim"], self.dir)
        self.assertIn("queue", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(exit=False)
