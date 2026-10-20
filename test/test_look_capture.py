# !/usr/bin/python
# coding=utf-8
"""Tests for pythontk.net_utils.preview.look_capture: the viewer, headless, for lighting work.

The browser-free half (the payload a relight sends, the frames a shot returns)
runs anywhere. The live half needs Playwright and a lightmapped template GLB:
set ``LOOK_CAPTURE_TEMPLATE`` to one (client scenes are never fixtures here).
"""

import base64
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from pythontk.net_utils.preview.look_capture import LookCapture

MANIFEST = {
    "materials": {
        "MAT_ROOM": {"map": "ROOM_Lightmap.png"},
        "MAT_DESK": {"map": "DESK_Lightmap.png"},
    },
    "probe": {
        "position": [0.0, 1.5, 0.0],
        "box": {"min": [-3, 0, -3], "max": [3, 3, 3]},
    },
}


class TestPayload(unittest.TestCase):
    def test_a_bake_maps_onto_the_template_by_image_name(self):
        bakes = {
            "ROOM_Lightmap.exr": np.full((4, 8, 3), 0.5, np.float32),
            "DESK_Lightmap.exr": np.full((2, 2, 3), 2.0, np.float32),
            "SCENE_Probe.exr": np.ones((2, 4, 3), np.float32),
            "STRAY_Lightmap.exr": np.zeros(
                (2, 2, 3), np.float32
            ),  # no material wears it
        }
        payload = LookCapture.payload(bakes, MANIFEST)
        self.assertEqual(
            payload["maps"], {"MAT_ROOM": "ROOM_Lightmap", "MAT_DESK": "DESK_Lightmap"}
        )
        self.assertEqual(sorted(payload["images"]), ["DESK_Lightmap", "ROOM_Lightmap"])
        room = payload["images"]["ROOM_Lightmap"]
        self.assertEqual((room["width"], room["height"]), (8, 4))
        self.assertEqual(payload["probe"]["position"], [0.0, 1.5, 0.0])
        self.assertEqual(payload["probe"]["box"], MANIFEST["probe"]["box"])

    def test_images_travel_as_rgba_half_floats(self):
        rgb = np.random.default_rng(0).random((3, 5, 3)).astype(np.float32) * 4
        image = LookCapture.payload({"ROOM_Lightmap.exr": rgb}, MANIFEST)["images"][
            "ROOM_Lightmap"
        ]
        half = np.frombuffer(base64.b64decode(image["b64"]), np.float16).reshape(
            3, 5, 4
        )
        np.testing.assert_allclose(half[..., :3], rgb, rtol=1e-3)
        np.testing.assert_array_equal(half[..., 3], 1.0)

    def test_a_probe_can_be_placed_elsewhere(self):
        bakes = {"SCENE_Probe.exr": np.ones((2, 4, 3), np.float32)}
        moved = LookCapture.payload(bakes, MANIFEST, probe={"position": [1, 2, 3]})
        self.assertEqual(moved["probe"]["position"], [1, 2, 3])
        self.assertEqual(moved["probe"]["box"], MANIFEST["probe"]["box"])  # kept
        unboxed = LookCapture.payload(bakes, MANIFEST, probe={"box": None})
        self.assertIsNone(unboxed["probe"]["box"])

    def test_the_manifest_is_read_where_the_viewer_reads_it(self):
        """REGRESSION (2026-10-19): the template's manifest was read from the
        root's extras alone, so a natively exported template -- its
        ``lightmap_web`` in the first scene's extras, as a JSON string --
        swapped no map and no probe, and so drew no highlights, while the
        viewer and the highlight pass both found it. One lookup now."""
        import struct

        from pythontk import TempArtifacts

        artifacts = TempArtifacts("look_capture_manifest", policy="scoped")
        self.addCleanup(artifacts.cleanup)
        gltf = {
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [], "extras": {"lightmap_web": json.dumps(MANIFEST)}}],
            "scene": 0,
        }
        payload = json.dumps(gltf).encode("utf-8")
        payload += b" " * (-len(payload) % 4)
        path = os.path.join(artifacts.dir_path(), "native.glb")
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, 20 + len(payload)))
            f.write(struct.pack("<I4s", len(payload), b"JSON") + payload)
        self.assertEqual(LookCapture(path).manifest, MANIFEST)

    def test_without_a_probe_bake_no_probe_is_sent(self):
        payload = LookCapture.payload(
            {"ROOM_Lightmap.exr": np.zeros((2, 2, 3), np.float32)}, MANIFEST
        )
        self.assertIsNone(payload["probe"])


class TestShots(unittest.TestCase):
    def test_a_png_shot_decodes_to_rgb(self):
        buf = io.BytesIO()
        Image.fromarray(np.full((3, 4, 3), 200, np.uint8)).save(buf, "PNG")
        url = "png:data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        frame = LookCapture.decode(url)
        self.assertEqual(frame.shape, (3, 4, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_a_linear_shot_decodes_to_float(self):
        values = np.arange(2 * 3 * 3, dtype=np.float16).reshape(2, 3, 3)
        text = f"f16:3x2:{base64.b64encode(values.tobytes()).decode()}"
        np.testing.assert_array_equal(
            LookCapture.decode(text), values.astype(np.float32)
        )

    def test_an_unknown_shot_is_refused(self):
        with self.assertRaises(ValueError):
            LookCapture.decode("gif:...")


class _FakeLayers:
    def names(self):
        return ["a", "b"]

    def bake_files(self, name):
        return {"x.exr": "x"} if name == "a" else {}


class _FakeMatch:
    layers = _FakeLayers()
    mix = None
    predictor = None


class TestRequests(unittest.TestCase):
    """What a session answers, without a browser (the frame-making calls stubbed)."""

    def setUp(self):
        self.capture = LookCapture("scene.glb")
        self.tmp = tempfile.mkdtemp(prefix="look_capture_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.captured = []
        self.capture.capture_layer = lambda layers, name, black=(): (
            self.captured.append((name, tuple(black)))
        )
        self.capture.idle = lambda asleep=True: None  # no page to sleep

    def test_status(self):
        self.assertEqual(
            self.capture.handle(_FakeMatch(), {}, {"op": "status"}),
            {"template": "scene.glb", "layers": 2},
        )

    def test_capture_defaults_to_every_layer_with_a_bake(self):
        reply = self.capture.handle(
            _FakeMatch(), {"black": ["^DESK"]}, {"op": "capture"}
        )
        self.assertEqual(list(reply["captured"]), ["a"])
        self.assertEqual(self.captured, [("a", ("^DESK",))])

    def test_an_exact_frame_as_a_layer_leaves_out_what_layers_do(self):
        calls = []
        self.capture.exact = lambda predictor, mix, mode, camera, black, *rest: (
            calls.append(tuple(black)) or np.zeros((2, 2, 3), np.float32)
        )
        out = os.path.join(self.tmp, "f.npy")
        settings = {"black": ["^DESK", "!^GLASS"]}
        for as_layer in (False, True):
            request = {
                "op": "exact",
                "mode": "linear",
                "as_layer": as_layer,
                "out": out,
            }
            self.capture.handle(_FakeMatch(), settings, request)
        self.assertEqual(calls, [(), ("^DESK", "!^GLASS")])
        self.assertTrue(os.path.exists(out))

    def test_the_files_own_bake_is_shot_as_it_ships(self):
        """The baseline every candidate is held to: the template's own lighting."""
        calls = []
        self.capture.file_frame = lambda mode="lit", camera=None: (
            calls.append((mode, camera)) or np.zeros((2, 2, 3), np.uint8)
        )
        out = os.path.join(self.tmp, "shipped.png")
        reply = self.capture.handle(_FakeMatch(), {}, {"op": "file", "out": out})
        self.assertEqual(reply, {"out": out})
        self.assertEqual(calls, [("lit", None)])
        self.assertTrue(os.path.isfile(out))

    def test_a_service_answers_over_http_and_stops_when_asked(self):
        import socket
        import threading

        with socket.socket() as probe:  # a free loopback port
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.capture.open_session = lambda session: (
            _FakeMatch(),
            {},
            "scene.glb",
            (4, 4),
        )
        thread = threading.Thread(
            target=self.capture.serve, args=("session.json",), kwargs={"port": port}
        )
        thread.start()
        url = f"http://127.0.0.1:{port}/"
        for _ in range(100):  # until it listens
            try:
                reply = LookCapture.client(url).invoke("status", timeout=5)
                break
            except OSError:
                thread.join(0.05)
        self.assertEqual(reply["layers"], 2)
        with self.assertRaises(RuntimeError) as ctx:  # an error is reported, not fatal
            LookCapture.client(url).invoke("sparkle", timeout=5)
        self.assertIn("sparkle", str(ctx.exception))
        self.assertEqual(
            LookCapture.client(url).invoke("stop", timeout=5), {"stopping": True}
        )
        thread.join(10)
        self.assertFalse(thread.is_alive())

    def test_a_service_sleeps_its_page_between_requests(self):
        """An idle viewer still draws every frame -- in software GL, cores a
        bake is waiting on -- so the page sleeps whenever no request runs."""
        import socket
        import threading

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.capture.open_session = lambda session: (_FakeMatch(), {}, "x", (4, 4))
        states = []
        self.capture.idle = lambda asleep=True: states.append(asleep)
        thread = threading.Thread(
            target=self.capture.serve, args=("session.json",), kwargs={"port": port}
        )
        thread.start()
        url = f"http://127.0.0.1:{port}/"
        for _ in range(100):
            try:
                LookCapture.client(url).invoke("status", timeout=5)
                break
            except OSError:
                thread.join(0.05)
        with self.assertRaises(RuntimeError):  # an unknown op never wakes it
            LookCapture.client(url).invoke("sparkle", timeout=5)
        with self.assertRaises(RuntimeError):  # a failed op sleeps it again
            LookCapture.client(url).invoke("file", timeout=5)  # no out
        LookCapture.client(url).invoke("stop", timeout=5)
        thread.join(10)
        self.assertEqual(states, [True, False, True, False, True])

    def _bake_folder(self, root, name, kind="light"):
        import cv2

        folder = os.path.join(root, name)
        os.makedirs(folder)
        cv2.imwrite(
            os.path.join(folder, "ROOM_Lightmap.exr"), np.ones((4, 4, 3), np.float32)
        )
        spec = {"lights": [{"type": "area", "pos": [0, 2, 0]}]}
        with open(os.path.join(folder, "layer.json"), "w") as f:
            json.dump(
                {
                    "name": name,
                    "kind": kind,
                    "spec": spec,
                    "gain": 3.14,
                    "files": ["ROOM_Lightmap.exr"],
                },
                f,
            )
        return folder, spec

    def test_a_bake_folder_outside_the_contract_is_refused(self):
        """What a host's baker wrote is held to LightLayerInfo before anything is
        captured or stored."""
        from pythontk import LightLayers

        folder, _spec = self._bake_folder(self.tmp, "rim")
        info_path = os.path.join(folder, "layer.json")
        with open(info_path) as f:
            info = json.load(f)
        info.pop("kind")
        with open(info_path, "w") as f:
            json.dump(info, f)
        self.capture.shoot = lambda *a, **k: self.fail("captured a malformed layer")
        with self.assertRaises(ValueError) as ctx:
            self.capture.capture_bake(LightLayers(os.path.join(self.tmp, "store")), folder)
        self.assertIn("kind", str(ctx.exception))

    def test_a_bake_folder_becomes_a_layer(self):
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        from pythontk import LightLayers

        folder, spec = self._bake_folder(self.tmp, "rim")
        sent = []
        self.capture.relight = lambda bakes, emissive=None, **kw: (
            sent.append(bakes) or 1
        )
        self.capture.shoot = lambda mode="lit", camera=None, black=(), **kw: np.full(
            (2, 3, 3), 0.25, np.float32
        )
        store = LightLayers(os.path.join(self.tmp, "store"))
        self.capture.capture_bake(store, folder)
        self.assertEqual(list(sent[0]), ["ROOM_Lightmap.exr"])
        np.testing.assert_allclose(sent[0]["ROOM_Lightmap.exr"], 1.0)
        info = store.info("rim")
        self.assertEqual(
            (info["kind"], info["spec"], info["gain"]), ("light", spec, 3.14)
        )
        self.assertEqual(list(store.bake_files("rim")), ["ROOM_Lightmap.exr"])
        np.testing.assert_allclose(store.load("rim"), 0.25)

    def test_an_emission_bakes_target_reaches_the_store(self):
        """What a mix weight scales (LookMatch.to_rig): without it an emission
        layer could be mixed but never built into a rig."""
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        from pythontk import LightLayers

        folder, _spec = self._bake_folder(self.tmp, "glow", kind="emission")
        info_path = os.path.join(folder, "layer.json")
        with open(info_path) as f:
            info = json.load(f)
        target = {"node": "MAT_PANELS", "attr": "emissive_intensity", "scale": 3.14}
        info["target"] = target
        with open(info_path, "w") as f:
            json.dump(info, f)
        self.capture.relight = lambda bakes, emissive=None, **kw: 1
        self.capture.shoot = lambda mode="lit", camera=None, black=(), **kw: np.zeros(
            (2, 3, 3), np.float32
        )
        store = LightLayers(os.path.join(self.tmp, "store"))
        self.capture.capture_bake(store, folder)
        self.assertEqual(store.info("glow")["target"], target)

    def test_a_bakes_scene_record_reaches_the_store(self):
        """Which save of the scene a layer came from: what tells a layer baked
        before the scene changed (LookLoop.stale)."""
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        from pythontk import LightLayers

        folder, _spec = self._bake_folder(self.tmp, "rim")
        info_path = os.path.join(folder, "layer.json")
        with open(info_path) as f:
            info = json.load(f)
        info["scene"] = {"file": "room.ma", "mtime": 1760000000.0, "bytes": 2048}
        with open(info_path, "w") as f:
            json.dump(info, f)
        self.capture.relight = lambda bakes, emissive=None, **kw: 1
        self.capture.shoot = lambda mode="lit", camera=None, black=(), **kw: np.zeros(
            (2, 3, 3), np.float32
        )
        store = LightLayers(os.path.join(self.tmp, "store"))
        self.capture.capture_bake(store, folder)
        self.assertEqual(store.info("rim")["scene"], info["scene"])

    def test_a_bake_folders_lights_light_their_highlights_in_the_layer(self):
        """A layer's lights (layer.json, the record's form) reach the capture in
        metres, drawn into the probe's highlight copies as the deliverable's are,
        so the layer holds its highlights."""
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        from pythontk import LightLayers

        folder, _spec = self._bake_folder(self.tmp, "key")
        info_path = os.path.join(folder, "layer.json")
        with open(info_path) as f:
            info = json.load(f)
        info["lights"] = [
            {
                "name": "K",
                "type": "point",
                "position": [0.0, 100.0, 0.0],
                "intensity": 10000.0,
                "unit_scale": 0.01,
            }
        ]
        with open(info_path, "w") as f:
            json.dump(info, f)
        sent = []
        self.capture.relight = lambda bakes, emissive=None, probe=None, lights=None: (
            sent.append(lights) or 1
        )
        self.capture.shoot = lambda mode="lit", camera=None, black=(), **kw: np.zeros(
            (2, 3, 3), np.float32
        )
        self.capture.capture_bake(LightLayers(os.path.join(self.tmp, "store")), folder)
        (light,) = sent[0]
        self.assertEqual(light["position"], [0.0, 1.0, 0.0])
        self.assertAlmostEqual(light["intensity"], 1.0)  # candela over metres

    def test_recapturing_a_layer_keeps_its_bake(self):
        """A re-capture shoots a layer from the bake the store holds: handing
        those files back to add, which empties the layer's bake folder before
        it copies, lost them (FileNotFoundError) with the frame half-stored."""
        from pythontk import LightLayers

        store = LightLayers(os.path.join(self.tmp, "store"))
        incoming = os.path.join(self.tmp, "incoming")
        os.makedirs(incoming)
        for name in ("ROOM_Lightmap.exr", "SCENE_Probe.exr"):
            with open(os.path.join(incoming, name), "wb") as f:
                f.write(name.encode("ascii"))
        store.add(
            "key",
            np.zeros((2, 3, 3), np.float32),
            spec={"type": "spot"},
            bakes=[os.path.join(incoming, n) for n in sorted(os.listdir(incoming))],
            gain=2.0,
        )
        capture = LookCapture("scene.glb")
        capture.exact = lambda *a, **k: np.full((2, 3, 3), 0.5, np.float32)
        capture.capture_layer(store, "key")
        again = LightLayers(store.root)
        files = again.bake_files("key")
        self.assertEqual(sorted(files), ["ROOM_Lightmap.exr", "SCENE_Probe.exr"])
        for name, path in files.items():
            with open(path, "rb") as f:
                self.assertEqual(f.read(), name.encode("ascii"))
        np.testing.assert_allclose(again.load("key"), 0.5)
        self.assertEqual((again.info("key")["spec"], again.info("key")["gain"]), ({"type": "spot"}, 2.0))

    def test_capture_from_a_folder_takes_every_layer_in_it(self):
        root = os.path.join(self.tmp, "job")
        self._bake_folder(root, "a")
        self._bake_folder(root, "b", kind="emission")
        os.makedirs(os.path.join(root, "not_a_layer"))
        taken = []
        self.capture.capture_bake = lambda layers, folder, black=(), replace=True: (
            taken.append((os.path.basename(folder), tuple(black)))
        )
        reply = self.capture.handle(
            _FakeMatch(), {"black": ["^DESK"]}, {"op": "capture", "source": root}
        )
        self.assertEqual(sorted(reply["captured"]), ["a", "b"])
        self.assertEqual(sorted(taken), [("a", ("^DESK",)), ("b", ("^DESK",))])

    def test_an_exact_frame_can_swap_a_bake_file_for_one_on_disk(self):
        import cv2

        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        desk = os.path.join(self.tmp, "desk.exr")
        cv2.imwrite(desk, np.full((2, 4, 3), 7.0, np.float32))

        class Predictor:
            @staticmethod
            def bakes(mix):
                return {
                    "ROOM_Lightmap.exr": np.ones((2, 2, 3), np.float32),
                    "SCENE_Probe.exr": np.zeros((2, 4, 3), np.float32),
                }

            @staticmethod
            def lights(mix):
                return []

        sent = []
        self.capture.relight = lambda bakes, emissive=None, probe=None, lights=None: (
            sent.append((bakes, probe))
        )
        self.capture.shoot = lambda mode="lit", camera=None, black=(), **kw: np.zeros(
            (2, 2, 3), np.float32
        )
        self.capture.exact(
            Predictor,
            None,
            "linear",
            swap={"SCENE_Probe.exr": desk},
            probe={"position": [0, 1, 0]},
        )
        bakes, probe = sent[0]
        np.testing.assert_allclose(bakes["SCENE_Probe.exr"], 7.0)
        np.testing.assert_allclose(bakes["ROOM_Lightmap.exr"], 1.0)
        self.assertEqual(probe, {"position": [0, 1, 0]})

    def test_an_exact_frame_carries_the_mixs_lights_by_default(self):
        """The frame is what ships: the mix's lights drawn in as highlights."""

        class Predictor:
            @staticmethod
            def bakes(mix):
                return {}

            @staticmethod
            def lights(mix):
                return [
                    {
                        "name": "K",
                        "type": "point",
                        "position": [0.0, 100.0, 0.0],
                        "intensity": 10000.0,
                        "unit_scale": 0.01,
                    }
                ]

        sent = []
        self.capture.relight = lambda bakes, emissive=None, probe=None, lights=None: (
            sent.append(lights)
        )
        self.capture.shoot = lambda *a, **k: np.zeros((2, 2, 3), np.float32)
        self.capture.exact(Predictor, None)
        (light,) = sent[0]
        self.assertEqual(light["position"], [0.0, 1.0, 0.0])
        self.capture.exact(Predictor, None, lights=[])  # an explicit list wins
        self.assertEqual(sent[1], [])

    def test_an_object_id_render_attaches_to_a_target(self):
        from pythontk.core_utils.engines.lighting.look_match.target import (
            LookRegion,
            LookTarget,
        )

        img = np.zeros((4, 6, 3), np.uint8)
        img[:, 3:] = (0, 0, 1)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "PNG")
        legend = [{"color": [0, 0, 1], "names": ["TABLE", "TRAY"], "material": "M"}]
        self.capture._call = lambda name, options: {
            "png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "legend": legend,
        }
        ref = os.path.join(self.tmp, "ref.png")
        Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(ref)
        target_path = os.path.join(self.tmp, "target.json")
        LookTarget(
            ref,
            [
                LookRegion(
                    "tray", "area", {"objects": ["^TRAY$"]}, ref={"rect": [0, 0, 1, 1]}
                )
            ],
        ).save(target_path)
        out = os.path.join(self.tmp, "ours_ids.png")
        reply = self.capture.handle(
            _FakeMatch(), {}, {"op": "ids", "out": out, "target": target_path}
        )
        self.assertEqual(reply["objects"], 1)
        target = LookTarget.load(target_path)
        self.assertEqual(target.ids["image"], "ours_ids.png")
        mask = target.region("tray").mask("ours", (6, 4))
        self.assertTrue(mask[:, 3:].all() and not mask[:, :3].any())

    def test_an_unknown_request_is_refused_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            self.capture.handle(_FakeMatch(), {}, {"op": "sparkle"})
        self.assertIn("sparkle", str(ctx.exception))


class TestSpawn(unittest.TestCase):
    """A fresh service in a process of its own, found by the port it announces."""

    def _spawning(self, code):
        """``AppLauncher.spawn`` standing in: a Python running *code*, launched
        the way spawn launches (merged binary stdout), and stopped after."""
        import subprocess
        import sys
        from unittest import mock

        procs = []

        def spawn(app, args=None, cwd=None, env=None, **kw):
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            procs.append((args, env, proc))
            return proc

        self.addCleanup(lambda: [p.kill() for _, _, p in procs if p.poll() is None])
        return procs, mock.patch(
            "pythontk.core_utils.app_launcher._app_launcher.AppLauncher.spawn", spawn
        )

    def test_the_url_is_the_port_the_service_announces(self):
        procs, patched = self._spawning(
            "import time; print('loading'); "
            "print('[look_capture] serving on http://127.0.0.1:54321', flush=True); "
            "time.sleep(30)"
        )
        with patched:
            proc, url = LookCapture.spawn("C:/sessions/room/session.json", timeout=20)
        self.assertEqual(url, "http://127.0.0.1:54321/")
        self.assertIs(proc, procs[0][2])
        args, env, _ = procs[0]
        argv = json.loads(env["PYTHONTK_ARGV"]) if "PYTHONTK_ARGV" in env else args
        self.assertIn("serve", argv)
        self.assertEqual(argv[argv.index("--port") + 1], "0")  # never a fixed port

    def test_a_service_that_exits_first_says_why(self):
        _procs, patched = self._spawning(
            "import sys; print('No module named playwright', flush=True); sys.exit(3)"
        )
        with patched, self.assertRaises(RuntimeError) as ctx:
            LookCapture.spawn("session.json", timeout=20)
        self.assertIn("No module named playwright", str(ctx.exception))

    def test_a_service_prints_a_non_ascii_path_intact(self):
        """The service announces its template's path on a pipe: in the ANSI
        code page a letter outside it (a star) raised UnicodeEncodeError and
        the service died before it served. Launched with the environment
        spawn builds, as AppLauncher.spawn launches it."""
        import subprocess
        import sys
        from unittest import mock

        from pythontk.core_utils.process_stream import OutputStream

        procs = []

        def spawn(app, args=None, cwd=None, env=None, **kw):
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import time; print('look capture: C:/caf\\u00e9/\\u2605/room.glb', flush=True); "
                    "print('[look_capture] serving on http://127.0.0.1:54321', flush=True); "
                    "time.sleep(30)",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            procs.append(proc)
            return proc

        self.addCleanup(lambda: [p.kill() for p in procs if p.poll() is None])
        stream = OutputStream()
        with mock.patch.dict(os.environ), mock.patch(
            "pythontk.core_utils.app_launcher._app_launcher.AppLauncher.spawn", spawn
        ):
            for key in ("PYTHONIOENCODING", "PYTHONUTF8"):  # the host's own setting
                os.environ.pop(key, None)
            LookCapture.spawn("session.json", timeout=20, output=stream)
        said = [line for _, line in stream.history()]
        self.assertIn("look capture: C:/caf\u00e9/\u2605/room.glb", said)


class TestCommandLine(unittest.TestCase):
    """What the command line sends a service: paths it can open from its own folder."""

    def _sent(self, argv):
        from unittest import mock

        with mock.patch.object(LookCapture, "_cli_send") as send:
            LookCapture.cli(argv)
        (_args, request), _ = send.call_args
        return request

    def test_a_bake_folder_is_sent_absolute(self):
        request = self._sent(
            ["capture", "session.json", "--from", "bakes", "--service", "http://127.0.0.1:1/"]
        )
        self.assertEqual(request["source"], os.path.abspath("bakes"))
        request = self._sent(["capture", "session.json", "--layer", "key"])
        self.assertIsNone(request["source"])  # none given: none sent

    def test_a_swapped_bake_file_is_sent_absolute(self):
        request = self._sent(
            [
                "exact",
                "session.json",
                "--out",
                "frame.png",
                "--swap",
                "SCENE_Probe.exr=probes/desk.exr",
                "--service",
                "http://127.0.0.1:1/",
            ]
        )
        self.assertEqual(
            request["swap"], {"SCENE_Probe.exr": os.path.abspath("probes/desk.exr")}
        )


@unittest.skipUnless(
    os.environ.get("LOOK_CAPTURE_TEMPLATE")
    and os.environ.get("LOOK_CAPTURE_CAMERA")
    and LookCapture.available(),
    "set LOOK_CAPTURE_TEMPLATE (a lightmapped GLB) and LOOK_CAPTURE_CAMERA (a camera JSON); install Playwright",
)
class TestLive(unittest.TestCase):
    """The real viewer: a dark bake renders dark, a linear frame is linear."""

    @classmethod
    def setUpClass(cls):
        camera = None
        if os.environ.get(
            "LOOK_CAPTURE_CAMERA"
        ):  # a fixed view: the page's own orbit may still be settling
            with open(os.environ["LOOK_CAPTURE_CAMERA"], encoding="utf-8") as f:
                camera = json.load(f)
        cls.capture = LookCapture(
            os.environ["LOOK_CAPTURE_TEMPLATE"], camera=camera, size=(320, 188)
        ).start()

    @classmethod
    def tearDownClass(cls):
        cls.capture.stop()

    def test_a_blacked_shot_leaves_the_scene_as_it_was(self):
        """Every material, shared or not, gets its own reflection level back."""
        before = self.capture.shoot("linear")
        self.capture.shoot("linear", black=[".*"])
        np.testing.assert_allclose(self.capture.shoot("linear"), before, atol=1e-3)

    def test_a_blacked_material_reflects_nothing(self):
        """three r163+ reads a scene-lit material's level off the scene, not the material."""
        lit = {
            os.path.splitext(e["map"])[0] + ".exr": np.ones((8, 8, 3), np.float32)
            for e in self.capture.manifest["materials"].values()
        }
        self.capture.relight(lit)
        before = self.capture.shoot("linear")
        blacked = self.capture.shoot("linear", black=[".*"])
        self.assertLess(float(blacked.mean()), float(before.mean()) * 0.98)

    def test_an_emissive_preview_lasts_one_relight(self):
        """A previewed emissive level is that frame's alone: the next relight
        without one, and the file's own lighting, show the material's own."""
        images = {
            os.path.splitext(e["map"])[0] + ".exr": np.zeros((8, 8, 3), np.float32)
            for e in self.capture.manifest["materials"].values()
        }
        self.capture.relight(images)
        plain = self.capture.shoot("linear")
        self.capture.relight(images, {"pattern": ".*", "intensity": 1000.0})
        previewed = self.capture.shoot("linear")
        self.assertGreater(float(np.abs(previewed - plain).mean()), 1e-4)  # it took
        self.capture.relight(images)
        np.testing.assert_allclose(self.capture.shoot("linear"), plain, atol=1e-3)

    def test_a_black_bake_renders_black_and_a_lit_one_does_not(self):
        manifest = self.capture.manifest
        images = {
            os.path.splitext(e["map"])[0] + ".exr": np.zeros((8, 8, 3), np.float32)
            for e in manifest["materials"].values()
        }
        self.assertGreater(self.capture.relight(images), 0)
        dark = self.capture.shoot("linear")
        lit = {k: v + 1.0 for k, v in images.items()}
        self.capture.relight(lit)
        bright = self.capture.shoot("linear")
        self.capture.relight(images)
        again = self.capture.shoot("linear")
        np.testing.assert_allclose(
            again, dark, atol=1e-3
        )  # a frame is a function of its bake
        self.assertEqual(dark.shape[2], 3)
        self.assertGreater(float(bright.mean()), float(dark.mean()) + 0.05)


if __name__ == "__main__":
    unittest.main(exit=False)
