# !/usr/bin/python
# coding=utf-8
"""Tests for :class:`pythontk.PreviewServer`.

Covers the contract the live-preview loop depends on: a loopback-only bind
(the secure-context guarantee WebXR needs), the ``/manifest.json`` version
signal a polling viewer watches, republish-in-place semantics, and the
no-store caching posture that keeps a warm browser cache from pinning a stale
asset.
"""

import base64
import json
import os
import re
import struct
import sys
import unittest
import unittest.mock
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk.core_utils.app_handoff import HandoffBridge, HandoffRequest, Payload
from pythontk.file_utils.mesh_convert._mesh_convert import MeshConvert
from pythontk.file_utils.temp_artifacts import TempArtifacts
from pythontk.net_utils.preview_server import (
    VIEWER_CLOSED_PATH,
    PreviewBridge,
    PreviewDeliverer,
    PreviewServer,
)


class PreviewServerTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = TempArtifacts("test_preview_server", policy="scoped")
        self.root = Path(self.temp.dir_path())
        self.assets = Path(self.temp.dir_path())
        self.server = None

    def tearDown(self):
        if self.server is not None:
            self.server.stop()
        self.temp.cleanup()

    # -- helpers --------------------------------------------------------

    def _serve(self, **kwargs):
        kwargs.setdefault("root", self.root)
        kwargs.setdefault("port", 0)  # ephemeral: never collide with a real one
        self.server = PreviewServer(**kwargs).start()
        return self.server

    def _asset(self, name="cube.glb", data=b"glTF-stub-0"):
        path = self.assets / name
        path.write_bytes(data)
        return path

    def _get(self, path):
        with urllib.request.urlopen(self.server.url + path, timeout=5) as response:
            return response.status, response.read(), dict(response.headers)

    def _post(self, path, data=b"", origin=None):
        headers = {"Origin": origin} if origin else {}
        request = urllib.request.Request(
            self.server.url + path, data=data, headers=headers
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status

    # -- lifecycle ------------------------------------------------------

    def test_start_binds_loopback_and_reports_url(self):
        server = self._serve()
        self.assertTrue(server.is_running)
        self.assertEqual(server.host, "127.0.0.1")
        self.assertTrue(server.url.startswith("http://127.0.0.1:"))
        self.assertIsInstance(server.port, int)

    def test_default_host_is_loopback(self):
        """The secure-context guarantee is the default, not an opt-in."""
        self.assertEqual(PreviewServer(root=self.root, port=0).host, "127.0.0.1")

    def test_stop_is_idempotent_and_releases_state(self):
        server = self._serve()
        server.stop()
        server.stop()
        self.assertFalse(server.is_running)
        self.assertIsNone(server.url)

    def test_context_manager_stops_on_exit(self):
        with PreviewServer(root=self.root, port=0) as server:
            self.assertTrue(server.is_running)
        self.assertFalse(server.is_running)

    def test_explicit_port_is_taken_literally(self):
        """A pinned port must bind or fail -- never silently relocate."""
        first = self._serve()
        clash = PreviewServer(root=self.root, port=first.port)
        with self.assertRaises(OSError):
            clash.start()

    def test_preferred_port_lost_in_the_probe_gap_falls_back_to_ephemeral(self):
        """The bindability probe releases the port before the real bind.

        Another process can take it in that gap; ``port=None`` asked for
        "stable if you can, ephemeral otherwise", and that promise has to hold
        against the race too, not just against a port already taken when the
        probe ran. Simulated by making the probe lie about an occupied port.
        """
        blocker = self._serve()  # occupies its ephemeral port for real
        with unittest.mock.patch.object(
            PreviewServer, "DEFAULT_PORT", blocker.port
        ), unittest.mock.patch(
            "pythontk.net_utils.preview_server.NetUtils.is_port_bindable",
            return_value=True,
        ):
            fallback = PreviewServer(root=self.root).start()  # port=None
        try:
            self.assertTrue(fallback.is_running)
            self.assertNotEqual(fallback.port, blocker.port)
        finally:
            fallback.stop()

    # -- viewer ---------------------------------------------------------

    def test_viewer_is_materialized_and_served_at_root(self):
        self._serve()
        self.assertTrue((self.root / "index.html").is_file())
        status, body, _ = self._get("")
        self.assertEqual(status, 200)
        self.assertIn(b"navigator.xr", body)

    def test_viewer_is_not_overwritten(self):
        (self.root / "index.html").write_text("<!-- hand-edited -->", encoding="utf-8")
        self._serve()
        self.assertEqual(
            (self.root / "index.html").read_text(encoding="utf-8"),
            "<!-- hand-edited -->",
        )

    def test_managed_root_refreshes_a_stale_viewer(self):
        """A session-long server must never keep serving a previous version's page.

        The deliverer holding a server lives for the whole DCC session, so its
        temp root is created once and reused; a never-overwrite rule stranded
        such a session on whatever viewer shipped the day it started.
        """
        server = PreviewServer(port=0)  # managed (temp) root
        try:
            stale = Path(server.root) / "index.html"
            stale.write_text("STALE VIEWER", encoding="utf-8")
            server.start()
            self.assertNotIn("STALE", stale.read_text(encoding="utf-8"))
            self.assertIn("navigator.xr", stale.read_text(encoding="utf-8"))
        finally:
            server.stop()

    def test_publish_refreshes_the_viewer_on_a_running_server(self):
        """`start` is idempotent, so publish is the only hook a live session has.

        The deliverer's server outlives every push, so refreshing only in
        `start` meant an edited viewer could never reach an already-running
        session — the earlier version of this fix was inert for exactly the
        case it claimed to address.
        """
        server = PreviewServer(port=0).start()  # managed root
        try:
            page = Path(server.root) / "index.html"
            page.write_text("STALE VIEWER", encoding="utf-8")
            server.publish(self._asset())
            self.assertIn("navigator.xr", page.read_text(encoding="utf-8"))
        finally:
            server.stop()

    def test_publish_does_not_clobber_a_caller_supplied_viewer(self):
        """An explicit root is a working directory; edits there must survive."""
        (self.root / "index.html").write_text("<!-- mine -->", encoding="utf-8")
        server = self._serve()
        server.publish(self._asset())
        self.assertEqual(
            (self.root / "index.html").read_text(encoding="utf-8"), "<!-- mine -->"
        )

    def test_viewer_can_be_disabled(self):
        self._serve(viewer=False)
        self.assertFalse((self.root / "index.html").exists())

    # -- manifest -------------------------------------------------------

    def test_manifest_before_publish_has_no_asset(self):
        self._serve()
        _, body, _ = self._get("manifest.json")
        manifest = json.loads(body)
        self.assertEqual(manifest["version"], 0)
        self.assertIsNone(manifest["asset"])

    def test_publish_bumps_version_and_serves_asset(self):
        server = self._serve(title="Selection")
        version = server.publish(self._asset(data=b"first"))
        self.assertEqual(version, 1)

        _, body, _ = self._get("manifest.json")
        manifest = json.loads(body)
        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest["asset"], "scene.glb")
        self.assertEqual(manifest["title"], "Selection")
        self.assertIsNotNone(manifest["updated"])

        status, payload, _ = self._get("scene.glb")
        self.assertEqual(status, 200)
        self.assertEqual(payload, b"first")

    def test_republish_replaces_bytes_under_a_stable_name(self):
        """The viewer polls one URL; only the version tells it to reload."""
        server = self._serve()
        server.publish(self._asset("a.glb", b"first"))
        server.publish(self._asset("b.glb", b"second"))

        self.assertEqual(server.version, 2)
        _, payload, _ = self._get("scene.glb")
        self.assertEqual(payload, b"second")
        self.assertEqual(json.loads(self._get("manifest.json")[1])["asset"], "scene.glb")

    def test_publish_respects_an_explicit_name(self):
        server = self._serve()
        server.publish(self._asset(data=b"named"), name="proxy.glb")
        self.assertEqual(self._get("proxy.glb")[1], b"named")

    def test_publish_leaves_no_partial_file_behind(self):
        """The staged `.part` write must be renamed away, not left in the tree."""
        server = self._serve()
        server.publish(self._asset())
        self.assertEqual(list(self.root.glob("*.part")), [])

    def test_publish_move_consumes_the_source(self):
        server = self._serve()
        src = self._asset(data=b"moved")
        server.publish(src, move=True)
        self.assertFalse(src.exists())
        self.assertEqual(self._get("scene.glb")[1], b"moved")

    def test_publish_missing_source_raises(self):
        server = self._serve()
        with self.assertRaises(FileNotFoundError):
            server.publish(self.assets / "nope.glb")

    # -- caching --------------------------------------------------------

    def test_responses_are_not_cacheable(self):
        """A cached GLB would pin the preview to whatever was pushed first."""
        server = self._serve()
        server.publish(self._asset())
        for path in ("manifest.json", "scene.glb"):
            headers = self._get(path)[2]
            self.assertIn("no-store", headers.get("Cache-Control", ""))

    def test_unknown_path_is_404(self):
        self._serve()
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("missing.glb")
        self.assertEqual(ctx.exception.code, 404)

    # -- viewer liveness ------------------------------------------------
    #
    # What drives the "open a tab or not" decision on every push. Getting it
    # wrong is invisible in both directions: too eager and a second tab covers
    # the DCC on every push, too lazy and pushes go to a page that is gone.

    def test_no_viewer_before_anything_polls(self):
        self.assertFalse(self._serve().has_viewer())

    def test_manifest_poll_marks_a_viewer_present(self):
        server = self._serve()
        self._get("manifest.json")
        self.assertTrue(server.has_viewer())

    def test_asset_request_alone_does_not_count_as_a_viewer(self):
        """Only the polled manifest proves a page is still there.

        An asset GET happens once per publish and a favicon probe happens once
        per tab, so counting either would keep reporting a viewer long after
        the tab that fetched it closed.
        """
        server = self._serve()
        server.publish(self._asset())
        self._get("scene.glb")
        self.assertFalse(server.has_viewer())

    def test_viewer_lapses_once_polling_stops(self):
        """A page killed with the browser leaves no beacon; the window catches it."""
        server = self._serve()
        self._get("manifest.json")
        self.assertTrue(server.has_viewer())
        # Rewind the last poll past the window rather than sleeping through it.
        server._viewer_seen -= server.VIEWER_TIMEOUT + 1
        self.assertFalse(server.has_viewer())

    def test_close_beacon_clears_the_viewer_immediately(self):
        """A closed tab must be known now, not after the throttling window.

        The timeout has to clear a minute to tolerate hidden-tab timer
        throttling, so without the beacon "close the tab, push again" would
        show nothing for 90 seconds.
        """
        server = self._serve()
        self._get("manifest.json")
        self.assertEqual(self._post("viewer-closed"), 204)
        self.assertFalse(server.has_viewer())

    def test_viewer_page_beacons_the_path_the_handler_listens_on(self):
        """The page and the handler agree on one path, or the beacon is a 404."""
        self._serve()
        page = (self.root / "index.html").read_text(encoding="utf-8")
        self.assertIn(f"sendBeacon('{VIEWER_CLOSED_PATH}'", page)

    def test_a_lightmapped_model_keeps_some_environment_lighting(self):
        """Zero environment on a lightmapped model renders it dead flat.

        three.js adds ``lightMap`` irradiance through ``BRDF_Lambert``, which
        has no normal term -- a bake supplies light that does not vary with the
        surface normal. Switch the viewer's own lighting fully off (as this
        once did) and nothing left in the render samples the normal at all, so
        every correctly-bound normal map, all roughness variation and every
        specular highlight go inert. Measured on a production room GLB: 51 of
        57 materials lightmapped, all 54 normal maps bound at texCoord 0, and
        no surface detail visible anywhere. The environment is what stays on to
        carry the normal-dependent specular term, so a zero (or absent) level
        here is the regression.

        The dimming is applied PER MATERIAL, and getting there needs one
        specific trick: a material-level ``envMapIntensity`` is overwritten
        from the scene value for precisely the materials this affects
        (``isMeshStandardMaterial`` with no ``envMap`` of its own, which is
        every GLTFLoader material), so setting it alone renders the whole fix
        inert -- verified against the three.js 0.169.0 source, and the first
        cut of this fix did exactly that. Assigning the shared environment
        onto the material opts it out of that override, which is what makes
        the per-material value stick. A scene-wide lever would work too, but
        it dims the model's UN-baked props along with the baked geometry, and
        a scene is routinely partly baked.
        """
        self._serve()
        page = (self.root / "index.html").read_text(encoding="utf-8")
        match = re.search(r"const LIGHTMAP_ENV_INTENSITY = ([0-9.]+)", page)
        self.assertIsNotNone(
            match, "the viewer no longer declares a lightmapped-scene env level"
        )
        self.assertGreater(float(match.group(1)), 0.0)
        # Per material, and only for the materials the manifest named.
        self.assertIn("material.envMapIntensity = bakeOnly ? 0 : LIGHTMAP_ENV_INTENSITY", page)
        self.assertIn("material.envMap = scene.environment", page)
        # The override opt-out is only safe while disposeModel spares the
        # shared texture; without that guard the second push runs unlit.
        self.assertIn("value !== scene.environment", page)
        # And the scene-wide lever must NOT be keyed off the bake any more.
        self.assertNotIn("scene.environmentIntensity = lightmapped", page)

    def test_published_rendering_policy_matches_the_viewer(self):
        """The hand-off's lighting recipe must be what the viewer actually does.

        ``MeshConvert.RENDERING_POLICY`` ships inside every deliverable as
        ``handoff.rendering`` -- it is how a recipient reproduces the look the
        asset was approved in, since the asset carries no lights of its own.
        The viewer declares the same numbers in JavaScript, which no Python test
        executes, so nothing but this pins the two together. A published recipe
        that quietly stops matching the reference render is worse than none:
        it is followed exactly and still yields a different image, and the
        recipient has no way to tell it was the recipe that was wrong.
        """
        from pythontk import MeshConvert

        self._serve()
        page = (self.root / "index.html").read_text(encoding="utf-8")
        policy = MeshConvert.RENDERING_POLICY

        for const, published in (
            ("DEFAULT_KEY_INTENSITY", policy["keyLight"]["intensity"]),
            (
                "LIGHTMAP_ENV_INTENSITY",
                policy["lightmappedMaterials"]["envMapIntensity"],
            ),
        ):
            match = re.search(rf"const {const} = ([0-9.]+)", page)
            self.assertIsNotNone(match, f"the viewer no longer declares {const}")
            self.assertEqual(
                float(match.group(1)),
                float(published),
                f"{const} disagrees with the published handoff.rendering policy",
            )

        # The key light's OFF-CONDITION, not just its magnitude. 0.9 stayed 0.9
        # right through the 2026-08-12 regression that gated the light on the
        # asset being FULLY baked rather than partly, so a number-only check
        # cannot see a condition invert -- and `disabled_when` is not a note,
        # it is a field a recipient ACTS on: read the wrong way it puts a
        # directional light on geometry whose lighting is already baked in.
        #
        # Asserted as "must not say the WRONG thing" rather than "must say this
        # exact thing". Demanding a keyword (an earlier cut of this required
        # "any") fails on a correct rewording -- "when the asset contains a
        # lightmapped material" means the same and has no "any" -- and a test
        # that fails on correct behaviour gets weakened rather than fixed. The
        # structural answer is for the viewer to READ this policy instead
        # (deferred -- see BACKLOG, "the WebXR viewer defines the viewing policy
        # it is supposed to consume").
        disabled_when = policy["keyLight"].get("disabled_when", "").lower()
        self.assertTrue(
            disabled_when.strip(),
            "the recipe no longer states WHEN the key light is disabled -- the "
            "half a recipient cannot infer from the intensity",
        )
        for misreading in ("all material", "every material", "fully baked"):
            self.assertNotIn(
                misreading,
                disabled_when,
                "the published off-condition has drifted to the FULLY-baked "
                "reading the viewer was fixed away from",
            )

        # The named pieces of the rig, spelled as the viewer builds them.
        self.assertIn("THREE.ACESFilmicToneMapping", page)
        self.assertEqual(policy["renderer"]["toneMapping"], "ACESFilmic")
        self.assertIn("new RoomEnvironment()", page)
        self.assertIn("RoomEnvironment", policy["environment"]["source"])
        blur = re.search(r"pmrem\.fromScene\(new RoomEnvironment\(\), ([0-9.]+)\)", page)
        self.assertIsNotNone(blur, "the viewer no longer prefilters the environment")
        # Compared as a NUMBER. Substring-matching this against the prose form
        # ("PMREM, blur 0.04", as this first did) silently accepts a viewer that
        # changed the blur to 0.0 -- "0.0" is a substring of "0.04".
        self.assertEqual(
            float(blur.group(1)), float(policy["environment"]["prefilterBlur"])
        )
        env = re.search(r"scene\.environmentIntensity = ([0-9.]+)", page)
        self.assertIsNotNone(env, "the viewer no longer sets an environment level")
        self.assertEqual(
            float(env.group(1)), float(policy["environment"]["intensity"])
        )

    def test_the_key_light_goes_off_for_any_bake_at_all(self):
        """A partly-baked model must not wear the key light over its bake.

        The key light is scene-wide -- it cannot be withdrawn per material
        without render layers -- so gating it on the model being FULLY baked
        put a 0.9 directional light on top of geometry whose lighting is
        already in its lightmap. A room is routinely partly baked (measured:
        51 of 57 materials), so that gate never opened and every baked room
        rendered blown out. The added light is invisible to the bake settings,
        which is the tell: lowering the Maya lights and re-baking changes
        nothing, because the extra term is applied downstream of the EXR.

        The un-baked props the gate was added for are not left dark. They keep
        the FULL environment via the per-material split above (they were on
        0.25 when the key light mattered), and the environment is normal- and
        view-dependent, so their normal maps and specular survive without it.

        This also restores the "bake only" toggle as real isolation: it is
        guarded on ``lightmapped``, so wherever it can be flipped the key light
        is already off and a baked material is lit by its bake alone. Un-baked
        props keep the full environment under the toggle -- they have no bake
        to isolate, and blacking them out would answer no question.
        """
        self._serve()
        page = (self.root / "index.html").read_text(encoding="utf-8")
        self.assertIn("keyLight.intensity = lightmapped ? 0 : DEFAULT_KEY_INTENSITY", page)
        # The DECLARATION, not the bare word: this file's comments discuss the
        # removed gate by name, and a test that fires on prose about a dead
        # mechanism teaches the next reader to delete the explanation rather
        # than the code.
        self.assertNotIn("const fullyBaked", page)
        # The isolation holds only while the toggle stays guarded on the same
        # condition the key light is keyed to; without this guard "bake only is
        # reachable" and "key light is off" stop being the same state. Matched
        # loosely on purpose -- a formatter adding braces to the guard clause
        # would change nothing about the behaviour being pinned.
        self.assertRegex(page, r"if \(!lightmapped\)\s*\{?\s*return")

    def test_timeout_tolerates_hidden_tab_throttling(self):
        """Browsers throttle a hidden tab's timers to ~1/min; 90s clears that."""
        self.assertGreater(PreviewServer.VIEWER_TIMEOUT, 60)

    def test_the_close_beacon_accepts_its_own_origin(self):
        server = self._serve()
        self._get("manifest.json")
        self.assertEqual(
            self._post(VIEWER_CLOSED_PATH, origin=server.url.rstrip("/")), 204
        )
        self.assertFalse(server.has_viewer())

    def test_the_close_beacon_is_accepted_from_the_localhost_spelling(self):
        """`url` is always 127.0.0.1, but the page is just as validly `localhost`.

        Matching the beacon against the server's own spelling would 403 the
        real viewer whenever it was reached by name — silently, since the only
        symptom is falling back to the 90s timeout.
        """
        server = self._serve()
        self._get("manifest.json")
        origin = f"http://localhost:{server.port}"
        request = urllib.request.Request(
            f"{origin}/{VIEWER_CLOSED_PATH}", data=b"", headers={"Origin": origin}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 204)
        self.assertFalse(server.has_viewer())

    def test_a_cross_origin_close_beacon_is_rejected(self):
        """A beacon needs no preflight, so any site could otherwise send one.

        The damage is small — the next push pops a tab over the DCC — but it
        is the one request that changes server state, and it costs three lines
        to require it come from the page we served.
        """
        server = self._serve()
        self._get("manifest.json")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post(VIEWER_CLOSED_PATH, origin="https://evil.example")
        self.assertEqual(ctx.exception.code, 403)
        self.assertTrue(server.has_viewer())

    def test_unknown_post_is_404(self):
        self._serve()
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post("whatever")
        self.assertEqual(ctx.exception.code, 404)

    def test_stop_forgets_the_viewer(self):
        """A restart usually lands on a new port, so the old page is gone.

        Carrying the flag across would suppress the tab a restarted server
        most needs — the one case where nothing is watching for certain.
        """
        server = self._serve()
        self._get("manifest.json")
        server.stop()
        self.assertFalse(server.has_viewer())


class _StubBridge(HandoffBridge):
    """A minimal hand-off bridge standing in for a DCC-side one.

    Real subclasses get ``_resolve_objects`` / ``_produce`` from the Maya or
    Blender export mixin; only those two hooks are DCC-specific, so stubbing
    them exercises the same delivery path the engines use.
    """

    payload_prefix = "test_preview_bridge"

    def _resolve_objects(self, objects):
        return objects or ["stub_object"]

    def _produce(self, objects, request):
        path = self._make_payload_path(extension=".fbx")
        Path(path).write_bytes(b"fake-fbx-payload")
        return Payload(primary=path)


def _sidecar(sections):
    """A sidecar envelope as `PreviewBridge._attach_sidecar` shapes it.

    Deliverer tests build their fixtures through this rather than as bare
    section maps, so a drift between what the producer wraps and what the
    deliverer unwraps fails here instead of only in a live DCC.
    """
    return {
        "version": MeshConvert.SIDECAR_VERSION,
        "source": {"application": "test", "version": "0"},
        "asset": "scene.fbx",
        "color_space": "linear",
        "sections": sections,
    }


class _StubPreviewBridge(PreviewBridge):
    """A minimal `PreviewBridge` exercising the real producer-side envelope.

    Unlike `_StubBridge` this attaches its sidecar through `_attach_sidecar`
    -- the same call the Maya and Blender producers make -- so the
    producer -> deliverer pair is tested end to end against the one schema
    owner rather than against a hand-built dict.
    """

    payload_prefix = "test_preview_envelope"
    sections = {"emissive": {"m": {"color": [1, 0, 0]}}}

    def _resolve_objects(self, objects):
        return objects or ["stub_object"]

    def _produce(self, objects, request):
        path = self._make_payload_path(extension=".fbx")
        Path(path).write_bytes(b"fake-fbx-payload")
        return self._attach_sidecar(
            Payload(primary=path),
            self.sections,
            source={"application": "stub", "version": "0"},
        )


class PreviewDelivererTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = TempArtifacts("test_preview_deliverer", policy="scoped")
        self.server = PreviewServer(root=self.temp.dir_path(), port=0).start()
        self.bridge = _StubBridge()
        self.converted = []

    def tearDown(self):
        self.server.stop()
        self.temp.cleanup()

    def _fbx(self):
        """A stand-in exported payload for a direct `deliver` call."""
        path = Path(self.temp.path(extension=".fbx"))
        path.write_bytes(b"fake-fbx-payload")
        return str(path)

    def _fake_convert(self, src, dst=None, **kwargs):
        """Stand in for FBX2glTF: record the call and write a GLB-shaped file.

        A *parseable* GLB, not just the magic bytes: the sidecar step opens the
        converted file once for all its sections, so a stub that only looked
        like a GLB from four bytes away would now fail at the container rather
        than exercise the appliers under test.
        """
        self.converted.append({"src": src, "dst": dst, **kwargs})
        payload = json.dumps({"asset": {"version": "2.0"}}).encode("utf-8")
        payload += b" " * ((4 - (len(payload) % 4)) % 4)
        Path(dst).write_bytes(
            b"glTF"
            + struct.pack("<I", 2)
            + struct.pack("<I", 12 + 8 + len(payload))
            + struct.pack("<I", len(payload))
            + b"JSON"
            + payload
        )
        return dst

    def _deliver(self, **kwargs):
        """One push through a deliverer built from *kwargs* (its own defaults)."""
        kwargs.setdefault("server", self.server)
        kwargs.setdefault("open_browser", False)
        self.bridge.deliverer = PreviewDeliverer(**kwargs)
        return self._push()

    def _push(self, **extras):
        """One push through the bridge's *existing* deliverer.

        *extras* ride on `HandoffRequest.extras` -- the per-request knobs. Kept
        separate from `_deliver` because the request-scoped tests need two
        pushes over one deliverer, which is exactly where a knob held only as
        instance state leaks from the first push into the second.
        """
        target = "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.fbx_to_glb"
        with unittest.mock.patch(target, side_effect=self._fake_convert):
            return self.bridge.send(**extras)

    def test_deliver_publishes_and_reports_the_url(self):
        result = self._deliver()
        self.assertEqual(result["version"], 1)
        self.assertEqual(result["url"], self.server.url)
        self.assertEqual(result["asset"], "scene.glb")
        self.assertEqual((Path(self.server.root) / "scene.glb").read_bytes()[:4], b"glTF")

    def test_glb_is_allocated_through_the_bridge_payload_workflow(self):
        """The GLB must join the bridge's swept prefix, not be derived ad hoc.

        A path built from the FBX would still land in a managed directory, but
        outside any allocation the age-gated sweep knows about — so a hard DCC
        crash (no `finally`, no atexit) would leak it permanently.
        """
        self._deliver()
        dst = Path(self.converted[0]["dst"])
        self.assertEqual(dst.suffix, ".glb")
        self.assertTrue(
            dst.name.startswith(_StubBridge.payload_prefix),
            f"GLB {dst.name!r} is outside the {_StubBridge.payload_prefix!r} sweep scope",
        )

    def test_conversion_never_prompts(self):
        """A DCC has no tty; a prompting install would raise on the first push."""
        self._deliver()
        self.assertIs(self.converted[0]["prompt"], False)

    def test_publish_moves_the_glb_out_of_temp(self):
        self._deliver()
        self.assertFalse(Path(self.converted[0]["dst"]).exists())

    def test_repeated_pushes_bump_the_version(self):
        self._deliver()
        result = self._deliver()
        self.assertEqual(result["version"], 2)

    def test_failed_conversion_reports_none_without_publishing(self):
        self.bridge.deliverer = PreviewDeliverer(server=self.server, open_browser=False)
        target = "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.fbx_to_glb"
        with unittest.mock.patch(target, side_effect=RuntimeError("FBX2glTF exploded")):
            self.assertIsNone(self.bridge.send())
        self.assertEqual(self.server.version, 0)

    def test_missing_payload_reports_none(self):
        self.bridge.deliverer = PreviewDeliverer(server=self.server, open_browser=False)
        self.assertIsNone(
            self.bridge.deliverer.deliver(self.bridge, Payload(primary=None), None)
        )

    def test_ktx2_missing_encoder_raises_instead_of_shipping_unoptimized(self):
        """docs/webxr_preview.md promises the push raises with the install URL
        when toktx is missing under KTX2 mode -- it must not be swallowed by
        the broad `except Exception` around the optimize step and silently
        ship the unoptimized GLB."""
        from pythontk.img_utils.ktx2_encoder import Ktx2Encoder

        with (
            unittest.mock.patch.object(Ktx2Encoder, "available", return_value=False),
            unittest.mock.patch.object(
                Ktx2Encoder,
                "resolve_toktx",
                side_effect=FileNotFoundError("KTX2 encoding requires 'toktx'"),
            ),
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                self._deliver(texture_format="KTX2")
        self.assertIn("toktx", str(ctx.exception))
        self.assertEqual(self.server.version, 0, "must not publish a partial push")

    # `texture_format` is a per-push knob, so these assert the request-scoped
    # half: a deliverer is bound once per bridge *class*, so a format written
    # onto the instance for one high-fidelity push would otherwise stick for
    # every bridge in the session -- silently requiring toktx and paying the
    # Basis encode on every later quick-iteration push, with no way back out.

    def _optimize_formats(self, *pushes):
        """Run *pushes* (each a dict of request extras) and return the
        `image_format` each one asked the optimizer for, plus the eager
        KTX2-encoder resolves they triggered."""
        from pythontk.img_utils._img_utils import ImgUtils

        with (
            unittest.mock.patch.object(
                MeshConvert, "optimize_glb_textures"
            ) as optimize,
            unittest.mock.patch.object(ImgUtils, "resolve_ktx2_encoder") as resolve,
        ):
            for extras in pushes:
                self._push(**extras)
        formats = [call.kwargs["image_format"] for call in optimize.call_args_list]
        return formats, resolve.call_count

    def test_a_per_request_texture_format_does_not_leak_into_the_next_push(self):
        self.bridge.deliverer = PreviewDeliverer(server=self.server, open_browser=False)
        formats, resolves = self._optimize_formats({"texture_format": "KTX2"}, {})
        self.assertEqual(formats, ["KTX2", "WEBP"])
        # The eager `resolve_ktx2_encoder(required=True)` must lapse with the
        # request that asked for KTX2: left inherited, the next push raises
        # FileNotFoundError on a machine that never wanted the encoder.
        self.assertEqual(resolves, 1)
        # ...and the override never wrote through to the shared instance.
        self.assertEqual(self.bridge.deliverer.texture_format, "WEBP")

    def test_a_request_without_a_format_falls_back_to_the_instance_default(self):
        self.bridge.deliverer = PreviewDeliverer(
            server=self.server, open_browser=False, texture_format="KTX2"
        )
        formats, resolves = self._optimize_formats({})
        self.assertEqual(formats, ["KTX2"])
        self.assertEqual(resolves, 1)

    def test_a_falsy_instance_default_still_optimizes(self):
        """Making the format request-scoped moved it from "absent kwarg" to
        "explicit value", so a falsy instance default stopped inheriting
        ``optimize_glb_textures``' own ``WEBP`` default and started handing it
        ``None`` -- which raises inside the optimizer and is swallowed by the
        broad ``except``, silently shipping an UNOPTIMIZED GLB (the 94.7MB ->
        ~15MB pass is the whole point of the step). Both falsy forms have to
        land back on WEBP."""
        for default in (None, ""):
            with self.subTest(instance_default=default):
                self.bridge.deliverer = PreviewDeliverer(
                    server=self.server, open_browser=False, texture_format=default
                )
                formats, _ = self._optimize_formats({})
                self.assertEqual(formats, ["WEBP"])

    def test_push_forwards_the_format_as_a_request_knob_not_an_export_param(self):
        """`push(**params)` sweeps unknown kwargs into the *export* params, so
        without an explicit parameter a `push(texture_format=...)` would be
        handed to the exporter and never reach the deliverer at all."""
        bridge = _StubPreviewBridge()
        bridge.deliverer = PreviewDeliverer(server=self.server, open_browser=False)
        from pythontk.img_utils._img_utils import ImgUtils

        with (
            unittest.mock.patch.object(
                MeshConvert, "optimize_glb_textures"
            ) as optimize,
            unittest.mock.patch.object(ImgUtils, "resolve_ktx2_encoder"),
            unittest.mock.patch.object(
                MeshConvert, "fbx_to_glb", side_effect=self._fake_convert
            ),
        ):
            bridge.push(texture_format="KTX2")
            self.assertEqual(optimize.call_args.kwargs["image_format"], "KTX2")
            bridge.push()
        # Omitting it must leave the deliverer's own default in force rather
        # than overriding it with the parameter's `None` sentinel.
        self.assertEqual(optimize.call_args.kwargs["image_format"], "WEBP")

    # `webbrowser.open` is patched rather than `open_in_browser`, because the
    # method under test is what registers the launch as a viewer — stubbing it
    # out would hide the very interaction these cover.

    def test_auto_open_does_not_repeat_while_a_page_is_watching(self):
        """A second tab per push would pile up and steal focus from the DCC."""
        with unittest.mock.patch("webbrowser.open", return_value=True) as opened:
            self.assertTrue(self._deliver(open_browser="auto")["opened_browser"])
            self.assertFalse(self._deliver(open_browser="auto")["opened_browser"])
            self.assertEqual(opened.call_count, 1)

    def test_auto_open_reopens_after_the_viewer_goes_away(self):
        """The bug this replaced: a closed tab was never reopened again.

        "Has anyone published yet" was standing in for "is anyone watching",
        and the two only agree until the user closes the tab. The server lives
        for the whole DCC session, so from that point every push published to
        a page that no longer existed — silently, since the push itself
        succeeded.
        """
        with unittest.mock.patch("webbrowser.open", return_value=True) as opened:
            self._deliver(open_browser="auto")
            self.server._clear_viewer()  # the tab's unload beacon
            self.assertTrue(self._deliver(open_browser="auto")["opened_browser"])
            self.assertEqual(opened.call_count, 2)

    def test_open_browser_true_opens_every_push(self):
        with unittest.mock.patch("webbrowser.open", return_value=True) as opened:
            self._deliver(open_browser=True)
            self._deliver(open_browser=True)
            self.assertEqual(opened.call_count, 2)

    def test_open_browser_false_never_opens(self):
        with unittest.mock.patch("webbrowser.open") as opened:
            self.assertFalse(self._deliver(open_browser=False)["opened_browser"])
            opened.assert_not_called()

    def test_a_launched_browser_counts_before_its_first_poll(self):
        """Two pushes in quick succession must not race the browser's startup.

        A cold browser takes seconds to make its first request, and a push
        landing in that gap would otherwise see no polls and open a second tab.
        """
        with unittest.mock.patch("webbrowser.open", return_value=True):
            self._deliver(open_browser="auto")
        self.assertTrue(self.server.has_viewer())

    def test_a_browser_that_failed_to_launch_is_not_counted(self):
        with unittest.mock.patch("webbrowser.open", return_value=False):
            result = self._deliver(open_browser="auto")
        self.assertFalse(self.server.has_viewer())
        # The decision to open is not the outcome: reporting it as one would
        # claim a tab exists on the one machine where none does.
        self.assertFalse(result["opened_browser"])

    def test_scene_sidecar_section_is_applied_to_the_glb(self):
        """The sidecar carries what FBX can't; sections dispatch by name."""
        self.bridge.deliverer = PreviewDeliverer(server=self.server, open_browser=False)
        sections = {"emissive": {"m": {"color": (1, 0, 0)}}}
        target = "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.fbx_to_glb"
        emissive_target = (
            "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.set_glb_emissive"
        )
        with unittest.mock.patch(target, side_effect=self._fake_convert):
            with unittest.mock.patch(emissive_target, return_value=[{}]) as applied:
                self.bridge.deliverer.deliver(
                    self.bridge,
                    Payload(
                        primary=self._fbx(),
                        extras={"scene_sidecar": _sidecar(sections)},
                    ),
                    HandoffRequest(),
                )
        applied.assert_called_once()
        self.assertEqual(applied.call_args[0][1], sections["emissive"])

    def test_absent_sidecar_leaves_the_glb_untouched(self):
        """Sidecar off must be a true passthrough — that is what makes it a probe."""
        self.bridge.deliverer = PreviewDeliverer(server=self.server, open_browser=False)
        target = "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.fbx_to_glb"
        emissive_target = (
            "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.set_glb_emissive"
        )
        with unittest.mock.patch(target, side_effect=self._fake_convert):
            with unittest.mock.patch(emissive_target) as applied:
                self.bridge.deliverer.deliver(
                    self.bridge, Payload(primary=self._fbx()), HandoffRequest()
                )
        applied.assert_not_called()

    def test_a_failing_sidecar_section_still_publishes(self):
        """A broken section must not cost the whole preview."""
        self.bridge.deliverer = PreviewDeliverer(server=self.server, open_browser=False)
        target = "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.fbx_to_glb"
        emissive_target = (
            "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.set_glb_emissive"
        )
        with unittest.mock.patch(target, side_effect=self._fake_convert):
            with unittest.mock.patch(emissive_target, side_effect=ValueError("bad")):
                result = self.bridge.deliverer.deliver(
                    self.bridge,
                    Payload(
                        primary=self._fbx(),
                        extras={
                            "scene_sidecar": _sidecar(
                                {"emissive": {"m": {"color": (1, 0, 0)}}}
                            )
                        },
                    ),
                    HandoffRequest(),
                )
        self.assertEqual(result["version"], 1)

    def test_sidecar_summary_rides_back_on_the_result(self):
        """The panel needs to say what happened; silence was the original bug."""
        self.bridge.deliverer = PreviewDeliverer(server=self.server, open_browser=False)
        target = "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.fbx_to_glb"
        emissive_target = (
            "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.set_glb_emissive"
        )
        sections = {"emissive": {"a": {"color": (1, 0, 0)}, "b": {"color": (0, 1, 0)}}}
        with unittest.mock.patch(target, side_effect=self._fake_convert):
            with unittest.mock.patch(emissive_target, return_value=[{}, {}]):
                result = self.bridge.deliverer.deliver(
                    self.bridge,
                    Payload(
                        primary=self._fbx(),
                        extras={"scene_sidecar": _sidecar(sections)},
                    ),
                    HandoffRequest(),
                )
        self.assertEqual(result["sidecar"], {"emissive": "2 of 2"})

    def test_sidecar_that_matches_nothing_is_reported_not_silent(self):
        """A name mismatch makes the section a no-op — it must not look like success."""
        self.bridge.deliverer = PreviewDeliverer(server=self.server, open_browser=False)
        target = "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.fbx_to_glb"
        emissive_target = (
            "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.set_glb_emissive"
        )
        with unittest.mock.patch(target, side_effect=self._fake_convert):
            with unittest.mock.patch(emissive_target, return_value=[]):
                result = self.bridge.deliverer.deliver(
                    self.bridge,
                    Payload(
                        primary=self._fbx(),
                        extras={
                            "scene_sidecar": _sidecar(
                                {"emissive": {"gone": {"color": (1, 0, 0)}}}
                            )
                        },
                    ),
                    HandoffRequest(),
                )
        self.assertEqual(result["sidecar"], {"emissive": "0 of 1 matched"})

    def test_summary_separates_off_from_nothing_to_carry(self):
        """Both render the same unlit preview; the report must tell them apart."""
        self.assertIn("off", PreviewBridge.sidecar_summary({"sidecar_requested": False}))
        self.assertIn(
            "nothing to carry",
            PreviewBridge.sidecar_summary({"sidecar_requested": True, "sidecar": {}}),
        )
        self.assertIn(
            "emissive 2 of 3",
            PreviewBridge.sidecar_summary(
                {"sidecar_requested": True, "sidecar": {"emissive": "2 of 3"}}
            ),
        )

    def test_result_records_whether_a_sidecar_was_offered(self):
        """An empty summary alone cannot distinguish 'off' from 'found nothing'."""
        target = "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.fbx_to_glb"
        self.bridge.deliverer = PreviewDeliverer(server=self.server, open_browser=False)
        with unittest.mock.patch(target, side_effect=self._fake_convert):
            off = self.bridge.deliverer.deliver(
                self.bridge, Payload(primary=self._fbx()), HandoffRequest()
            )
            on = self.bridge.deliverer.deliver(
                self.bridge,
                Payload(primary=self._fbx(), extras={"scene_sidecar": {}}),
                HandoffRequest(),
            )
        self.assertFalse(off["sidecar_requested"])
        self.assertTrue(on["sidecar_requested"])

    def test_attach_sidecar_builds_the_versioned_envelope(self):
        """The `.scene.json` top level is a frozen contract for standalone readers.

        A dev tool holding only the FBX and this file parses against exactly
        these keys — which is why they are pinned by name rather than by "some
        dict came through": adding one later is free, but reshaping any of
        these means a schema-version bump.
        """
        bridge = _StubPreviewBridge()
        sections = {"emissive": {"m": {"color": [1, 0, 0]}}}
        fbx = self._fbx()
        payload = bridge._attach_sidecar(
            Payload(primary=fbx),
            sections,
            source={"application": "stub", "version": "0"},
        )

        envelope = payload.extras["scene_sidecar"]
        self.assertEqual(
            set(envelope),
            {"version", "source", "asset", "color_space", "sections", "handoff"},
        )
        self.assertEqual(envelope["version"], MeshConvert.SIDECAR_VERSION)
        self.assertEqual(envelope["source"], {"application": "stub", "version": "0"})
        self.assertEqual(envelope["asset"], Path(fbx).name)
        self.assertEqual(envelope["color_space"], "linear")
        self.assertEqual(envelope["sections"], sections)

        # The written file IS the handoff artifact: it must round-trip to the
        # same envelope the in-process path carries, not a variant of it.
        path = Path(payload.extras["scene_sidecar_path"])
        self.assertEqual(path.suffixes[-2:], [".scene", ".json"])
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), envelope)

    def test_attach_sidecar_with_no_sections_still_attaches_the_envelope(self):
        """Requested-but-empty must stay distinguishable from switched off.

        The producer calls `_attach_sidecar` only when SCENE_SIDECAR is on, so
        key presence in extras is the "was it requested" signal the panel
        summary reads. Skipping the attach on an empty scene collapsed the two
        cases: a user with the checkbox ON was told "Scene sidecar off".
        """
        payload = _StubPreviewBridge()._attach_sidecar(
            Payload(primary=self._fbx()), {}, source={"application": "stub"}
        )
        self.assertEqual(payload.extras["scene_sidecar"]["sections"], {})
        path = Path(payload.extras["scene_sidecar_path"])
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8"))["sections"], {}
        )

    def test_requested_but_empty_scene_reports_nothing_to_carry_not_off(self):
        """The panel must not claim "off" while the user's checkbox is on.

        An unlit, untextured scene produces no sections. Before the fix the
        producer then attached nothing at all, `sidecar_requested` read False,
        and the summary line was "Scene sidecar off - showing what the FBX
        carried" — the exact misreport `sidecar_summary` exists to prevent.
        """
        bridge = _StubPreviewBridge()
        bridge.sections = {}
        bridge.deliverer = PreviewDeliverer(server=self.server, open_browser=False)
        target = "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.fbx_to_glb"
        with unittest.mock.patch(target, side_effect=self._fake_convert):
            result = bridge.send()
        self.assertTrue(result["sidecar_requested"])
        self.assertEqual(result["sidecar"], {})
        self.assertIn("nothing to carry", PreviewBridge.sidecar_summary(result))

    def test_producer_envelope_reaches_the_appliers_unwrapped(self):
        """Producer wraps, deliverer unwraps — the pair contract, end to end.

        Everything between `_attach_sidecar` and the applier call is real
        here; only the converter and the GLB writer are stubbed. A schema
        drift between the two halves fails this test rather than surfacing as
        a silently unlit preview in a live DCC.
        """
        bridge = _StubPreviewBridge()
        bridge.deliverer = PreviewDeliverer(server=self.server, open_browser=False)
        target = "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.fbx_to_glb"
        emissive_target = (
            "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.set_glb_emissive"
        )
        with unittest.mock.patch(target, side_effect=self._fake_convert):
            with unittest.mock.patch(emissive_target, return_value=[{}]) as applied:
                result = bridge.send()
        applied.assert_called_once()
        self.assertEqual(applied.call_args[0][1], _StubPreviewBridge.sections["emissive"])
        self.assertEqual(result["sidecar"], {"emissive": "1 of 1"})
        self.assertTrue(result["sidecar_requested"])

    def test_envelope_is_embedded_in_the_published_glb(self):
        """The published artifact is self-describing — envelope + outcome inside.

        No applier is patched here: the fake GLB has no materials, so the
        emissive section reports "0 of 1 matched" — and both the envelope and
        that summary must still be readable back out of the served file with
        no side files, which is the whole embed contract.
        """
        bridge = _StubPreviewBridge()
        bridge.deliverer = PreviewDeliverer(server=self.server, open_browser=False)
        target = "pythontk.file_utils.mesh_convert._mesh_convert.MeshConvert.fbx_to_glb"
        with unittest.mock.patch(target, side_effect=self._fake_convert):
            result = bridge.send()
        served = Path(self.server.root) / result["asset"]
        embedded = MeshConvert.read_scene_sidecar(str(served))
        self.assertEqual(embedded["sections"], _StubPreviewBridge.sections)
        self.assertEqual(embedded["version"], MeshConvert.SIDECAR_VERSION)
        with MeshConvert.open_glb(str(served)) as edit:
            self.assertEqual(
                edit.gltf["extras"]["scene_sidecar_applied"],
                {"emissive": "0 of 1 matched"},
            )
        self.assertEqual(result["sidecar"], {"emissive": "0 of 1 matched"})

    def test_ensure_server_creates_one_lazily_and_reuses_it(self):
        deliverer = PreviewDeliverer(title="Lazy")
        self.assertIsNone(deliverer.server)
        try:
            first = deliverer.ensure_server()
            self.assertTrue(first.is_running)
            self.assertIs(deliverer.ensure_server(), first)
        finally:
            if deliverer.server is not None:
                deliverer.server.stop()


class SetGlbEmissiveTestCase(unittest.TestCase):
    """The GLB-side repair for the emissive channel Maya's FBX exporter drops."""

    def setUp(self):
        self.temp = TempArtifacts("test_glb_emissive", policy="scoped")
        self.glb = Path(self.temp.path(extension=".glb"))

    def tearDown(self):
        self.temp.cleanup()

    def _write_glb(self, gltf, bin_chunk=b"\x01\x02\x03\x04"):
        payload = json.dumps(gltf).encode("utf-8")
        payload += b" " * ((4 - (len(payload) % 4)) % 4)
        rest = struct.pack("<I", len(bin_chunk)) + b"BIN\x00" + bin_chunk
        with open(self.glb, "wb") as f:
            f.write(b"glTF")
            f.write(struct.pack("<I", 2))
            f.write(struct.pack("<I", 12 + 8 + len(payload) + len(rest)))
            f.write(struct.pack("<I", len(payload)))
            f.write(b"JSON")
            f.write(payload)
            f.write(rest)
        return self.glb

    def _read_back(self):
        edit = MeshConvert._read_glb(str(self.glb))
        return edit.gltf, edit.rest, edit.bin_data

    def _png(self, name="emis.png"):
        """A minimal valid 1x1 PNG on disk."""
        path = Path(self.temp.dir_path()) / name
        path.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z"
                "8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
            )
        )
        return str(path)

    def test_writes_emissive_factor_by_material_name(self):
        self._write_glb({"materials": [{"name": "m_arnold"}]})
        records = MeshConvert.set_glb_emissive(str(self.glb), {"m_arnold": {"color": (1, 0.5, 0)}})
        self.assertEqual(len(records), 1)
        gltf, _rest, _bin = self._read_back()
        self.assertEqual(gltf["materials"][0]["emissiveFactor"], [1, 0.5, 0])

    def test_bin_chunk_survives_the_edit(self):
        """The JSON chunk is rewritten; geometry must come back byte-identical."""
        self._write_glb({"materials": [{"name": "m"}]}, bin_chunk=b"GEOMETRYDATA1234")
        MeshConvert.set_glb_emissive(str(self.glb), {"m": {"color": (1, 1, 1)}})
        _gltf, _rest, bin_data = self._read_back()
        self.assertEqual(bin_data, b"GEOMETRYDATA1234")

    def test_over_unit_emission_becomes_strength_not_clipping(self):
        """A Maya emission of 5.0 must not flatten to 1.0."""
        self._write_glb({"materials": [{"name": "m"}]})
        MeshConvert.set_glb_emissive(str(self.glb), {"m": {"color": (5.0, 2.5, 0.0)}})
        gltf, _rest, _bin = self._read_back()
        mat = gltf["materials"][0]
        self.assertEqual(mat["emissiveFactor"], [1.0, 0.5, 0.0])
        ext = mat["extensions"]["KHR_materials_emissive_strength"]
        self.assertEqual(ext["emissiveStrength"], 5.0)
        self.assertIn("KHR_materials_emissive_strength", gltf["extensionsUsed"])

    def test_sub_unit_emission_declares_no_extension(self):
        self._write_glb({"materials": [{"name": "m"}]})
        MeshConvert.set_glb_emissive(str(self.glb), {"m": {"color": (0.5, 0.5, 0.5)}})
        gltf, _rest, _bin = self._read_back()
        self.assertNotIn("extensions", gltf["materials"][0])
        self.assertNotIn("extensionsUsed", gltf)

    def test_texture_is_embedded_as_a_data_uri(self):
        self._write_glb({"materials": [{"name": "m"}]})
        records = MeshConvert.set_glb_emissive(str(self.glb), {"m": {"texture": self._png()}})
        self.assertEqual(len(records), 1)
        gltf, _rest, _bin = self._read_back()
        self.assertEqual(gltf["materials"][0]["emissiveTexture"]["index"], 0)
        self.assertTrue(gltf["images"][0]["uri"].startswith("data:image/png;base64,"))
        # A texture with no color must still light up, not stay black.
        self.assertEqual(gltf["materials"][0]["emissiveFactor"], [1.0, 1.0, 1.0])

    def test_repeated_texture_path_is_embedded_once(self):
        self._write_glb({"materials": [{"name": "a"}, {"name": "b"}]})
        png = self._png()
        MeshConvert.set_glb_emissive(
            str(self.glb), {"a": {"texture": png}, "b": {"texture": png}}
        )
        gltf, _rest, _bin = self._read_back()
        self.assertEqual(len(gltf["images"]), 1)
        self.assertEqual(len(gltf["textures"]), 1)
        self.assertEqual(gltf["materials"][0]["emissiveTexture"]["index"], 0)
        self.assertEqual(gltf["materials"][1]["emissiveTexture"]["index"], 0)

    def test_unreadable_image_is_skipped_not_embedded(self):
        """A file no decoder can read must be reported, not written as a data URI.

        Junk bytes under a TIFF extension: the Pillow fallback tries and fails
        to decode it, so it takes the same rejected path as a format with no
        decoder at all (EXR, or no Pillow installed).
        """
        self._write_glb({"materials": [{"name": "m"}]})
        tif = Path(self.temp.dir_path()) / "e.tif"
        tif.write_bytes(b"II*\x00")
        MeshConvert.set_glb_emissive(str(self.glb), {"m": {"texture": str(tif)}})
        gltf, _rest, _bin = self._read_back()
        self.assertNotIn("emissiveTexture", gltf["materials"][0])
        self.assertNotIn("images", gltf)

    def test_tga_texture_is_reencoded_to_png(self):
        """TGA is everywhere in game art, and glTF cannot hold it natively.

        Rejecting it by extension silently downgraded the material to
        colour-only; with Pillow present it must arrive as an embedded PNG.
        """
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        self._write_glb({"materials": [{"name": "m"}]})
        tga = Path(self.temp.dir_path()) / "emis.tga"
        Image.new("RGB", (2, 2), (255, 0, 0)).save(tga)
        records = MeshConvert.set_glb_emissive(str(self.glb), {"m": {"texture": str(tga)}})
        self.assertEqual(len(records), 1)
        gltf, _rest, _bin = self._read_back()
        uri = gltf["images"][0]["uri"]
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        # The payload really is a PNG, not TGA bytes under a PNG label.
        decoded = base64.b64decode(uri.split(",", 1)[1])
        self.assertTrue(decoded.startswith(b"\x89PNG"))

    def test_zero_factor_never_blacks_out_an_emissive_map(self):
        """glTF emission is factor x texture — a 0 factor silently kills the map.

        The DCC readers now omit a material whose emission weight is 0, so this
        pins the writer's half: a texture with an explicit all-zero colour must
        not be written as a black-multiplied map.
        """
        self._write_glb({"materials": [{"name": "m"}]})
        records = MeshConvert.set_glb_emissive(
            str(self.glb), {"m": {"texture": self._png(), "color": (0, 0, 0)}}
        )
        self.assertEqual(records, [])
        gltf, _rest, _bin = self._read_back()
        mat = gltf["materials"][0]
        self.assertNotIn("emissiveTexture", mat)
        self.assertNotIn("emissiveFactor", mat)
        # Skipped before embedding — no orphaned image left in the file.
        self.assertNotIn("images", gltf)

    def test_unknown_material_name_is_ignored(self):
        self._write_glb({"materials": [{"name": "present"}]})
        self.assertEqual(
            MeshConvert.set_glb_emissive(str(self.glb), {"absent": {"color": (1, 1, 1)}}),
            [],
        )

    def test_empty_mapping_is_a_no_op(self):
        self._write_glb({"materials": [{"name": "m"}]})
        before = self.glb.read_bytes()
        self.assertEqual(MeshConvert.set_glb_emissive(str(self.glb), {}), [])
        self.assertEqual(self.glb.read_bytes(), before)

    def test_base_color_factor_is_written_preserving_alpha(self):
        """Colour must not silently turn a transparent material opaque."""
        self._write_glb(
            {
                "materials": [
                    {
                        "name": "m",
                        "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 0.25]},
                    }
                ]
            }
        )
        MeshConvert.set_glb_base_color(str(self.glb), {"m": {"color": (0.2, 0.4, 0.8)}})
        gltf, _rest, _bin = self._read_back()
        self.assertEqual(
            gltf["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"],
            [0.2, 0.4, 0.8, 0.25],
        )

    def test_base_color_black_is_written_not_skipped(self):
        """Unlike emissive, black is a legitimate base colour."""
        self._write_glb({"materials": [{"name": "m"}]})
        records = MeshConvert.set_glb_base_color(str(self.glb), {"m": {"color": (0, 0, 0)}})
        self.assertEqual(len(records), 1)
        gltf, _rest, _bin = self._read_back()
        self.assertEqual(
            gltf["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"],
            [0.0, 0.0, 0.0, 1.0],
        )

    def test_base_color_is_clamped_not_normalized(self):
        """There is no strength extension for base colour — clamp into range."""
        self._write_glb({"materials": [{"name": "m"}]})
        MeshConvert.set_glb_base_color(str(self.glb), {"m": {"color": (2.0, -1.0, 0.5)}})
        gltf, _rest, _bin = self._read_back()
        self.assertEqual(
            gltf["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"],
            [1.0, 0.0, 0.5, 1.0],
        )
        self.assertNotIn("extensions", gltf["materials"][0])

    def test_base_color_texture_embeds(self):
        self._write_glb({"materials": [{"name": "m"}]})
        MeshConvert.set_glb_base_color(str(self.glb), {"m": {"texture": self._png()}})
        gltf, _rest, _bin = self._read_back()
        pbr = gltf["materials"][0]["pbrMetallicRoughness"]
        self.assertEqual(pbr["baseColorTexture"]["index"], 0)
        self.assertTrue(gltf["images"][0]["uri"].startswith("data:image/png;base64,"))

    def test_base_color_unknown_material_is_ignored(self):
        self._write_glb({"materials": [{"name": "present"}]})
        self.assertEqual(
            MeshConvert.set_glb_base_color(str(self.glb), {"absent": {"color": (1, 0, 0)}}),
            [],
        )

    def test_non_glb_input_raises(self):
        self.glb.write_bytes(b"not a glb at all")
        with self.assertRaises(ValueError):
            MeshConvert.set_glb_emissive(str(self.glb), {"m": {"color": (1, 1, 1)}})


if __name__ == "__main__":
    unittest.main()
