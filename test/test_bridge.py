# !/usr/bin/python
# coding=utf-8
"""Tests for the Qt-free / DCC-free app hand-off engine (pythontk.core_utils).

Covers the shared machinery the per-DCC bridge engines previously duplicated:
generic script-template discovery / mode parsing / ``__KEY__`` substitution
(``core_utils.script_template``), executable resolution
(``AppLauncher.resolve_app_path``), and the Template-Method + Strategy orchestration
(``core_utils.app_handoff``). No DCC runtime required.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock
from dataclasses import replace
from pathlib import Path

from pythontk.core_utils import script_template
from pythontk.core_utils.app_launcher import AppLauncher
from pythontk.core_utils import app_handoff
from pythontk.core_utils.app_handoff import (
    AppSpec,
    HandoffBridge,
    HandoffRequest,
    Payload,
    ScriptLaunchBridge,
    ScriptLaunchSpec,
    ROUND_TRIP,
    SAVE_AS,
    SEND_TO,
)
from pythontk.core_utils.script_run import ScriptRunResult


class TemplatesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, body=""):
        path = self.tmp / name
        path.write_text(body, encoding="utf-8")
        return path

    def test_list_templates_skips_underscore(self):
        self._write("import.py")
        self._write("replace_scene.py")
        self._write("_helper.py")  # underscore-prefixed -> hidden
        stems = [
            p.stem for p in script_template.ScriptTemplate.list_templates(self.tmp)
        ]
        self.assertEqual(stems, ["import", "replace_scene"])

    def test_list_templates_honors_extension(self):
        self._write("load.lua")
        self._write("ignore.py")
        stems = [
            p.stem
            for p in script_template.ScriptTemplate.list_templates(self.tmp, ".lua")
        ]
        self.assertEqual(stems, ["load"])

    def test_template_modes_declared(self):
        path = self._write("a.py", "BRIDGE_MODES = ('send_to',)\n")
        self.assertEqual(
            script_template.ScriptTemplate.template_modes(path), ("send_to",)
        )

    def test_template_modes_fallback_when_absent(self):
        path = self._write("a.py", "x = 1\n")
        self.assertEqual(
            script_template.ScriptTemplate.template_modes(path), (SEND_TO,)
        )

    def test_template_modes_filters_unknown(self):
        path = self._write("a.py", "BRIDGE_MODES = ('send_to', 'bogus')\n")
        # 'bogus' isn't in the allowed set -> dropped; only 'send_to' survives.
        self.assertEqual(
            script_template.ScriptTemplate.template_modes(path), ("send_to",)
        )

    def test_declared_modes_reports_the_raw_declaration(self):
        """The strict read: what the template SAYS, with no assumed mode."""
        path = self._write("a.py", "BRIDGE_MODES = ('send_to', 'bogus')\n")
        self.assertEqual(
            script_template.ScriptTemplate.declared_modes(path), ("send_to", "bogus")
        )

    def test_declared_modes_folds_legacy_spellings_to_the_canon(self):
        """A template written against the old ``roundtrip`` still loads.

        The mode tuple is an ON-DISK contract, so unifying the vocabulary in code would
        otherwise strand every template (including user-authored ones) that spells it the
        old way -- and strand them SILENTLY: an unrecognized mode does not raise,
        ``template_modes`` filters it out and falls back to the primary mode, turning a
        headless round trip into an interactive send with nothing naming the cause.
        """
        path = self._write("legacy.py", "BRIDGE_MODES = ('send_to', 'roundtrip')\n")
        self.assertEqual(
            script_template.ScriptTemplate.declared_modes(path),
            (SEND_TO, ROUND_TRIP),
        )
        # ...and it survives the filtered read too, rather than being dropped.
        self.assertEqual(
            script_template.ScriptTemplate.template_modes(
                path, (SEND_TO, ROUND_TRIP)
            ),
            (SEND_TO, ROUND_TRIP),
        )

    def test_declared_values_never_applies_the_mode_aliases(self):
        """The generic reader must not interpret -- other fields ride it too.

        Templates declare more than modes through the same ``<FIELD> = (...)``
        convention (``BRIDGE_OUTPUT_EXT`` / ``BRIDGE_TIMEOUT`` / ...), so folding mode
        spellings in the shared reader would silently rewrite any other field whose
        value collided with a legacy spelling.
        """
        path = self._write(
            "a.py", "BRIDGE_MODES = ('roundtrip',)\nBRIDGE_TAG = ('roundtrip',)\n"
        )
        self.assertEqual(
            script_template.ScriptTemplate.declared_values(path, "BRIDGE_TAG"),
            ("roundtrip",),
        )
        self.assertEqual(
            script_template.ScriptTemplate.declared_values(path, "BRIDGE_MODES"),
            ("roundtrip",),
        )
        # Only the mode-flavoured reader folds it.
        self.assertEqual(
            script_template.ScriptTemplate.declared_modes(path), (ROUND_TRIP,)
        )

    def test_declared_values_is_none_when_unannotated(self):
        """``None`` (not ``()``) has to survive the split from ``declared_modes``."""
        self.assertIsNone(
            script_template.ScriptTemplate.declared_values(
                self._write("a.py", "x = 1\n"), "BRIDGE_MODES"
            )
        )
        self.assertIsNone(
            script_template.ScriptTemplate.declared_values(
                self.tmp / "nope.py", "BRIDGE_MODES"
            )
        )

    def test_declared_modes_is_none_when_unannotated(self):
        """``None`` (not ``()``) marks "declares nothing" -- what makes the lenient
        fallback safe to apply only where it belongs."""
        self.assertIsNone(
            script_template.ScriptTemplate.declared_modes(self._write("a.py", "x = 1\n"))
        )
        self.assertIsNone(
            script_template.ScriptTemplate.declared_modes(self.tmp / "nope.py")
        )

    def test_template_modes_missing_file_fallback(self):
        self.assertEqual(
            script_template.ScriptTemplate.template_modes(
                self.tmp / "nope.py", ("send_to",)
            ),
            ("send_to",),
        )

    def test_template_modes_custom_field(self):
        path = self._write("a.py", "MODES = ('send_to',)\n")
        # Default field name finds nothing -> fallback.
        self.assertEqual(
            script_template.ScriptTemplate.template_modes(path), (SEND_TO,)
        )
        # Custom field name reads the declaration.
        self.assertEqual(
            script_template.ScriptTemplate.template_modes(path, field="MODES"),
            ("send_to",),
        )

    def test_list_template_modes_pairs(self):
        self._write("import.py", "BRIDGE_MODES = ('send_to',)\n")
        self._write("frame.py")  # no declaration -> fallback mode
        self.assertEqual(
            script_template.ScriptTemplate.list_template_modes(self.tmp),
            [("frame", "send_to"), ("import", "send_to")],
        )

    def test_render_template_substitutes(self):
        path = self._write("t.py", 'FBX = r"__FBX_PATH__"\nN = __COUNT__\n')
        out = script_template.ScriptTemplate.render_template(
            path, {"FBX_PATH": "C:/x.fbx", "COUNT": "3"}
        )
        self.assertIn('FBX = r"C:/x.fbx"', out)
        self.assertIn("N = 3", out)


class AppScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("BRIDGE_TEST_EXE", None)
        os.environ.pop("BRIDGE_TEST_LOC", None)

    def test_env_var_hit(self):
        exe = self.tmp / "app.exe"
        exe.write_text("", encoding="utf-8")
        os.environ["BRIDGE_TEST_EXE"] = str(exe)
        self.assertEqual(
            AppLauncher.resolve_app_path(env_vars=("BRIDGE_TEST_EXE",)), str(exe)
        )

    def test_location_env_var_hit(self):
        (self.tmp / "bin").mkdir()
        exe = self.tmp / "bin" / "maya.exe"
        exe.write_text("", encoding="utf-8")
        os.environ["BRIDGE_TEST_LOC"] = str(self.tmp)
        got = AppLauncher.resolve_app_path(
            location_env_vars=(("BRIDGE_TEST_LOC", ("bin", "maya.exe")),)
        )
        self.assertEqual(got, str(exe))

    def test_scan_glob_newest_wins(self):
        for v in ("App 4.0", "App 5.1", "App 4.2"):
            d = self.tmp / v
            d.mkdir()
            (d / "app.exe").write_text("", encoding="utf-8")
        glob_pat = str(self.tmp / "App *" / "app.exe")
        got = AppLauncher.resolve_app_path(scan_globs=(glob_pat,))
        self.assertEqual(got, str(self.tmp / "App 5.1" / "app.exe"))

    def test_scan_glob_pattern_order_is_priority(self):
        """The first pattern's matches outrank later patterns' matches.

        Regression: the pooled reverse-sort made the FILENAME an accidental
        tiebreaker inside one install dir -- RizomUV's ``rizomuv_RS.exe``
        outranked the intended ``Rizomuv_VS.exe`` by ASCII order, inverting
        the caller's tuple priority (lowercase sorts above uppercase).
        """
        d = self.tmp / "Rizom Lab" / "RizomUV 2020.1"
        d.mkdir(parents=True)
        for name in ("Rizomuv_VS.exe", "rizomuv_RS.exe", "rizomuv.exe"):
            (d / name).write_text("", encoding="utf-8")
        base = self.tmp / "Rizom Lab" / "*"
        got = AppLauncher.resolve_app_path(
            scan_globs=(
                str(base / "Rizomuv_VS.exe"),
                str(base / "rizomuv_RS.exe"),
                str(base / "rizomuv.exe"),
            )
        )
        self.assertEqual(got, str(d / "Rizomuv_VS.exe"))

    def test_scan_glob_priority_still_prefers_newest_within_pattern(self):
        """Within one pattern, newer install dirs still win."""
        for v in ("Tool 2020.1", "Tool 2022.2"):
            d = self.tmp / v
            d.mkdir()
            (d / "primary.exe").write_text("", encoding="utf-8")
        (self.tmp / "Tool 2020.1" / "fallback.exe").write_text("", encoding="utf-8")
        got = AppLauncher.resolve_app_path(
            scan_globs=(
                str(self.tmp / "Tool *" / "primary.exe"),
                str(self.tmp / "Tool *" / "fallback.exe"),
            )
        )
        self.assertEqual(got, str(self.tmp / "Tool 2022.2" / "primary.exe"))

    def test_returns_none_when_nothing_resolves(self):
        self.assertIsNone(
            AppLauncher.resolve_app_path(
                scan_globs=(str(self.tmp / "missing" / "*.exe"),)
            )
        )

    def test_app_spec_resolve(self):
        exe = self.tmp / "app.exe"
        exe.write_text("", encoding="utf-8")
        os.environ["BRIDGE_TEST_EXE"] = str(exe)
        spec = AppSpec(name="App", env_vars=("BRIDGE_TEST_EXE",))
        self.assertEqual(spec.resolve(), str(exe))
        self.assertEqual(spec.not_found_message, "App executable not found.")

    def test_app_spec_path_is_cached(self):
        """``path`` resolves once; the discovery scan is not repeated.

        The visibility gate asks every panel build whether an app is installed,
        so an uncached probe would re-glob Program Files on each panel show
        (12-155ms measured per app). ``resolve()`` stays uncached for callers
        that genuinely want a fresh look.
        """
        exe = self.tmp / "cached.exe"
        exe.write_text("", encoding="utf-8")
        os.environ["BRIDGE_CACHE_EXE"] = str(exe)
        spec = AppSpec(name="Cached", env_vars=("BRIDGE_CACHE_EXE",))

        calls = []
        real = AppLauncher.resolve_app_path

        def counting(**kwargs):
            calls.append(kwargs)
            return real(**kwargs)

        with mock.patch.object(AppLauncher, "resolve_app_path", counting):
            self.assertEqual(spec.path, str(exe))
            self.assertEqual(spec.path, str(exe))
            self.assertTrue(spec.available)
        self.assertEqual(len(calls), 1, "path must resolve exactly once")

    def test_app_spec_available_is_false_when_missing(self):
        """A spec that resolves to nothing reads as unavailable (and caches that)."""
        spec = AppSpec(
            name="Ghost", scan_globs=(str(self.tmp / "nope" / "*.exe"),)
        )
        self.assertFalse(spec.available)
        self.assertIsNone(spec.path)

    def test_app_spec_refresh_clears_the_cache(self):
        """``refresh()`` re-probes -- an app installed mid-session becomes visible."""
        target = self.tmp / "late.exe"
        os.environ["BRIDGE_LATE_EXE"] = str(target)
        spec = AppSpec(name="Late", env_vars=("BRIDGE_LATE_EXE",))
        self.assertFalse(spec.available)

        target.write_text("", encoding="utf-8")
        self.assertFalse(spec.available, "cached miss must persist until refresh")
        spec.refresh()
        self.assertTrue(spec.available)
        self.assertEqual(spec.path, str(target))

    def test_app_spec_cache_is_per_instance_not_shared(self):
        """Two specs cache independently (they are separate module singletons)."""
        exe = self.tmp / "one.exe"
        exe.write_text("", encoding="utf-8")
        os.environ["BRIDGE_ONE_EXE"] = str(exe)
        found = AppSpec(name="One", env_vars=("BRIDGE_ONE_EXE",))
        missing = AppSpec(name="Two", scan_globs=(str(self.tmp / "x" / "*.exe"),))
        self.assertTrue(found.available)
        self.assertFalse(missing.available)
        self.assertTrue(found.available)

    def test_app_spec_stays_hashable_and_frozen(self):
        """Caching must not break the frozen-dataclass contract."""
        spec = AppSpec(name="Frozen", app_names=("nothing-here",))
        self.assertFalse(spec.available)  # populate the cache
        hash(spec)  # must not raise
        with self.assertRaises(Exception):
            spec.name = "mutated"


class _StubScriptBridge(ScriptLaunchBridge):
    """A ScriptLaunchBridge wired to fakes so send() runs with no DCC/launch."""

    def __init__(self, template_dir, launched, **kw):
        # Build the dataclass spec (instance attr shadows the None class default);
        # the stub app always "resolves" via app_path below.
        self.spec = ScriptLaunchSpec(
            app=AppSpec(name="StubApp"),
            template_dir=Path(template_dir),
            launch_args=lambda script: ["--run", script],
            payload_prefix="stub_to_app",
        )
        super().__init__(**kw)
        self._launched = launched  # list mutated by the patched launcher
        self.exported = []
        # Pretend an app is always installed.
        self.app_path = "C:/fake/stubapp.exe"

    def params_defaults(self):
        return {"SCALE": 1.0}

    def render_context(self, params):
        return {k: repr(v) for k, v in params.items()}

    def _resolve_objects(self, objects):
        return objects if objects is not None else ["objA", "objB"]

    def _export_fbx(self, objects, fbx_path, params):
        Path(fbx_path).write_text("fbx", encoding="utf-8")
        self.exported.append((tuple(objects), fbx_path, dict(params)))

    def _produce(self, objects, request):
        path = self._make_payload_path()
        self._export_fbx(objects, path, request.params)
        return Payload(primary=path)


class HandoffSendTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "import.py").write_text(
            "BRIDGE_MODES = ('send_to',)\nFBX = r\"__FBX_PATH__\"\nSCALE = __SCALE__\n",
            encoding="utf-8",
        )
        self.launched = []
        # Patch the AppLauncher.launch the deliverer calls.
        self._orig_launch = app_handoff.AppLauncher.launch

        def _fake_launch(app, args=None, detached=True, **kw):
            self.launched.append((app, list(args or []), detached))
            return object()  # truthy "process"

        app_handoff.AppLauncher.launch = staticmethod(_fake_launch)

    def tearDown(self):
        app_handoff.AppLauncher.launch = staticmethod(self._orig_launch)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _bridge(self):
        return _StubScriptBridge(self.tmp, self.launched)

    def test_launch_env_hook_shapes_child_env_and_never_costs_the_send(self):
        # The hook's env reaches the launcher...
        seen = {}

        def _capturing_launch(app, args=None, detached=True, env=None, **kw):
            seen["env"] = env
            return object()

        app_handoff.AppLauncher.launch = staticmethod(_capturing_launch)
        br = self._bridge()
        # The deliverer holds the spec (wired at __init__) -- swap it there.
        br.deliverer.spec = replace(br.deliverer.spec, launch_env=lambda: {"CLEAN": "1"})
        self.assertIsNotNone(br.send(template="import", mode=SEND_TO))
        self.assertEqual(seen["env"], {"CLEAN": "1"})
        # ...and a RAISING hook degrades to the inherited env instead of
        # killing the send (sanitizing is best-effort by contract).
        br.deliverer.spec = replace(br.deliverer.spec, launch_env=lambda: 1 / 0)
        self.assertIsNotNone(br.send(template="import", mode=SEND_TO))
        self.assertIsNone(seen["env"])

    def test_send_renders_writes_and_launches(self):
        br = self._bridge()
        result = br.send(objects=["a", "b", "c"], template="import", mode=SEND_TO)
        self.assertIsNotNone(result)
        self.assertEqual(result["template"], "import")
        self.assertEqual(result["mode"], SEND_TO)
        # Script written next to the payload with the template extension.
        script = Path(result["script"])
        self.assertTrue(script.is_file())
        self.assertTrue(script.name.endswith(".py"))
        body = script.read_text(encoding="utf-8")
        self.assertIn(result["payload"].replace("\\", "/"), body)
        self.assertIn("SCALE = 1.0", body)  # default merged + rendered
        # Launched once with our argv.
        self.assertEqual(len(self.launched), 1)
        _app, args, detached = self.launched[0]
        self.assertEqual(args, ["--run", str(script)])
        self.assertTrue(detached)

    def test_user_params_override_defaults(self):
        br = self._bridge()
        result = br.send(template="import", mode=SEND_TO, params={"SCALE": 2.5})
        body = Path(result["script"]).read_text(encoding="utf-8")
        self.assertIn("SCALE = 2.5", body)

    def test_unknown_mode_aborts_before_export(self):
        br = self._bridge()
        result = br.send(template="import", mode="round_trip")
        self.assertIsNone(result)
        self.assertEqual(br.exported, [])  # never exported
        self.assertEqual(self.launched, [])  # never launched

    def test_missing_template_aborts(self):
        br = self._bridge()
        result = br.send(template="does_not_exist", mode=SEND_TO)
        self.assertIsNone(result)
        self.assertEqual(self.launched, [])

    def test_empty_selection_aborts(self):
        br = self._bridge()
        # _resolve_objects returns [] for an explicit empty list.
        result = br.send(objects=[], template="import", mode=SEND_TO)
        # [] is falsy -> resolved is [] -> "No valid objects" abort.
        self.assertIsNone(result)

    def test_export_failure_returns_none(self):
        br = self._bridge()

        def _boom(objects, fbx_path, params):
            raise RuntimeError("export blew up")

        br._export_fbx = _boom
        result = br.send(template="import", mode=SEND_TO)
        self.assertIsNone(result)
        self.assertEqual(self.launched, [])

    def test_missing_app_aborts(self):
        br = self._bridge()
        br.app_path = None
        result = br.send(template="import", mode=SEND_TO)
        self.assertIsNone(result)
        self.assertEqual(self.launched, [])


class HandoffContractTest(unittest.TestCase):
    """The bare base leaves the polymorphic steps abstract."""

    def test_resolve_objects_is_abstract(self):
        br = HandoffBridge()
        with self.assertRaises(NotImplementedError):
            br._resolve_objects(None)

    def test_produce_is_abstract(self):
        br = HandoffBridge()
        with self.assertRaises(NotImplementedError):
            br._produce(["a"], HandoffRequest())

    def test_deliver_without_strategy_raises(self):
        br = HandoffBridge()
        with self.assertRaises(NotImplementedError):
            br._deliver(Payload(primary="x.fbx"), HandoffRequest())

    def test_scene_objects_defaults_to_unsupported(self):
        """``None`` = "this host can't enumerate itself" -> save_as uses the selection."""
        self.assertIsNone(HandoffBridge()._scene_objects())


class RunScratchTest(unittest.TestCase):
    """The run-scratch lifetime: who is allowed to delete what a run staged.

    A hand-off the target app reads AFTER we return has no safe delete
    (``detached``); a blocking run that CONSUMES what it staged does
    (``scoped``), and takes it away in ``_ingest``.  Added: 2026-08-18
    """

    def _bridge(self, policy="detached", delivered=()):
        class _B(HandoffBridge):
            payload_prefix = "ptk_scratch_test"

            def _scratch_policy(self, request):
                return policy

            def _delivered_paths(self, result):
                return list(delivered)

        b = _B()
        b.logger.setLevel("CRITICAL")  # keep the suite output clean
        return b

    def test_detached_run_reuses_one_named_folder_and_never_deletes(self):
        br, req = self._bridge("detached"), HandoffRequest()
        first = br._scratch_dir(req, "handoff")
        self.assertTrue(os.path.isdir(first))
        self.addCleanup(shutil.rmtree, first, True)
        # A fixed name is what keeps a never-deleted folder from growing a
        # generation per send.
        self.assertEqual(br._scratch_dir(HandoffRequest(), "handoff"), first)
        br._discard_scratch(req, {})
        self.assertTrue(os.path.isdir(first), "a detached hand-off must outlive us")

    def test_scoped_run_shares_one_root_and_ingest_removes_it(self):
        br, req = self._bridge("scoped"), HandoffRequest()
        work = br._scratch_dir(req, "handoff")
        stage = br._scratch_dir(req, "staging")
        root = os.path.dirname(work)
        self.addCleanup(shutil.rmtree, root, True)
        self.assertEqual(os.path.dirname(stage), root, "one root per run")
        # Unique per run, so two hosts cannot share -- or delete -- one dir.
        other = br._scratch_dir(HandoffRequest(), "handoff")
        self.addCleanup(shutil.rmtree, os.path.dirname(other), True)
        self.assertNotEqual(os.path.dirname(other), root)

        br._discard_scratch(req, {"ok": True})
        self.assertFalse(os.path.exists(root), "a clean scoped run removes its scratch")

    def test_scoped_scratch_holding_the_output_is_kept(self):
        req = HandoffRequest()
        br = self._bridge("scoped")
        work = br._scratch_dir(req, "handoff")
        root = os.path.dirname(work)
        self.addCleanup(shutil.rmtree, root, True)
        # The guard: the run's real output landed inside the scratch, so the
        # scratch holds the only copy.
        br._delivered_paths = lambda result: [os.path.join(work, "map.png")]
        br._discard_scratch(req, {})
        self.assertTrue(os.path.isdir(root))

    def test_a_failed_run_keeps_its_scratch(self):
        """``_ingest`` is never reached, so the scratch stays inspectable."""
        br, req = self._bridge("scoped"), HandoffRequest()
        work = br._scratch_dir(req, "handoff")
        root = os.path.dirname(work)
        self.addCleanup(shutil.rmtree, root, True)
        self.assertTrue(os.path.isdir(root))

    def test_discard_is_idempotent(self):
        br, req = self._bridge("scoped"), HandoffRequest()
        root = os.path.dirname(br._scratch_dir(req, "handoff"))
        self.addCleanup(shutil.rmtree, root, True)
        br._discard_scratch(req, {})
        br._discard_scratch(req, {})  # must not raise on the second pass
        self.assertFalse(os.path.exists(root))

    def test_default_policy_is_detached(self):
        """The safe default: nothing deletes a payload another app may be reading."""
        self.assertEqual(HandoffBridge()._scratch_policy(HandoffRequest()), "detached")

    def test_policy_reads_the_declared_modes(self):
        """The ordinary opt-in is one line of DATA, not an overridden method."""

        class _B(HandoffBridge):
            scoped_scratch_modes = (ROUND_TRIP,)

        br = _B()
        self.assertEqual(br._scratch_policy(HandoffRequest(mode=ROUND_TRIP)), "scoped")
        self.assertEqual(br._scratch_policy(HandoffRequest(mode=SEND_TO)), "detached")

    def _staging_bridge(self, ingest=None):
        """A minimal end-to-end bridge that stages a scoped scratch in ``_produce``."""
        seen = {}

        class _B(HandoffBridge):
            payload_prefix = "ptk_scratch_run"
            requires_objects = False

            def _scratch_policy(self, request):
                return "scoped"

            def _resolve_objects(self, objects):
                return list(objects or ["x"])

            def _produce(self, objects, request):
                seen["root"] = os.path.dirname(self._scratch_dir(request, "handoff"))
                return Payload(primary="p")

            def _deliver(self, payload, request):
                return {"ok": True}

        if ingest is not None:
            _B._ingest = ingest
        b = _B()
        b.logger.setLevel("CRITICAL")
        return b, seen

    def test_the_skeleton_discards_even_when_ingest_is_overridden(self):
        """The invariant is in ``_run``, so a return leg that skips ``super()``
        cannot leak the scratch -- ``blender_bridge`` is exactly such a leg."""
        br, seen = self._staging_bridge(
            ingest=lambda self, result, objects, payload, request: result
        )
        self.assertIsNotNone(br.send())
        self.addCleanup(shutil.rmtree, seen["root"], True)
        self.assertFalse(os.path.exists(seen["root"]))

    def test_a_return_leg_reporting_failure_keeps_the_scratch(self):
        """``_ingest`` -> ``None`` is a handled failure; leave it inspectable."""
        br, seen = self._staging_bridge(
            ingest=lambda self, result, objects, payload, request: None
        )
        self.assertIsNone(br.send())
        self.addCleanup(shutil.rmtree, seen["root"], True)
        self.assertTrue(os.path.isdir(seen["root"]))


class _StubSaveBridge(_StubScriptBridge):
    """A stub bridge with BOTH routes wired: detached send + blocking save_as."""

    save_extensions = (".stub", ".stubb")

    def __init__(self, template_dir, launched, runs, **kw):
        # Set BEFORE super().__init__: that is where the mode->deliverer registry is
        # built, so a run_spec assigned afterwards would never be wired in.
        self.run_spec = ScriptLaunchSpec(
            app=AppSpec(name="StubApp"),
            template_dir=Path(template_dir),
            launch_args=lambda script: ["--headless", script],
            modes=(SAVE_AS,),
            timeout=42,
        )
        super().__init__(template_dir, launched, **kw)
        self._runs = runs

    def _scene_objects(self):
        return ["sceneA", "sceneB", "sceneC"]


class HandoffSaveAsTest(unittest.TestCase):
    """``save_as``: one export pipeline, delivered blocking, judged by the artifact.

    The point of the mode registry is that the SAME produce step feeds both routes --
    these pin that the blocking route reuses it rather than growing a second exporter.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "import.py").write_text(
            "BRIDGE_MODES = ('send_to',)\nFBX = r\"__FBX_PATH__\"\nSCALE = __SCALE__\n",
            encoding="utf-8",
        )
        (self.tmp / "_save_scene.py").write_text(
            "BRIDGE_MODES = ('save_as',)\n"
            'FBX = r"__FBX_PATH__"\n'
            'OUT = r"__OUT_FILE__"\n'
            "SCALE = __SCALE__\n",
            encoding="utf-8",
        )
        self.launched, self.runs = [], []

        # Stub the blocking runner: record the call, create the artifact it promised.
        self._orig_run = app_handoff.ScriptRunDeliverer.run

        def _fake_run(app_exe, script_text, *, artifact, launch_args, timeout, env=None):
            self.runs.append(
                {
                    "app": app_exe,
                    "script": script_text,
                    "artifact": artifact,
                    "args": list(launch_args("S.py")),
                    "timeout": timeout,
                    "env": env,
                }
            )
            Path(artifact).write_text("saved", encoding="utf-8")
            return ScriptRunResult(
                artifact=artifact,
                returncode=0,
                output="",
                duration=0.5,
                script_path="S.py",
            )

        app_handoff.ScriptRunDeliverer.run = staticmethod(_fake_run)

    def tearDown(self):
        app_handoff.ScriptRunDeliverer.run = staticmethod(self._orig_run)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _bridge(self):
        return _StubSaveBridge(self.tmp, self.launched, self.runs)

    def test_save_as_runs_headless_and_returns_the_artifact(self):
        br = self._bridge()
        out = self.tmp / "asset.stub"
        result = br.save_as(str(out))

        self.assertIsNotNone(result)
        self.assertEqual(result["mode"], SAVE_AS)
        self.assertEqual(result["template"], "_save_scene")
        self.assertEqual(Path(result["output"]), out)
        self.assertEqual(result["returncode"], 0)
        # Blocking route only -- nothing was launched detached.
        self.assertEqual(self.launched, [])
        self.assertEqual(len(self.runs), 1)
        self.assertEqual(self.runs[0]["args"], ["--headless", "S.py"])
        self.assertEqual(self.runs[0]["timeout"], 42)  # from the run spec

    def test_out_file_is_substituted_into_the_template(self):
        import re

        br = self._bridge()
        out = self.tmp / "asset.stub"
        br.save_as(str(out))
        body = self.runs[0]["script"]
        staged = br.deliverers[SAVE_AS]._staging_path(str(out))
        # The template writes the STAGING sibling; the caller's path is the promotion
        # target, not what the child app is told to save.
        self.assertIn(f'OUT = r"{staged.replace(chr(92), "/")}"', body)
        self.assertIn("SCALE = 1.0", body)  # params ride along as usual
        self.assertFalse(re.findall(r"__[A-Z][A-Z0-9_]*__", body))  # all substituted

    def test_defaults_to_the_whole_scene_not_the_selection(self):
        """"Save the scene as ..." is about the scene; ``send`` stays selection-first."""
        br = self._bridge()
        br.save_as(str(self.tmp / "asset.stub"))
        self.assertEqual(br.exported[-1][0], ("sceneA", "sceneB", "sceneC"))

        br.send(template="import", mode=SEND_TO)
        self.assertEqual(br.exported[-1][0], ("objA", "objB"))

    def test_explicit_objects_win(self):
        br = self._bridge()
        br.save_as(str(self.tmp / "asset.stub"), ["only", "these"])
        self.assertEqual(br.exported[-1][0], ("only", "these"))

    def test_bare_path_gets_the_default_extension(self):
        br = self._bridge()
        result = br.save_as(str(self.tmp / "asset"))
        self.assertTrue(result["output"].endswith(".stub"))
        # An accepted alternative extension is left alone.
        result = br.save_as(str(self.tmp / "asset.stubb"))
        self.assertTrue(result["output"].endswith(".stubb"))

    def test_timeout_override_reaches_the_runner(self):
        br = self._bridge()
        br.save_as(str(self.tmp / "asset.stub"), timeout=7)
        self.assertEqual(self.runs[0]["timeout"], 7)

    def test_missing_artifact_is_a_handled_failure(self):
        """A raising runner is reported, never propagated at the caller."""
        br = self._bridge()

        def _boom(*a, **kw):
            raise RuntimeError("no artifact")

        app_handoff.ScriptRunDeliverer.run = staticmethod(_boom)
        self.assertIsNone(br.save_as(str(self.tmp / "asset.stub")))

    def test_a_failed_run_leaves_no_staging_file_behind(self):
        """An EMPTY artifact is a failure the runner reports WITHOUT removing, and a
        timeout kills the child mid-write -- neither may litter the output folder."""
        br = self._bridge()
        out = self.tmp / "asset.stub"
        staging = Path(br.deliverers[SAVE_AS]._staging_path(str(out)))

        def _half_written(app_exe, script_text, *, artifact, **kw):
            Path(artifact).write_text("", encoding="utf-8")  # empty == failure
            raise RuntimeError("did not produce the expected artifact")

        app_handoff.ScriptRunDeliverer.run = staticmethod(_half_written)
        self.assertIsNone(br.save_as(str(out)))
        self.assertFalse(staging.exists(), "staging file left in the output folder")

    def test_a_failed_promotion_keeps_the_result(self):
        """If only the RENAME fails, the scene is written -- discarding it would throw
        away the run the user waited for; it is kept and named in the error."""
        br = self._bridge()
        out = self.tmp / "asset.stub"
        staging = Path(br.deliverers[SAVE_AS]._staging_path(str(out)))

        orig_replace = app_handoff.os.replace
        app_handoff.os.replace = lambda *a, **kw: (_ for _ in ()).throw(
            OSError("target is locked")
        )
        try:
            self.assertIsNone(br.save_as(str(out)))
        finally:
            app_handoff.os.replace = orig_replace
        self.assertTrue(staging.is_file(), "the written scene was discarded")

    def test_a_failed_save_leaves_an_existing_file_intact(self):
        """The runner CLEARS the artifact path first, so a save-over must be staged.

        Without staging, "save over my scene" would destroy the previous file the
        moment the target app failed for any reason.
        """
        br = self._bridge()
        out = self.tmp / "asset.stub"
        out.write_text("PRECIOUS", encoding="utf-8")

        def _boom(*a, **kw):
            raise RuntimeError("target app died")

        app_handoff.ScriptRunDeliverer.run = staticmethod(_boom)
        self.assertIsNone(br.save_as(str(out)))
        self.assertEqual(out.read_text(encoding="utf-8"), "PRECIOUS")

    def test_the_staging_sibling_keeps_the_extension_and_is_promoted(self):
        """Templates branch on the extension (``.mb`` -> mayaBinary), so it must survive."""
        staging = app_handoff.ScriptRunDeliverer._staging_path(r"C:\out\asset.mb")
        self.assertTrue(staging.endswith(".mb"))
        self.assertEqual(Path(staging).parent, Path(r"C:\out"))
        self.assertNotEqual(Path(staging).name, "asset.mb")

        br = self._bridge()
        out = self.tmp / "asset.stub"
        self.assertIsNotNone(br.save_as(str(out)))
        # The child wrote the sibling; the caller sees only the promoted final file.
        self.assertEqual(self.runs[0]["artifact"], br.deliverers[SAVE_AS]._staging_path(str(out)))
        self.assertTrue(out.is_file())
        self.assertFalse(Path(self.runs[0]["artifact"]).exists())

    def test_send_only_bridge_reports_instead_of_raising(self):
        br = _StubScriptBridge(self.tmp, self.launched)  # no run_spec
        self.assertIsNone(br.save_as(str(self.tmp / "asset.stub")))
        self.assertEqual(self.runs, [])

    def test_modes_dispatch_to_distinct_deliverers(self):
        br = self._bridge()
        self.assertIsInstance(
            br.deliverers[SEND_TO], app_handoff.ScriptLaunchDeliverer
        )
        self.assertIsInstance(br.deliverers[SAVE_AS], app_handoff.ScriptRunDeliverer)
        # An unregistered mode falls back to the default strategy (back-compat).
        self.assertIs(
            br._deliverer_for(HandoffRequest(mode="other")), br.deliverer
        )

    def test_registry_is_instance_owned(self):
        """A class-level dict would leak one bridge's strategies into every other."""
        first, second = self._bridge(), self._bridge()
        self.assertIsNot(first.deliverers, second.deliverers)
        # The class default is None, not a shared {}: a mutable default would be the
        # one object every non-rebinding subclass mutates in place.
        self.assertIsNone(HandoffBridge.deliverers)

    def test_unregistered_mode_map_is_not_a_shared_mutable(self):
        """A bridge that never rebinds `deliverers` must not write to a shared dict."""

        class _Bare(HandoffBridge):
            deliverer = None

            def _resolve_objects(self, objects):
                return list(objects or [])

            def _produce(self, objects, request):
                return Payload(primary="x")

        # Reading through the None default must not materialise a class-level dict.
        self.assertIsNone(_Bare()._deliverer_for(HandoffRequest(mode=SEND_TO)))
        self.assertIsNone(HandoffBridge.deliverers)
        self.assertIsNone(_Bare.deliverers)

    def test_save_as_rejects_a_template_that_does_not_declare_the_mode(self):
        br = self._bridge()
        self.assertIsNone(br.save_as(str(self.tmp / "a.stub"), template="import"))
        self.assertEqual(self.runs, [])
        self.assertEqual(br.exported, [])  # aborted in preflight, before exporting


class _StubRoundTripBridge(_StubScriptBridge):
    """A stub bridge with the ROUND-TRIP route wired: run headless, then ingest."""

    def __init__(self, template_dir, launched, **kw):
        # Before super().__init__ — that is where the mode->deliverer registry is built.
        self.round_trip_spec = ScriptLaunchSpec(
            app=AppSpec(name="StubApp"),
            template_dir=Path(template_dir),
            launch_args=lambda script: ["-cfi", script],
            modes=(ROUND_TRIP,),
            timeout=11,
        )
        super().__init__(template_dir, launched, **kw)
        self.ingested = []

    def _ingest(self, result, objects, payload, request):
        # What a real bridge does here: re-import result["artifact"], transfer onto
        # `objects`, clean up. The stub just records that it was handed both.
        self.ingested.append((dict(result), list(objects), payload.primary))
        return {**result, "transferred": len(objects)}


class HandoffRoundTripTest(unittest.TestCase):
    """The inbound axis: deliver blocking, then bring the result back.

    The whole point of ROUND_TRIP being a *mode* is that it reuses the one produce
    step; these pin that, and that the in-place contract differs from save_as in the
    two ways that matter (no staging, judged by change).
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "import.py").write_text(
            "BRIDGE_MODES = ('send_to',)\nFBX = r\"__FBX_PATH__\"\nSCALE = __SCALE__\n",
            encoding="utf-8",
        )
        (self.tmp / "unwrap.py").write_text(
            "BRIDGE_MODES = ('round_trip',)\n"
            'FBX = r"__FBX_PATH__"\n'
            'OUT = r"__OUT_FILE__"\n'
            "SCALE = __SCALE__\n",
            encoding="utf-8",
        )
        self.launched, self.runs = [], []
        self._orig_run = app_handoff.ScriptRunDeliverer.run

        def _fake_run(app_exe, script_text, *, artifact, launch_args, timeout,
                     env=None, expect=None):
            self.runs.append(
                {"artifact": artifact, "timeout": timeout, "expect": expect,
                 "args": list(launch_args("S.py")), "script": script_text}
            )
            # The target app edits the payload in place.
            Path(artifact).write_text("unwrapped", encoding="utf-8")
            return ScriptRunResult(
                artifact=artifact, returncode=0, output="", duration=0.5,
                script_path="S.py",
            )

        app_handoff.ScriptRunDeliverer.run = staticmethod(_fake_run)

    def tearDown(self):
        app_handoff.ScriptRunDeliverer.run = staticmethod(self._orig_run)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _bridge(self):
        return _StubRoundTripBridge(self.tmp, self.launched)

    def test_round_trip_runs_headless_then_ingests(self):
        br = self._bridge()
        result = br.round_trip(objects=["a", "b"], template="unwrap")

        self.assertIsNotNone(result)
        self.assertEqual(result["mode"], ROUND_TRIP)
        self.assertEqual(result["transferred"], 2)  # _ingest ran and enriched
        self.assertEqual(self.launched, [])  # blocking route: nothing detached
        self.assertEqual(len(self.runs), 1)
        self.assertEqual(self.runs[0]["args"], ["-cfi", "S.py"])
        self.assertEqual(self.runs[0]["timeout"], 11)

    def test_the_artifact_is_the_payload_itself(self):
        """In place: the app reads and writes ONE path — no staging sibling."""
        br = self._bridge()
        result = br.round_trip(objects=["a"], template="unwrap")
        self.assertEqual(self.runs[0]["artifact"], result["payload"])
        self.assertEqual(result["artifact"], result["payload"])

    def test_the_runner_is_asked_to_judge_by_change_not_creation(self):
        """Clearing the path first would delete the app's own input."""
        from pythontk.core_utils.script_run import REWRITTEN

        br = self._bridge()
        br.round_trip(objects=["a"], template="unwrap")
        self.assertEqual(self.runs[0]["expect"], REWRITTEN)

    def test_ingest_receives_the_same_objects_produce_exported(self):
        """Pairing the result back onto the originals needs no extra bookkeeping."""
        br = self._bridge()
        br.round_trip(objects=["a", "b", "c"], template="unwrap")
        _result, objects, primary = br.ingested[0]
        self.assertEqual(objects, ["a", "b", "c"])
        self.assertEqual(tuple(br.exported[0][0]), ("a", "b", "c"))
        self.assertEqual(br.exported[0][1], primary)

    def test_out_file_points_at_the_payload(self):
        br = self._bridge()
        br.round_trip(objects=["a"], template="unwrap")
        script = self.runs[0]["script"]
        payload = self.runs[0]["artifact"].replace("\\", "/")
        self.assertIn(f'OUT = r"{payload}"', script)

    def test_a_failed_run_is_handled_and_never_ingests(self):
        br = self._bridge()

        def _boom(*a, **kw):
            raise RuntimeError("target app died")

        app_handoff.ScriptRunDeliverer.run = staticmethod(_boom)
        self.assertIsNone(br.round_trip(objects=["a"], template="unwrap"))
        self.assertEqual(br.ingested, [])

    def test_round_trip_rejects_a_template_that_does_not_declare_the_mode(self):
        br = self._bridge()
        self.assertIsNone(br.round_trip(objects=["a"], template="import"))
        self.assertEqual(self.runs, [])
        self.assertEqual(br.exported, [])  # aborted in preflight, before exporting

    def test_a_bridge_without_the_spec_reports_instead_of_raising(self):
        br = _StubScriptBridge(self.tmp, self.launched)  # no round_trip_spec
        self.assertIsNone(br.round_trip(template="unwrap"))
        self.assertEqual(self.runs, [])

    def test_a_bridge_that_forgets_ingest_is_warned_not_silently_pointless(self):
        """The default identity ingest runs the target and throws the result away.

        Warned rather than refused: by the time it could be detected the target has
        already run, and discarding that to raise would be worse than reporting it.
        """

        class _NoIngest(_StubRoundTripBridge):
            pass

        _NoIngest._ingest = HandoffBridge._ingest  # undo the stub's override
        br = _NoIngest(self.tmp, self.launched)
        with self.assertLogs(br.logger, level="WARNING") as caught:
            result = br.round_trip(objects=["a"], template="unwrap")
        self.assertIsNotNone(result)  # the run still happened and is reported
        self.assertIn("_ingest", "\n".join(caught.output))

    def test_an_overriding_bridge_is_not_warned(self):
        br = self._bridge()  # _StubRoundTripBridge DOES override _ingest
        with self.assertNoLogs(br.logger, level="WARNING"):
            br.round_trip(objects=["a"], template="unwrap")

    def test_a_secondary_spec_that_forgets_modes_is_refused_at_construction(self):
        """ScriptLaunchSpec.modes defaults to (SEND_TO,) -- a secondary spec that
        omits it would REPLACE the interactive send deliverer, so send() would run
        the target headlessly and fail on a missing artifact with nothing pointing
        at the one-word omission. Registering a mode twice is a declaration bug."""

        class _Clashing(_StubScriptBridge):
            def __init__(self, template_dir, launched, **kw):
                self.round_trip_spec = ScriptLaunchSpec(
                    app=AppSpec(name="StubApp"),
                    template_dir=Path(template_dir),
                    launch_args=lambda s: [s],
                    # modes= omitted on purpose -> defaults to (SEND_TO,)
                )
                super().__init__(template_dir, launched, **kw)

        with self.assertRaises(ValueError) as ctx:
            _Clashing(self.tmp, self.launched)
        self.assertIn("round_trip_spec", str(ctx.exception))
        self.assertIn(SEND_TO, str(ctx.exception))

    def test_the_two_secondary_specs_cannot_claim_the_same_mode(self):
        """run_spec and round_trip_spec are checked against each other too."""

        def _spec(modes):
            return ScriptLaunchSpec(
                app=AppSpec(name="StubApp"),
                template_dir=Path(self.tmp),
                launch_args=lambda s: [s],
                modes=modes,
            )

        class _Both(_StubScriptBridge):
            def __init__(inner, template_dir, launched, **kw):
                inner.run_spec = _spec((SAVE_AS,))
                inner.round_trip_spec = _spec((SAVE_AS,))  # collides with run_spec
                super().__init__(template_dir, launched, **kw)

        with self.assertRaises(ValueError):
            _Both(self.tmp, self.launched)

    def test_default_ingest_is_the_identity_for_one_way_bridges(self):
        """A send-only bridge must pay nothing for the step existing."""
        br = _StubScriptBridge(self.tmp, self.launched)
        delivered = {"script": "s", "template": "import"}
        self.assertIs(
            br._ingest(delivered, ["a"], Payload(primary="p"), HandoffRequest()),
            delivered,
        )


class RoundTripArtifactShapeTest(unittest.TestCase):
    """The second round-trip shape: the target writes a NEW artifact to re-ingest.

    RizomUV's round trip rewrites the payload in place; mayatk's lightmap bake returns a
    manifest instead. Both land the result back in the HOST, which is what the mode
    means -- so the mode must not be welded to one artifact shape, or the bake is forced
    to advertise itself as ``save_as`` and tell the artist to go find a file that is
    deliberately never kept.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "bake.py").write_text(
            "BRIDGE_MODES = ('round_trip',)\n"
            'FBX = r"__FBX_PATH__"\n'
            'OUT = r"__OUT_FILE__"\n'
            "SCALE = __SCALE__\n",
            encoding="utf-8",
        )
        self.launched, self.runs = [], []
        self._orig_run = app_handoff.ScriptRunDeliverer.run

        def _fake_run(app_exe, script_text, *, artifact, launch_args, timeout,
                      env=None, expect=None):
            self.runs.append({"artifact": artifact, "expect": expect})
            Path(artifact).write_text("{}", encoding="utf-8")
            return ScriptRunResult(
                artifact=artifact, returncode=0, output="", duration=0.5,
                script_path="S.py",
            )

        app_handoff.ScriptRunDeliverer.run = staticmethod(_fake_run)

    def tearDown(self):
        app_handoff.ScriptRunDeliverer.run = staticmethod(self._orig_run)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _bridge(self):
        tmp, launched, ingested = self.tmp, self.launched, []

        class _ArtifactRoundTripBridge(_StubScriptBridge):
            # ONE spec serving both blocking modes: identical machinery, and only the
            # destination of the artifact differs. Assigned before super().__init__,
            # which is where the mode->deliverer registry is built.
            def __init__(inner, **kw):
                inner.run_spec = ScriptLaunchSpec(
                    app=AppSpec(name="StubApp"),
                    template_dir=tmp,
                    launch_args=lambda script: ["-b", script],
                    modes=(SAVE_AS, ROUND_TRIP),
                )
                super().__init__(tmp, launched, **kw)

            def _ingest(inner, result, objects, payload, request):
                ingested.append(dict(result))
                return {**result, "reassembled": len(objects)}

        br = _ArtifactRoundTripBridge()
        br.ingested = ingested
        return br

    def test_round_trip_serves_an_artifact_writing_template(self):
        br = self._bridge()
        out = str(self.tmp / "bake.result.json")
        result = br.round_trip(objects=["a", "b"], template="bake", out=out)

        self.assertIsNotNone(result)
        self.assertEqual(result["mode"], ROUND_TRIP)
        self.assertEqual(result["output"], out)
        # Judged by CREATION (a new file), not by the payload having changed -- and
        # written to the staging sibling first, exactly like save_as.
        self.assertIsNone(self.runs[0]["expect"])
        self.assertNotEqual(self.runs[0]["artifact"], out)
        # The return leg ran: that is the whole difference from save_as.
        self.assertEqual(result["reassembled"], 2)
        self.assertEqual(len(br.ingested), 1)

    def test_round_trip_is_gated_on_the_deliverer_not_on_round_trip_spec(self):
        """This bridge has no ``round_trip_spec`` at all and must still work."""
        br = self._bridge()
        self.assertIsNone(br.round_trip_spec)
        self.assertIn(ROUND_TRIP, br.deliverers)
        self.assertIsNotNone(
            br.round_trip(objects=["a"], template="bake", out=str(self.tmp / "o.json"))
        )

    def test_a_bridge_without_the_mode_refuses_and_says_where_to_declare_it(self):
        br = _StubScriptBridge(self.tmp, self.launched)  # send-only
        self.assertNotIn(ROUND_TRIP, br.deliverers)
        self.assertIsNone(br.round_trip(objects=["a"], template="bake"))
        self.assertEqual(self.runs, [])

    def test_the_combo_listing_covers_every_registered_mode(self):
        """A secondary spec's modes must reach the panel's (template, mode) listing.

        ``list_template_modes`` filters declarations against the allowed tuple and
        silently falls back to its FIRST entry, so listing only ``spec.modes`` relabels
        every blocking template as the interactive send -- which the panel then routes
        through ``send()``, leaving ``__OUT_FILE__`` empty and the run to fail minutes
        in. Derived from the deliverer registry so it cannot fall out of step.
        """
        br = self._bridge()
        self.assertEqual(br.modes, (SEND_TO, SAVE_AS, ROUND_TRIP))
        self.assertEqual(br.modes[0], SEND_TO)  # the lenient fallback is preserved
        self.assertIn(("bake", ROUND_TRIP), br.list_template_modes())

        # A send-only bridge is unaffected: no secondary spec, no extra modes.
        self.assertEqual(_StubScriptBridge(self.tmp, self.launched).modes, (SEND_TO,))


if __name__ == "__main__":
    unittest.main()
