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
        self.assertEqual(HandoffBridge.deliverers, {})

    def test_save_as_rejects_a_template_that_does_not_declare_the_mode(self):
        br = self._bridge()
        self.assertIsNone(br.save_as(str(self.tmp / "a.stub"), template="import"))
        self.assertEqual(self.runs, [])
        self.assertEqual(br.exported, [])  # aborted in preflight, before exporting


if __name__ == "__main__":
    unittest.main()
