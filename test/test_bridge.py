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
    SEND_TO,
)


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
        stems = [p.stem for p in script_template.list_templates(self.tmp)]
        self.assertEqual(stems, ["import", "replace_scene"])

    def test_list_templates_honors_extension(self):
        self._write("load.lua")
        self._write("ignore.py")
        stems = [p.stem for p in script_template.list_templates(self.tmp, ".lua")]
        self.assertEqual(stems, ["load"])

    def test_template_modes_declared(self):
        path = self._write("a.py", "BRIDGE_MODES = ('send_to',)\n")
        self.assertEqual(script_template.template_modes(path), ("send_to",))

    def test_template_modes_fallback_when_absent(self):
        path = self._write("a.py", "x = 1\n")
        self.assertEqual(script_template.template_modes(path), (SEND_TO,))

    def test_template_modes_filters_unknown(self):
        path = self._write("a.py", "BRIDGE_MODES = ('send_to', 'bogus')\n")
        # 'bogus' isn't in the allowed set -> dropped; only 'send_to' survives.
        self.assertEqual(script_template.template_modes(path), ("send_to",))

    def test_template_modes_missing_file_fallback(self):
        self.assertEqual(
            script_template.template_modes(self.tmp / "nope.py", ("send_to",)),
            ("send_to",),
        )

    def test_template_modes_custom_field(self):
        path = self._write("a.py", "MODES = ('send_to',)\n")
        # Default field name finds nothing -> fallback.
        self.assertEqual(script_template.template_modes(path), (SEND_TO,))
        # Custom field name reads the declaration.
        self.assertEqual(
            script_template.template_modes(path, field="MODES"), ("send_to",)
        )

    def test_list_template_modes_pairs(self):
        self._write("import.py", "BRIDGE_MODES = ('send_to',)\n")
        self._write("frame.py")  # no declaration -> fallback mode
        self.assertEqual(
            script_template.list_template_modes(self.tmp),
            [("frame", "send_to"), ("import", "send_to")],
        )

    def test_render_template_substitutes(self):
        path = self._write("t.py", 'FBX = r"__FBX_PATH__"\nN = __COUNT__\n')
        out = script_template.render_template(
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
            "BRIDGE_MODES = ('send_to',)\n"
            'FBX = r"__FBX_PATH__"\nSCALE = __SCALE__\n',
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


if __name__ == "__main__":
    unittest.main()
