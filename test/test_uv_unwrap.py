# !/usr/bin/python
# coding=utf-8
"""Unit tests for UvUnwrap.

Network- and engine-free: executable resolution and the subprocess call are
mocked, so the real Ministry of Flat / BFF binaries are never needed. One test
drives a genuine subprocess (a generated Python "engine") to cover the
AppLauncher path the mocks bypass. Opt-in integration tests run the real
engines when ``PYTHONTK_INTEGRATION_TESTS=1``.

Run with:
    python -m pytest test_uv_unwrap.py -v
    python test_uv_unwrap.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk import UvUnwrap, UsdMeshWriter
from pythontk.file_utils.uv_unwrap import _uv_unwrap


# A quad cube — n-gon-free but non-triangulated, which both engines accept.
CUBE_OBJ = """\
o pCube1
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v -0.5 0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
f 1 2 4 3
f 3 4 6 5
f 5 6 8 7
f 7 8 2 1
f 2 8 6 4
f 7 1 3 5
"""

def _unwrapped(obj_text):
    """The cube as an engine returns it: face-varying UVs referenced as v/vt.

    Mirrors real Ministry of Flat / BFF output, where topology is unchanged and
    every face corner gains a texture-coordinate index.
    """
    head, faces, uvs, corner = [], [], [], 0
    for line in obj_text.splitlines():
        if not line.startswith("f "):
            head.append(line)
            continue
        indices = []
        for token in line.split()[1:]:
            corner += 1
            uvs.append(f"vt {corner / 32.0:.4f} {corner / 32.0:.4f}")
            indices.append(f"{token}/{corner}")
        faces.append("f " + " ".join(indices))
    return "\n".join(head + uvs + faces) + "\n"


UNWRAPPED_OBJ = _unwrapped(CUBE_OBJ)


class _Result:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _UvUnwrapTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="uvunwrap_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = self._write("cube.obj", CUBE_OBJ)
        self.dst = os.path.join(self.tmp, "out.obj")
        self.exe = self._write("engine.exe", "")

    def _write(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def _run_stub(self, returncode=0, payload=UNWRAPPED_OBJ, record=None):
        """A fake AppLauncher.run that writes *payload* to the output path."""

        def stub(app, args=None, timeout=None, hide_window=False, **kwargs):
            if record is not None:
                record.update(
                    exe=app, args=list(args or []), timeout=timeout,
                    hide_window=hide_window,
                )
            if payload is not None:
                with open(args[1], "w", encoding="utf-8") as f:
                    f.write(payload)
            return _Result(returncode)

        return stub

    def _patch_run(self, **kwargs):
        return patch.object(_uv_unwrap.AppLauncher, "run", self._run_stub(**kwargs))

    def _patch_resolve(self, path=None):
        return patch.object(UvUnwrap, "resolve_engine", return_value=path or self.exe)


class TestResolveEngine(_UvUnwrapTestCase):
    def test_env_var_override_wins(self):
        with patch.dict(os.environ, {"PYTHONTK_MOF_EXE": self.exe}):
            self.assertEqual(UvUnwrap.resolve_engine("mof"), self.exe)

    def test_resolves_from_path(self):
        with patch.dict(os.environ, {}, clear=False), patch.object(
            _uv_unwrap.AppLauncher, "resolve_app_path", return_value=self.exe
        ):
            os.environ.pop("PYTHONTK_BFF_EXE", None)
            self.assertEqual(UvUnwrap.resolve_engine("bff"), self.exe)

    def test_resolves_from_managed_catalog(self):
        with patch.object(
            _uv_unwrap.AppLauncher, "resolve_app_path", return_value=None
        ), patch(
            "pythontk.core_utils.app_installer.AppInstaller.get_path",
            return_value=self.exe,
        ):
            self.assertEqual(UvUnwrap.resolve_engine("bff"), self.exe)

    def test_unknown_engine_lists_valid_names(self):
        with self.assertRaises(ValueError) as ctx:
            UvUnwrap.resolve_engine("nope")
        self.assertIn("bff", str(ctx.exception))
        self.assertIn("mof", str(ctx.exception))

    def test_not_required_returns_none(self):
        with patch.object(
            _uv_unwrap.AppLauncher, "resolve_app_path", return_value=None
        ), patch(
            "pythontk.core_utils.app_installer.AppInstaller.get_path",
            return_value=None,
        ):
            self.assertIsNone(UvUnwrap.resolve_engine("mof", required=False))

    def test_mof_missing_message_is_actionable(self):
        with patch.object(
            _uv_unwrap.AppLauncher, "resolve_app_path", return_value=None
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                UvUnwrap.resolve_engine("mof")
        message = str(ctx.exception)
        self.assertIn(_uv_unwrap.MOF_DOWNLOAD_URL, message)
        self.assertIn("PYTHONTK_MOF_EXE", message)

    def test_mof_is_never_auto_installed(self):
        """Its license forbids redistribution — auto_install must not download."""
        with patch.object(
            _uv_unwrap.AppLauncher, "resolve_app_path", return_value=None
        ), patch(
            "pythontk.core_utils.app_installer.AppInstaller.ensure"
        ) as ensure:
            with self.assertRaises(FileNotFoundError):
                UvUnwrap.resolve_engine("mof", auto_install=True, prompt=False)
        ensure.assert_not_called()

    def test_bff_installs_with_pinned_url_and_hash(self):
        with patch.object(
            _uv_unwrap.AppLauncher, "resolve_app_path", return_value=None
        ), patch(
            "pythontk.core_utils.app_installer.AppInstaller.get_path",
            return_value=None,
        ), patch(
            "pythontk.core_utils.app_installer.AppInstaller.ensure",
            return_value=self.exe,
        ) as ensure:
            self.assertEqual(
                UvUnwrap.resolve_engine("bff", auto_install=True, prompt=False),
                self.exe,
            )
        kwargs = ensure.call_args.kwargs
        self.assertEqual(kwargs["platforms"], _uv_unwrap.BFF_PLATFORMS)
        self.assertEqual(kwargs["sha256"], _uv_unwrap.BFF_SHA256)
        self.assertEqual(kwargs["version"], _uv_unwrap.BFF_VERSION)

    def test_bff_refuses_silent_download_without_tty(self):
        with patch.object(
            _uv_unwrap.AppLauncher, "resolve_app_path", return_value=None
        ), patch(
            "pythontk.core_utils.app_installer.AppInstaller.get_path",
            return_value=None,
        ), patch(
            "pythontk.core_utils.app_installer.AppInstaller.ensure"
        ) as ensure, patch.object(
            sys, "stdin"
        ) as stdin:
            stdin.isatty.return_value = False
            with self.assertRaises(FileNotFoundError) as ctx:
                UvUnwrap.resolve_engine("bff", auto_install=True, prompt=True)
        ensure.assert_not_called()
        self.assertIn("prompt=False", str(ctx.exception))


class TestAvailableEngines(_UvUnwrapTestCase):
    def test_reports_every_engine(self):
        with patch.object(UvUnwrap, "resolve_engine", return_value=None):
            found = UvUnwrap.available_engines()
        self.assertEqual(set(found), set(UvUnwrap.ENGINES))
        self.assertTrue(all(v is None for v in found.values()))

    def test_never_raises_or_installs(self):
        with patch.object(
            UvUnwrap, "resolve_engine", side_effect=RuntimeError("boom")
        ), patch(
            "pythontk.core_utils.app_installer.AppInstaller.ensure"
        ) as ensure:
            found = UvUnwrap.available_engines()
        self.assertTrue(all(v is None for v in found.values()))
        ensure.assert_not_called()


class TestPreflight(_UvUnwrapTestCase):
    def test_missing_input(self):
        with self.assertRaises(FileNotFoundError):
            UvUnwrap.unwrap(os.path.join(self.tmp, "nope.obj"))

    def test_non_obj_input(self):
        fbx = self._write("mesh.fbx", "x")
        with self.assertRaises(ValueError):
            UvUnwrap.unwrap(fbx)

    def test_empty_input(self):
        empty = self._write("empty.obj", "")
        with self.assertRaises(RuntimeError):
            UvUnwrap.unwrap(empty)

    def test_unknown_engine(self):
        with self.assertRaises(ValueError):
            UvUnwrap.unwrap(self.src, engine="rizom")

    def test_unknown_param_lists_supported(self):
        with self.assertRaises(TypeError) as ctx:
            UvUnwrap.unwrap(self.src, engine="bff", nCones=4)
        self.assertIn("n_cones", str(ctx.exception))

    def test_force_hard_surface_is_not_a_real_flag(self):
        """Ministry of Flat classifies automatically — there is no such switch."""
        with self.assertRaises(TypeError):
            UvUnwrap.unwrap(self.src, engine="mof", force_hard_surface=True)

    def test_existing_output_requires_overwrite(self):
        self._write("out.obj", "stale")
        with self.assertRaises(FileExistsError):
            UvUnwrap.unwrap(self.src, self.dst)

    def test_overwrite_replaces_stale_output(self):
        self._write("out.obj", "stale")
        with self._patch_resolve(), self._patch_run():
            UvUnwrap.unwrap(self.src, self.dst, overwrite=True)
        with open(self.dst, encoding="utf-8") as f:
            self.assertNotIn("stale", f.read())

    def test_params_checked_before_engine_resolution(self):
        with patch.object(UvUnwrap, "resolve_engine") as resolve:
            with self.assertRaises(TypeError):
                UvUnwrap.unwrap(self.src, engine="bff", bogus=1)
        resolve.assert_not_called()


class TestArgv(_UvUnwrapTestCase):
    def test_mof_flags(self):
        rec = {}
        with self._patch_resolve(), self._patch_run(record=rec):
            UvUnwrap.unwrap(
                self.src,
                self.dst,
                engine="mof",
                resolution=2048,
                separate_hard_edges=True,
                use_normals=False,
            )
        args = rec["args"]
        self.assertEqual(args[0], os.path.abspath(self.src))
        self.assertEqual(args[1], os.path.abspath(self.dst))
        self.assertEqual(args[args.index("-RESOLUTION") + 1], "2048")
        self.assertEqual(args[args.index("-SEPARATE") + 1], "TRUE")
        self.assertEqual(args[args.index("-NORMALS") + 1], "FALSE")

    def test_mof_omits_unset_flags(self):
        rec = {}
        with self._patch_resolve(), self._patch_run(record=rec):
            UvUnwrap.unwrap(self.src, self.dst, engine="mof")
        self.assertEqual(rec["args"], [os.path.abspath(self.src), os.path.abspath(self.dst)])

    def test_bff_flags(self):
        rec = {}
        with self._patch_resolve(), self._patch_run(record=rec):
            UvUnwrap.unwrap(self.src, self.dst, engine="bff", n_cones=8)
        self.assertIn("--nCones=8", rec["args"])
        self.assertIn("--normalizeUVs", rec["args"])

    def test_bff_normalize_can_be_disabled(self):
        rec = {}
        with self._patch_resolve(), self._patch_run(record=rec):
            UvUnwrap.unwrap(self.src, self.dst, engine="bff", normalize_uvs=False)
        self.assertNotIn("--normalizeUVs", rec["args"])

    def test_hides_console_window_and_forwards_timeout(self):
        rec = {}
        with self._patch_resolve(), self._patch_run(record=rec):
            UvUnwrap.unwrap(self.src, self.dst, engine="bff", timeout=42)
        self.assertTrue(rec["hide_window"])
        self.assertEqual(rec["timeout"], 42)


class TestPostflight(_UvUnwrapTestCase):
    def test_nonzero_exit_with_valid_output_succeeds(self):
        """Ministry of Flat exits 1 on a fully successful run."""
        with self._patch_resolve(), self._patch_run(returncode=1):
            out = UvUnwrap.unwrap(self.src, self.dst, engine="mof")
        self.assertTrue(UvUnwrap._has_uvs(out))

    def test_missing_output_raises(self):
        with self._patch_resolve(), self._patch_run(payload=None):
            with self.assertRaises(RuntimeError) as ctx:
                UvUnwrap.unwrap(self.src, self.dst, engine="bff")
        self.assertIn("no output file", str(ctx.exception))

    def test_empty_output_raises(self):
        with self._patch_resolve(), self._patch_run(payload=""):
            with self.assertRaises(RuntimeError) as ctx:
                UvUnwrap.unwrap(self.src, self.dst, engine="bff")
        self.assertIn("empty output", str(ctx.exception))

    def test_output_without_uvs_raises_with_hint(self):
        with self._patch_resolve(), self._patch_run(payload=CUBE_OBJ):
            with self.assertRaises(RuntimeError) as ctx:
                UvUnwrap.unwrap(self.src, self.dst, engine="bff")
        message = str(ctx.exception)
        self.assertIn("no UVs", message)
        self.assertIn("BFF", message)

    def test_timeout_is_actionable(self):
        def boom(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="engine", timeout=1)

        with self._patch_resolve(), patch.object(_uv_unwrap.AppLauncher, "run", boom):
            with self.assertRaises(RuntimeError) as ctx:
                UvUnwrap.unwrap(self.src, self.dst, engine="bff", timeout=1)
        self.assertIn("timeout", str(ctx.exception).lower())


class TestRoundTrip(_UvUnwrapTestCase):
    def test_returns_parseable_obj_with_uvs(self):
        with self._patch_resolve(), self._patch_run():
            out = UvUnwrap.unwrap(self.src, self.dst, engine="mof")
        data = UsdMeshWriter.from_obj(out)
        self.assertTrue(data["uvs"])
        self.assertEqual(len(data["points"]), 8)

    def test_default_output_is_a_temp_path(self):
        with self._patch_resolve(), self._patch_run():
            out = UvUnwrap.unwrap(self.src, engine="mof")
        self.addCleanup(lambda: os.path.exists(out) and os.remove(out))
        self.assertIn(UvUnwrap.TEMP_PREFIX, os.path.basename(out))
        self.assertTrue(os.path.isfile(out))


class TestAliases(_UvUnwrapTestCase):
    def test_hard_surface_selects_mof(self):
        with patch.object(UvUnwrap, "unwrap", return_value="x") as unwrap:
            UvUnwrap.hard_surface(self.src)
        self.assertEqual(unwrap.call_args.kwargs["engine"], "mof")

    def test_organic_selects_bff(self):
        with patch.object(UvUnwrap, "unwrap", return_value="x") as unwrap:
            UvUnwrap.organic(self.src)
        self.assertEqual(unwrap.call_args.kwargs["engine"], "bff")


class TestRealSubprocess(_UvUnwrapTestCase):
    """Exercises the genuine AppLauncher.run path the other suites mock."""

    def test_drives_a_real_process(self):
        script = self._write(
            "fake_engine.py",
            textwrap.dedent(
                """
                import sys
                src, dst = sys.argv[1], sys.argv[2]
                with open(src) as f:
                    text = f.read()
                with open(dst, "w") as f:
                    f.write(text + "vt 0.5 0.5\\n")
                sys.exit(1)  # mimic Ministry of Flat's nonzero success
                """
            ),
        )
        spec = UvUnwrap.ENGINES["bff"]
        args = [self.src, self.dst]
        result = _uv_unwrap.AppLauncher.run(
            sys.executable, args=[script] + args, timeout=60, hide_window=True
        )
        self.assertEqual(result.returncode, 1)
        UvUnwrap._postflight(spec, self.dst, result)  # must not raise
        self.assertTrue(UvUnwrap._has_uvs(self.dst))


@unittest.skipUnless(
    os.environ.get("PYTHONTK_INTEGRATION_TESTS") == "1",
    "Set PYTHONTK_INTEGRATION_TESTS=1 to run the real engines.",
)
class TestRealEngines(_UvUnwrapTestCase):
    def _require(self, engine):
        path = UvUnwrap.resolve_engine(engine, required=False)
        if not path:
            self.skipTest(f"{engine} is not installed on this machine")

    def test_mof_unwraps_a_cube(self):
        self._require("mof")
        out = UvUnwrap.hard_surface(self.src, self.dst, resolution=1024)
        self.assertTrue(UvUnwrap._has_uvs(out))

    def test_bff_unwraps_a_cube(self):
        self._require("bff")
        out = UvUnwrap.organic(self.src, self.dst, n_cones=4)
        self.assertTrue(UvUnwrap._has_uvs(out))

    def test_engines_preserve_topology(self):
        """Both engines return the input mesh unchanged apart from UVs.

        The DCC layer maps UVs back by component index on this guarantee.
        """
        source = UsdMeshWriter.from_obj(self.src)
        for engine in ("mof", "bff"):
            self._require(engine)
            out = os.path.join(self.tmp, f"{engine}.obj")
            UvUnwrap.unwrap(self.src, out, engine=engine, overwrite=True)
            result = UsdMeshWriter.from_obj(out)
            self.assertEqual(source["points"], result["points"], engine)
            self.assertEqual(
                source["face_vertex_counts"], result["face_vertex_counts"], engine
            )
            self.assertEqual(
                source["face_vertex_indices"], result["face_vertex_indices"], engine
            )


class TestModuleInvariant(unittest.TestCase):
    """Helpers live on classes, not at module scope (package standard).

    The argv builders are ``ENGINES`` data and so must exist before the
    registry is built -- they live on ``_UvUnwrapInternal`` rather than at
    module scope, and ``UvUnwrap`` inherits them.
    """

    def test_no_top_level_functions(self):
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(_uv_unwrap))
        offenders = [
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.assertEqual(offenders, [], f"module-level def(s): {offenders}")

    def test_engine_builders_resolve_through_the_class(self):
        """Each spec's build_args stays callable after the move onto the base."""
        for name, spec in UvUnwrap.ENGINES.items():
            argv = spec.build_args("in.obj", "out.obj", {})
            self.assertEqual(argv[:2], ["in.obj", "out.obj"], name)


if __name__ == "__main__":
    unittest.main()
