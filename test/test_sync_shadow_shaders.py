#!/usr/bin/python
# coding=utf-8
"""Guard: the generated horizon shader bodies must not drift from the source.

``pythontk/geo_utils/shadow_horizon.glsl`` is the one implementation of the
coverage-aware horizon evaluation, and ``HorizonMap.alpha`` beside it is the
numeric oracle it is written against. Maya and Blender assemble it at runtime
through :meth:`ShadowHorizon.shader_source`; the WebXR viewer and Unity cannot
(a browser and a Unity project have no Python), so they carry generated
mirrors. A copy is only safe while it is provably identical, so the copies are
generated (``m3trik/scripts/sync_shadow_shaders.py``) and this pins them — the
same contract the API registries and the staged RPC cores use.

Skips cleanly outside the monorepo layout (a standalone pythontk checkout has
no sibling packages to mirror into).

Run with:
    python -m pytest test_sync_shadow_shaders.py -v
"""

import importlib.util
import re
import unittest
from pathlib import Path

import pythontk
from pythontk import ShadowHorizon


def _repo_root() -> Path:
    # .../<root>/pythontk/pythontk/__init__.py -> .../<root>
    return Path(pythontk.__file__).resolve().parents[2]


def _load_syncer():
    """Import the sync script by path — ``m3trik/scripts`` is not a package."""
    script = _repo_root() / "m3trik" / "scripts" / "sync_shadow_shaders.py"
    if not script.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_sync_shadow_shaders", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestShaderSourceSurface(unittest.TestCase):
    """:meth:`ShadowHorizon.shader_source`, which needs no sibling packages."""

    def test_each_language_gets_the_same_body_behind_its_own_prologue(self):
        glsl = ShadowHorizon.shader_source("glsl")
        hlsl = ShadowHorizon.shader_source("hlsl")
        self.assertTrue(hlsl.startswith("#define SH_HLSL"))
        self.assertEqual(hlsl[len("#define SH_HLSL 1\n") :], glsl)
        for text in (glsl, hlsl):
            self.assertIn("float ShAlpha(", text)
            self.assertIn("float ShLayerAlpha(", text)
            self.assertIn("ShGrid ShMakeGrid(", text)

    def test_the_body_declares_no_uniform_and_samples_no_texture(self):
        """It is a body, not a shader: the host owns every binding.

        A uniform or a sampler leaking in here would compile in whichever
        engine it was written for and fail in the other three, which is the
        whole failure mode a shared body exists to remove.
        """
        body = ShadowHorizon.shader_source("glsl")
        # Comments discuss the host's uniforms by name; only the code counts.
        code = re.sub(r"//[^\n]*", "", body)
        for banned in (
            "uniform",
            "sampler2D",
            "texelFetch",
            "Texture2D",
            "SamplerState",
        ):
            self.assertNotIn(
                banned,
                code,
                f"the shared body's CODE names {banned!r}, which only a host may",
            )
        # ...and it reaches the texture only through the host's hook.
        self.assertIn("SH_Fetch(", code)

    def test_the_body_carries_nothing_a_js_template_literal_would_end(self):
        """The viewer's mirror lands inside a template literal, where a
        backtick is a *syntax* error in a module the browser then refuses
        whole — with the shader nowhere in the message."""
        body = ShadowHorizon.shader_source("glsl")
        self.assertNotIn(chr(96), body)
        self.assertNotIn("${", body)

    def test_an_engine_is_not_a_language(self):
        """``shader_source("unity")`` would put the per-engine fork back in the
        one place that exists to remove it."""
        with self.assertRaises(ValueError):
            ShadowHorizon.shader_source("unity")


class TestMirrorsInSync(unittest.TestCase):
    def setUp(self):
        self.syncer = _load_syncer()
        if self.syncer is None:
            self.skipTest("m3trik sibling not present (standalone checkout)")
        self.targets = self.syncer.mirrors()
        if not self.targets:
            self.skipTest("no consumer packages checked out")

    def test_every_mirror_carries_the_current_body(self):
        drift = self.syncer.out_of_sync()
        self.assertEqual(
            [str(p) for p in drift],
            [],
            "generated shader mirror(s) have drifted from "
            "pythontk/geo_utils/shadow_horizon.glsl. Never hand-edit between "
            "the markers — edit the source and run "
            "`python m3trik/scripts/sync_shadow_shaders.py`.",
        )

    def test_each_mirror_names_its_source_and_warns_against_editing(self):
        """A mirror a reader cannot tell is generated gets hand-edited once."""
        for path, _ in self.targets:
            text = path.read_text(encoding="utf-8")
            self.assertIn("shadow_horizon.glsl", text, str(path))
            self.assertIn("sync_shadow_shaders", text, str(path))

    def test_the_body_is_declared_as_package_data(self):
        """It ships in the wheel, or Maya and Blender assemble nothing.

        ``check_temp_artifacts.py`` fails an undeclared file in a package tree
        for the same reason, so this catches the omission in the suite rather
        than in the gate.
        """
        root = _repo_root() / "pythontk"
        suffix = Path(ShadowHorizon.SHADER_FILE).suffix
        for name in ("pyproject.toml", "MANIFEST.in"):
            self.assertIn(
                f"*{suffix}",
                (root / name).read_text(encoding="utf-8"),
                f"{name} does not declare {suffix} files",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
