#!/usr/bin/python
# coding=utf-8
"""Guard: the staged RPC cores must not drift from pythontk's source.

``plugin_core.py`` is the one implementation of the in-application RPC server.
The mayatk / blendertk plugin payloads carry a verbatim copy as ``_rpc_core.py``
because they are *installed* into Toolbag / Painter, where pythontk is not
importable. A copy is only safe while it is provably identical, so the copies are
generated (``m3trik/scripts/sync_rpc_core.py``) and this pins them — the same
contract the API registries and the shared ``package-manager.bat`` mirror use.

Skips cleanly outside the monorepo layout (a standalone pythontk checkout has no
sibling packages to stage into).

Run with:
    python -m pytest test_sync_rpc_core.py -v
"""

import importlib.util
import unittest
from pathlib import Path

import pythontk


def _repo_root() -> Path:
    # .../<root>/pythontk/pythontk/__init__.py -> .../<root>
    return Path(pythontk.__file__).resolve().parents[2]


def _load_stager():
    """Import the stager by path — ``m3trik/scripts`` is not an importable package."""
    script = _repo_root() / "m3trik" / "scripts" / "sync_rpc_core.py"
    if not script.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_sync_rpc_core", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStagedCoresInSync(unittest.TestCase):
    def setUp(self):
        self.stager = _load_stager()
        if self.stager is None:
            self.skipTest("m3trik sibling not present (standalone checkout)")
        self.targets = self.stager.mirrors()
        if not self.targets:
            self.skipTest("no consumer packages checked out")

    def test_every_payload_carries_the_current_core(self):
        drift = self.stager.out_of_sync()
        self.assertEqual(
            [str(p) for p in drift],
            [],
            "staged RPC core(s) have drifted from pythontk's plugin_core.py. "
            "Never hand-edit a staged _rpc_core.py — edit the source and run "
            "`python m3trik/scripts/sync_rpc_core.py`.",
        )

    def test_the_source_is_importable_with_no_pythontk_on_the_path(self):
        """The staged copy runs where pythontk does not exist, so it must be stdlib-only.

        An import added to ``plugin_core.py`` would work in-repo and fail only
        once inside Toolbag / Painter — the slowest possible place to find out.
        """
        import ast

        source = self.stager.SOURCE.read_text(encoding="utf-8")
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("pythontk", imported)
        # PySide is resolved lazily inside a try/except by the marshaller, which is
        # correct — but it must never be a hard module-level dependency either.
        self.assertEqual(imported & {"pythontk", "qtpy", "uitk"}, set())

    def test_each_payload_is_self_contained(self):
        """A payload importing a sibling package would break once installed."""
        for staged in self.targets:
            plugin_dir = staged.parent
            for py in plugin_dir.rglob("*.py"):
                if "__pycache__" in py.parts:
                    continue
                text = py.read_text(encoding="utf-8")
                for forbidden in ("import pythontk", "from pythontk"):
                    self.assertNotIn(
                        forbidden,
                        text,
                        f"{py} imports pythontk; an INSTALLED plugin payload has "
                        f"no pythontk on the host's sys.path",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
