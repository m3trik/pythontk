# !/usr/bin/python
# coding=utf-8
"""Tests for the module reloader helper."""
from __future__ import annotations

import importlib
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

from pythontk.core_utils.module_reloader import ModuleReloader


class ModuleReloaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self._tmp_path = Path(self._tempdir.name)
        sys.path.insert(0, self._tempdir.name)
        self.addCleanup(self._pop_temp_path)
        self._imported_modules: list[str] = []

    def tearDown(self) -> None:
        for name in self._imported_modules:
            sys.modules.pop(name, None)

    def _pop_temp_path(self) -> None:
        try:
            sys.path.remove(self._tempdir.name)
        except ValueError:
            pass

    def _make_package(
        self,
        package_name: str,
        *,
        init_body: str,
        modules: dict[str, str],
    ) -> types.ModuleType:
        package_root = self._tmp_path / package_name
        package_root.mkdir()
        for relative_module, code in modules.items():
            module_path = package_root / relative_module
            module_path.parent.mkdir(parents=True, exist_ok=True)
            module_path.write_text(textwrap.dedent(code), encoding="utf-8")

        (package_root / "__init__.py").write_text(
            textwrap.dedent(init_body), encoding="utf-8"
        )

        importlib.invalidate_caches()
        module = importlib.import_module(package_name)
        self._imported_modules.extend(
            name
            for name in list(sys.modules)
            if name == package_name or name.startswith(f"{package_name}.")
        )
        return module

    def test_reload_submodules_and_dependencies(self) -> None:
        pkg = self._make_package(
            "reloader_pkg_a",
            init_body="""
                value = "initial"
                from . import dep
            """,
            modules={
                "dep.py": """
                    value = "dep-initial"
                """,
                "child.py": """
                    value = "child-initial"
                """,
            },
        )

        importlib.import_module("reloader_pkg_a.child")

        importlib.import_module("reloader_pkg_a.dep")

        child_mod = sys.modules["reloader_pkg_a.child"]
        dep_mod = sys.modules["reloader_pkg_a.dep"]

        child_mod.value = "child-updated"
        dep_mod.value = "dep-updated"
        sys.modules["reloader_pkg_a"].value = "pkg-updated"

        reloader = ModuleReloader(
            verbose=True, dependencies_first=["reloader_pkg_a.dep"]
        )
        reloaded = reloader.reload("reloader_pkg_a")

        self.assertIn(sys.modules["reloader_pkg_a"], reloaded)
        self.assertEqual(sys.modules["reloader_pkg_a"].value, "initial")
        self.assertEqual(sys.modules["reloader_pkg_a.child"].value, "child-initial")
        self.assertEqual(sys.modules["reloader_pkg_a.dep"].value, "dep-initial")

    def test_reload_skips_missing_modules_when_not_import_missing(self) -> None:
        pkg = self._make_package(
            "reloader_pkg_b",
            init_body="""
                from . import dep
            """,
            modules={
                "dep.py": """
                    value = "dep-initial"
                """,
                "extra.py": """
                    value = "extra-initial"
                """,
            },
        )

        importlib.import_module("reloader_pkg_b.dep")
        sys.modules.pop("reloader_pkg_b.extra", None)

        reloader = ModuleReloader(include_submodules=True, import_missing=False)
        reloaded = reloader.reload("reloader_pkg_b")

        refreshed_names = {mod.__name__ for mod in reloaded}
        self.assertIn("reloader_pkg_b", refreshed_names)
        self.assertIn("reloader_pkg_b.dep", refreshed_names)
        self.assertNotIn("reloader_pkg_b.extra", refreshed_names)

    def test_reload_handles_modules_named_init(self) -> None:
        pkg = self._make_package(
            "reloader_pkg_c",
            init_body="""
                from . import inner
            """,
            modules={
                "inner/__init__.py": """
                    value = "inner-initial"
                """,
            },
        )

        sys.modules["reloader_pkg_c.inner"].value = "inner-updated"

        reloader = ModuleReloader()
        reloader.reload("reloader_pkg_c")

        self.assertEqual(sys.modules["reloader_pkg_c.inner"].value, "inner-initial")

    def test_reload_from_within_target_package(self) -> None:
        pkg = self._make_package(
            "reloader_pkg_d",
            init_body="""
                from pythontk.core_utils.module_reloader import ModuleReloader

                def trigger_reload():
                    reloader = ModuleReloader(include_submodules=True)
                    return reloader.reload(__name__)
            """,
            modules={
                "child.py": """
                    value = "child-initial"
                """,
            },
        )

        importlib.import_module("reloader_pkg_d.child")
        sys.modules["reloader_pkg_d.child"].value = "child-updated"

        result = pkg.trigger_reload()

        self.assertTrue(result)
        self.assertEqual(sys.modules["reloader_pkg_d.child"].value, "child-initial")


if __name__ == "__main__":
    unittest.main()
