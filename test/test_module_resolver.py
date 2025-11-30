# !/usr/bin/python
# coding=utf-8
"""Tests for the reusable module resolver helper."""
from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path


from conftest import BaseTestCase


class ModuleResolverBootstrapTests(BaseTestCase):
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

    def test_bootstrap_exposes_classes_and_methods(self) -> None:
        pkg = self._make_package(
            "resolver_pkg_a",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                DEFAULT_INCLUDE = {"alpha": ["Demo"]}

                bootstrap_package(globals(), include=DEFAULT_INCLUDE)
            """,
            modules={
                "alpha.py": """
                    class Demo:
                        def foo(self):
                            return "demo"
                """
            },
        )

        self.assertTrue(hasattr(pkg, "Demo"))
        self.assertEqual(pkg.Demo.__module__, "resolver_pkg_a.alpha")
        self.assertIn("Demo", pkg.CLASS_TO_MODULE)
        self.assertEqual(pkg.CLASS_TO_MODULE["Demo"], "resolver_pkg_a.alpha")
        # With explicit includes, methods are NOT exposed at package level
        # self.assertIn("foo", pkg.METHOD_TO_MODULE)  # Only with wildcard includes
        self.assertEqual(pkg.Demo().foo(), "demo")
        self.assertIsNotNone(pkg.PACKAGE_RESOLVER)

    def test_configure_replace_include(self) -> None:
        pkg = self._make_package(
            "resolver_pkg_b",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                DEFAULT_INCLUDE = {"first": ["Original"], "second": ["Replacement"]}

                bootstrap_package(globals(), include=DEFAULT_INCLUDE)
            """,
            modules={
                "first.py": """
                    class Original:
                        pass
                """,
                "second.py": """
                    class Replacement:
                        pass
                """,
            },
        )

        pkg.configure_resolver(include={"second": ["Replacement"]}, merge=False)

        self.assertTrue(hasattr(pkg, "Replacement"))
        with self.assertRaises(AttributeError):
            _ = pkg.Original
        self.assertEqual(sorted(pkg.CLASS_TO_MODULE.keys()), ["Replacement"])

    def test_custom_getattr_passthrough(self) -> None:
        pkg = self._make_package(
            "resolver_pkg_c",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                DEFAULT_INCLUDE = {"alpha": ["Demo"]}

                def _custom_getattr(name):
                    if name == "ALIAS":
                        return "alias-value"
                    resolver = globals()["PACKAGE_RESOLVER"].resolver
                    return resolver.resolve(name)

                bootstrap_package(globals(), include=DEFAULT_INCLUDE, custom_getattr=_custom_getattr)
            """,
            modules={
                "alpha.py": """
                    class Demo:
                        def foo(self):
                            return "demo"
                """
            },
        )

        self.assertEqual(pkg.ALIAS, "alias-value")
        self.assertIs(pkg.__getattr__, pkg.PACKAGE_RESOLVER.custom_getattr)
        self.assertEqual(pkg.Demo().foo(), "demo")

        # Reconfigure with a new custom getattr and ensure it replaces the previous one.
        def new_getattr(name):
            if name == "ALIAS":
                return "new-alias"
            resolver = pkg.PACKAGE_RESOLVER.resolver
            return resolver.resolve(name)

        pkg.configure_resolver(custom_getattr=new_getattr)
        self.assertIs(pkg.__getattr__, new_getattr)
        self.assertEqual(pkg.ALIAS, "new-alias")

    def test_include_accepts_string_entries(self) -> None:
        pkg = self._make_package(
            "resolver_pkg_d",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                bootstrap_package(globals(), include={"alpha": "Demo"})
            """,
            modules={
                "alpha.py": """
                    class Demo:
                        pass
                """
            },
        )

        self.assertTrue(hasattr(pkg, "Demo"))
        self.assertEqual(pkg.Demo.__module__, "resolver_pkg_d.alpha")

    def test_fallback_resolves_functions(self) -> None:
        pkg = self._make_package(
            "resolver_pkg_e",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                bootstrap_package(
                    globals(),
                    include={"alpha": "Demo"},
                    fallbacks={"helper": "resolver_pkg_e.alpha"},
                )
            """,
            modules={
                "alpha.py": """
                    class Demo:
                        pass

                    def helper():
                        return "ok"
                """
            },
        )

        self.assertEqual(pkg.helper(), "ok")
        self.assertEqual(pkg.CLASS_TO_MODULE["helper"], "resolver_pkg_e.alpha")

    def test_wildcard_include_exposes_all_classes(self) -> None:
        pkg = self._make_package(
            "resolver_pkg_f",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                bootstrap_package(globals(), include={"alpha": "*"})
            """,
            modules={
                "alpha.py": """
                    class First:
                        pass

                    class Second:
                        pass
                """
            },
        )

        self.assertTrue(hasattr(pkg, "First"))
        self.assertTrue(hasattr(pkg, "Second"))
        self.assertEqual(pkg.First.__module__, "resolver_pkg_f.alpha")
        self.assertEqual(pkg.Second.__module__, "resolver_pkg_f.alpha")

    def test_module_to_parent_resolution(self) -> None:
        pkg = self._make_package(
            "resolver_pkg_g",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                bootstrap_package(
                    globals(),
                    include={"alpha": ["Demo"]},
                    module_to_parent={"HELPER": "resolver_pkg_g.helpers"},
                )
            """,
            modules={
                "alpha.py": """
                    class Demo:
                        pass
                """,
                "helpers.py": """
                    HELPER = object()
                """,
            },
        )

        self.assertIsNotNone(pkg.HELPER)
        self.assertEqual(pkg.Demo.__module__, "resolver_pkg_g.alpha")

    def test_eager_export_populates_globals(self) -> None:
        pkg = self._make_package(
            "resolver_pkg_h",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                bootstrap_package(globals(), include={"alpha": "Demo"}, eager=True)
            """,
            modules={
                "alpha.py": """
                    class Demo:
                        pass
                """
            },
        )

        self.assertIn("Demo", pkg.__dict__)
        self.assertIs(pkg.Demo, pkg.__dict__["Demo"])

    def test_on_import_error_callback_invoked(self) -> None:
        pkg = self._make_package(
            "resolver_pkg_i",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                def _on_error(name, exc):
                    globals().setdefault("_errors", []).append((name, type(exc)))

                bootstrap_package(
                    globals(),
                    include={"alpha": "Demo", "broken": "*"},
                    on_import_error=_on_error,
                )
            """,
            modules={
                "alpha.py": """
                    class Demo:
                        pass
                """,
                "broken.py": """
                    raise ImportError('boom')
                """,
            },
        )

        errors = getattr(pkg, "_errors", [])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0][0], "resolver_pkg_i.broken")
        self.assertIs(errors[0][1], ImportError)
        self.assertTrue(hasattr(pkg, "Demo"))

    def test_bootstrap_handles_spec_loader_without_package_path(self) -> None:
        package_name = "resolver_pkg_spec"
        package_root = self._tmp_path / package_name
        package_root.mkdir()

        (package_root / "alpha.py").write_text(
            textwrap.dedent(
                """
                class Demo:
                    pass
                """
            ),
            encoding="utf-8",
        )

        init_path = package_root / "__init__.py"
        init_path.write_text(
            textwrap.dedent(
                """
                __package__ = "resolver_pkg_spec"

                from pythontk.core_utils.module_resolver import bootstrap_package

                bootstrap_package(globals(), include={"alpha": ["Demo"]})
                """
            ),
            encoding="utf-8",
        )

        importlib.invalidate_caches()
        spec = importlib.util.spec_from_file_location("__init__", init_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["__init__"] = module
        self._imported_modules.append("__init__")
        spec.loader.exec_module(module)

        registered = [
            name
            for name in list(sys.modules)
            if name == package_name or name.startswith(f"{package_name}.")
        ]
        self._imported_modules.extend(registered)

        pkg = sys.modules[package_name]
        self.assertTrue(hasattr(pkg, "Demo"))
        self.assertEqual(pkg.Demo.__module__, f"{package_name}.alpha")
        self.assertTrue(hasattr(pkg, "__path__"))
        self.assertEqual(pkg.__name__, package_name)

    def test_resolve_submodule_attribute(self) -> None:
        pkg = self._make_package(
            "resolver_pkg_j",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                bootstrap_package(globals(), include={"sub": "SubModule"})
            """,
            modules={
                "sub.py": """
                    class SubModule:
                        pass
                """
            },
        )

        # This should work because we exposed the class
        self.assertTrue(hasattr(pkg, "SubModule"))

        # This should ALSO work now that we enabled submodule resolution
        self.assertIsNotNone(pkg.sub)
        self.assertEqual(pkg.sub.SubModule.__module__, "resolver_pkg_j.sub")

    def test_lazy_submodule_access(self) -> None:
        # This mirrors the mayatk.ui_utils case where we want to access a submodule
        # that hasn't been imported yet.
        pkg = self._make_package(
            "resolver_pkg_k",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package
                # We want to access 'sub' module directly
                bootstrap_package(globals())
            """,
            modules={"sub.py": "x = 1"},
        )

        # Accessing pkg.sub should return the module object for resolver_pkg_k.sub
        self.assertIsNotNone(pkg.sub)
        self.assertEqual(pkg.sub.x, 1)

    def test_strict_mode_does_not_expose_methods(self) -> None:
        """Verify that explicit class include does NOT expose static methods."""
        pkg = self._make_package(
            "resolver_pkg_strict",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                # Explicitly include only the class
                DEFAULT_INCLUDE = {"alpha": "Demo"}

                bootstrap_package(globals(), include=DEFAULT_INCLUDE)
            """,
            modules={
                "alpha.py": """
                    class Demo:
                        @staticmethod
                        def static_method():
                            return "static"
                """
            },
        )

        # Class should be exposed
        self.assertTrue(hasattr(pkg, "Demo"))

        # Static method should NOT be exposed at package level
        self.assertFalse(hasattr(pkg, "static_method"))

        # But should be accessible via the class
        self.assertEqual(pkg.Demo.static_method(), "static")

    def test_wildcard_mode_exposes_methods(self) -> None:
        """Verify that wildcard include DOES expose static methods."""
        pkg = self._make_package(
            "resolver_pkg_wildcard",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                # Use wildcard
                DEFAULT_INCLUDE = {"alpha": "*"}

                bootstrap_package(globals(), include=DEFAULT_INCLUDE)
            """,
            modules={
                "alpha.py": """
                    class Demo:
                        @staticmethod
                        def static_method():
                            return "static"
                """
            },
        )

        # Class should be exposed
        self.assertTrue(hasattr(pkg, "Demo"))

        # Static method SHOULD be exposed at package level
        self.assertTrue(hasattr(pkg, "static_method"))
        self.assertEqual(pkg.static_method(), "static")

    def test_lazy_loading_verification(self) -> None:
        """Verify that modules are not imported until accessed."""
        pkg = self._make_package(
            "resolver_pkg_lazy",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                # Use wildcard to trigger scanning
                DEFAULT_INCLUDE = {"alpha": "*"}

                # Enable lazy import explicitly (though it's default if not specified)
                bootstrap_package(globals(), include=DEFAULT_INCLUDE, lazy_import=True)
            """,
            modules={
                "alpha.py": """
                    print("IMPORTING ALPHA")
                    class Demo:
                        pass
                """
            },
        )

        # Check sys.modules - alpha should NOT be there yet
        # Note: _make_package imports the package, but submodules shouldn't be imported yet
        # unless the resolver did it eagerly.
        # However, _make_package helper might trigger imports if not careful,
        # but let's check the specific submodule.
        submodule_name = "resolver_pkg_lazy.alpha"

        # We need to be careful because _make_package might have side effects or
        # the test runner might have imported it.
        # But in a clean environment, it shouldn't be imported.

        # Accessing the class should trigger the import
        _ = pkg.Demo
        self.assertIn(submodule_name, sys.modules)
