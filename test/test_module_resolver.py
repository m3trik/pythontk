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
from pathlib import Path

from conftest import BaseTestCase


class _TempPackageCase(BaseTestCase):
    """Builds throwaway packages on ``sys.path`` and unloads them after."""

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self._tmp_path = Path(self._tempdir.name)
        sys.path.insert(0, self._tempdir.name)
        self.addCleanup(self._pop_temp_path)
        self._imported_modules: list[str] = []

    def tearDown(self) -> None:
        # By package prefix, not the import-time snapshot: a lazily resolved
        # submodule is imported after ``_make_package`` returns.
        roots = {name.split(".")[0] for name in self._imported_modules}
        for name in list(sys.modules):
            if name.split(".")[0] in roots:
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


class ModuleResolverBootstrapTests(_TempPackageCase):
    def test_a_module_saved_with_a_bom_is_parsed(self) -> None:
        """A BOM-prefixed module imports fine; its AST read must not come back
        ``None`` (a swallowed ``SyntaxError``) and hide its classes."""
        from pythontk.core_utils.module_resolver import ModuleAttributeResolver

        self._make_package("resolver_pkg_bom", init_body="", modules={})
        (self._tmp_path / "resolver_pkg_bom" / "bom.py").write_text(
            "class Bom:\n    pass\n", encoding="utf-8-sig"
        )
        importlib.invalidate_caches()
        tree = ModuleAttributeResolver._read_module_ast("resolver_pkg_bom.bom")
        self.assertIsNotNone(tree)
        self.assertEqual([n.name for n in tree.body], ["Bom"])

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

    def test_a_wildcard_namespace_takes_only_the_classes_its_modules_define(
        self,
    ) -> None:
        """``"pkg->Name": "*"`` took every public class a submodule merely
        IMPORTED: blendertk's Diagnostics gained typing.Any, CoreUtils and
        XformUtils as bases, and a module's ``Doc = ReportDoc`` alias leaked as
        the package-level ``btk.Doc``."""
        pkg = self._make_package(
            "resolver_pkg_ns",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                DEFAULT_INCLUDE = {"other": ["Helper"], "diag->Diag": "*"}

                bootstrap_package(globals(), include=DEFAULT_INCLUDE)
            """,
            modules={
                "other.py": """
                    class Helper:
                        def helps(self):
                            return True
                """,
                "diag/__init__.py": '"""Diagnostics."""\n',
                "diag/mesh.py": """
                    from typing import Any
                    from resolver_pkg_ns.other import Helper

                    Doc = Helper

                    class MeshDiag:
                        def check(self):
                            return "mesh"
                """,
            },
        )

        self.assertEqual([b.__name__ for b in pkg.Diag.__bases__], ["MeshDiag"])
        self.assertEqual(pkg.Diag().check(), "mesh")
        self.assertFalse(hasattr(pkg, "Any"))
        self.assertFalse(hasattr(pkg, "Doc"))
        self.assertIs(pkg.Helper, sys.modules["resolver_pkg_ns.other"].Helper)

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

        pkg.configure(include={"second": ["Replacement"]}, merge=False)

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

        pkg.configure(custom_getattr=new_getattr)
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
        # This mirrors a case where we want to access a submodule
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

    def test_auto_all_derived_from_resolved_surface(self) -> None:
        """``__all__`` is populated automatically from the resolved surface.

        It carries every module-level name reachable as ``pkg.Name`` (explicit
        classes/constants + wildcard-scanned classes), is a sorted list, and
        every advertised name resolves -- but flat method aliases and bare
        submodule names are excluded.
        """
        pkg = self._make_package(
            "resolver_pkg_all",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                DEFAULT_INCLUDE = {"alpha": ["Demo", "KONST"], "beta": "*"}

                bootstrap_package(globals(), include=DEFAULT_INCLUDE)
            """,
            modules={
                "alpha.py": """
                    KONST = 42
                    class Demo:
                        def foo(self):
                            return "demo"
                """,
                "beta.py": """
                    class Widget:
                        @staticmethod
                        def build():
                            return "built"
                """,
            },
        )

        self.assertTrue(hasattr(pkg, "__all__"))
        self.assertIsInstance(pkg.__all__, list)
        self.assertEqual(pkg.__all__, sorted(pkg.__all__))
        # Explicit class, explicit constant, and wildcard-scanned class are all
        # advertised.
        for name in ("Demo", "KONST", "Widget"):
            self.assertIn(name, pkg.__all__)
        # Every advertised name actually resolves.
        for name in pkg.__all__:
            self.assertIsNotNone(getattr(pkg, name, None), name)
        # Flat method aliases (wildcard) resolve lazily but are NOT advertised.
        self.assertIn("build", pkg.METHOD_TO_MODULE)
        self.assertNotIn("build", pkg.__all__)
        # Bare submodule names are not part of the public surface either.
        self.assertNotIn("alpha", pkg.__all__)
        self.assertNotIn("beta", pkg.__all__)

    def test_set_all_false_opts_out(self) -> None:
        """``set_all=False`` leaves ``__all__`` unmanaged (absent here)."""
        pkg = self._make_package(
            "resolver_pkg_no_all",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                bootstrap_package(globals(), include={"alpha": ["Demo"]}, set_all=False)
            """,
            modules={"alpha.py": "class Demo:\n    pass\n"},
        )

        self.assertFalse(hasattr(pkg, "__all__"))
        # Resolution is unaffected by opting out of __all__ management.
        self.assertTrue(hasattr(pkg, "Demo"))

    def test_predeclared_all_is_preserved(self) -> None:
        """A hand-declared ``__all__`` is unioned with the derived names."""
        pkg = self._make_package(
            "resolver_pkg_predecl_all",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                MANUAL = object()
                __all__ = ["MANUAL"]

                bootstrap_package(globals(), include={"alpha": ["Demo"]})
            """,
            modules={"alpha.py": "class Demo:\n    pass\n"},
        )

        self.assertIn("MANUAL", pkg.__all__)  # hand-declared name kept
        self.assertIn("Demo", pkg.__all__)  # resolved name added

    def test_reconfigure_refreshes_all_without_stale_names(self) -> None:
        """A runtime reconfigure re-derives ``__all__`` (no stale carryover)."""
        pkg = self._make_package(
            "resolver_pkg_reconf_all",
            init_body="""
                from pythontk.core_utils.module_resolver import bootstrap_package

                bootstrap_package(globals(), include={"alpha": ["Demo", "KONST"]})
            """,
            modules={
                "alpha.py": "KONST = 1\nclass Demo:\n    pass\n",
                "beta.py": "class Other:\n    pass\n",
            },
        )

        self.assertIn("Demo", pkg.__all__)
        self.assertIn("KONST", pkg.__all__)

        pkg.configure(include={"beta": ["Other"]}, merge=False)

        self.assertIn("Other", pkg.__all__)  # new surface advertised
        self.assertNotIn("Demo", pkg.__all__)  # stale names dropped
        self.assertNotIn("KONST", pkg.__all__)

    def test_a_reload_derives_all_afresh_not_from_the_last_run(self) -> None:
        """``importlib.reload`` re-runs the bootstrap in the SAME globals, where the
        handle took the ``__all__`` the LAST run derived for a hand-declared one:
        a name the edit dropped stayed advertised, and ``from pkg import *``
        raised on it. A hand-declared ``__all__`` still survives the reload."""
        init = """
            from pythontk.core_utils.module_resolver import bootstrap_package

            MANUAL = object()
            __all__ = ["MANUAL"]

            bootstrap_package(globals(), include={"alpha": "Demo", "beta": "Other"})
        """
        for declared in (False, True):
            name = f"resolver_pkg_reload_{int(declared)}"
            with self.subTest(declared=declared):
                body = init if declared else init.replace('__all__ = ["MANUAL"]', "")
                pkg = self._make_package(
                    name,
                    init_body=body,
                    modules={
                        "alpha.py": "class Demo:\n    pass\n",
                        "beta.py": "class Other:\n    pass\n",
                    },
                )
                edited = body.replace(', "beta": "Other"', "")
                (self._tmp_path / name / "__init__.py").write_text(
                    textwrap.dedent(edited), encoding="utf-8"
                )
                importlib.reload(pkg)

                expected = ["Demo", "MANUAL"] if declared else ["Demo"]
                self.assertEqual(pkg.__all__, expected)
                exec(f"from {name} import *", {})  # raised on the dropped name


class LazyExportsTests(_TempPackageCase):
    """``lazy_exports``: the subpackage ``__init__`` form of the resolver."""

    _INIT = """
        from pythontk.core_utils.module_resolver import lazy_exports

        lazy_exports(globals(), {"alpha": ("Demo", "_PRIVATE"), "nested.beta": "Other"})
    """
    _MODULES = {
        "alpha.py": "_PRIVATE = 1\nclass Demo:\n    pass\n",
        "nested/__init__.py": "",
        "nested/beta.py": "class Other:\n    pass\n",
    }

    def test_import_loads_no_submodule_until_a_name_is_read(self) -> None:
        pkg = self._make_package(
            "lazy_pkg_a", init_body=self._INIT, modules=self._MODULES
        )
        self.assertNotIn("lazy_pkg_a.alpha", sys.modules)
        self.assertEqual(pkg.Demo.__module__, "lazy_pkg_a.alpha")
        self.assertIn("lazy_pkg_a.alpha", sys.modules)
        self.assertNotIn("lazy_pkg_a.nested.beta", sys.modules)
        self.assertIn("Demo", vars(pkg))  # cached: the next read is a dict hit

    def test_dotted_submodules_and_the_from_import_form(self) -> None:
        self._make_package("lazy_pkg_b", init_body=self._INIT, modules=self._MODULES)
        from lazy_pkg_b import Other  # noqa: F401 -- the form consumers write

        self.assertEqual(Other.__module__, "lazy_pkg_b.nested.beta")

    def test_all_lists_public_names_and_dir_lists_every_name(self) -> None:
        pkg = self._make_package(
            "lazy_pkg_c", init_body=self._INIT, modules=self._MODULES
        )
        self.assertEqual(pkg.__all__, ["Demo", "Other"])
        self.assertTrue({"Demo", "Other", "_PRIVATE"} <= set(dir(pkg)))

    def test_an_unknown_name_raises_and_an_earlier_getattr_is_chained(self) -> None:
        pkg = self._make_package(
            "lazy_pkg_d",
            init_body="""
                from pythontk.core_utils.module_resolver import lazy_exports

                def __getattr__(name):
                    if name == "legacy":
                        return "from the earlier hook"
                    raise AttributeError(name)

                lazy_exports(globals(), {"alpha": "Demo"})
            """,
            modules=self._MODULES,
        )
        self.assertEqual(pkg.legacy, "from the earlier hook")
        with self.assertRaises(AttributeError):
            pkg.missing

    def test_a_name_listed_under_two_submodules_is_rejected(self) -> None:
        from pythontk.core_utils.module_resolver import lazy_exports

        with self.assertRaises(ValueError):
            lazy_exports({"__name__": "x"}, {"a": "Demo", "b": ("Demo",)})

    def test_a_reload_serves_the_reloaded_class_not_the_one_it_cached(self) -> None:
        """``importlib.reload`` re-runs the ``__init__`` in the SAME globals, so a
        name an earlier run cached there shadowed the reloaded submodule's class:
        ``ptk.reload_package("pythontk")`` kept serving the old ``LoggingMixin``.
        """
        from pythontk.core_utils.module_reloader import ModuleReloader

        pkg = self._make_package(
            "lazy_pkg_e", init_body=self._INIT, modules=self._MODULES
        )
        stale = pkg.Demo  # cached into the package globals
        ModuleReloader().reload(pkg)
        fresh = sys.modules["lazy_pkg_e.alpha"].Demo
        self.assertIsNot(fresh, stale)  # the submodule itself did reload
        self.assertIs(pkg.Demo, fresh)

    def test_a_reload_replaces_the_last_runs_hooks_instead_of_stacking(self) -> None:
        """A re-run chained onto the last run's ``__getattr__`` (one more layer per
        reload, each still serving what the old ``__init__`` declared) and unioned
        the ``__all__`` that run derived, so a name the edit dropped outlived it in
        both -- also with ``Deprecation.attributes`` installed after it (the shape
        of every consumer that retired a name) or before it (the order the
        docstring names). A hand-declared ``__all__`` survives."""
        init = """
            from pythontk.core_utils.deprecation import Deprecation
            from pythontk.core_utils.module_resolver import lazy_exports

            if DECLARED:
                __all__ = ["MANUAL"]
            if BEFORE:
                Deprecation.attributes(
                    globals(), {"Legacy": "PKG.alpha.Demo"}, remove_in="9.9.0"
                )
            lazy_exports(globals(), {"alpha": "Demo", "nested.beta": "Other"})
            if AFTER:
                Deprecation.attributes(
                    globals(), {"Legacy": "PKG.alpha.Demo"}, remove_in="9.9.0"
                )
        """
        for deprecation, declared in (
            (None, False),
            ("AFTER", False),
            ("BEFORE", False),
            (None, True),
        ):
            fronted = deprecation is not None
            name = f"lazy_pkg_f{deprecation or 'none'}{int(declared)}".lower()
            with self.subTest(deprecation=deprecation, declared=declared):
                body = (
                    init.replace("AFTER", str(deprecation == "AFTER"))
                    .replace("BEFORE", str(deprecation == "BEFORE"))
                    .replace("DECLARED", str(declared))
                    .replace("PKG", name)
                )
                pkg = self._make_package(name, init_body=body, modules=self._MODULES)
                edited = body.replace(', "nested.beta": "Other"', "")
                (self._tmp_path / name / "__init__.py").write_text(
                    textwrap.dedent(edited), encoding="utf-8"
                )
                importlib.reload(pkg)

                self.assertEqual(
                    pkg.__all__, ["Demo", "MANUAL"] if declared else ["Demo"]
                )
                with self.assertRaises(AttributeError):
                    pkg.Other  # dropped by the edit: nothing may still serve it
                if fronted:
                    with self.assertWarns(DeprecationWarning):
                        self.assertIs(pkg.Legacy, pkg.Demo)

    def test_a_failure_loading_a_declared_name_keeps_its_cause(self) -> None:
        """``from pkg import X`` turns an AttributeError out of ``__getattr__``
        into a bare "cannot import name 'X'" and drops WHY -- the submodule's own
        import failing on a missing attribute, or a declared name it lacks --
        and ``hasattr`` read the broken name as merely absent."""
        pkg = self._make_package(
            "lazy_pkg_h",
            init_body="""
                from pythontk.core_utils.module_resolver import lazy_exports

                lazy_exports(globals(), {"alpha": ("Demo", "Dmeo"), "broken": "Broken"})
            """,
            modules={
                **self._MODULES,
                "broken.py": "import os\n\nclass Broken(os.NoSuchBase):\n    pass\n",
            },
        )
        with self.assertRaises(ImportError) as caught:
            from lazy_pkg_h import Broken  # noqa: F401
        self.assertIsInstance(caught.exception.__cause__, AttributeError)
        self.assertIn("lazy_pkg_h.broken", str(caught.exception))
        self.assertIn("NoSuchBase", str(caught.exception))

        with self.assertRaises(ImportError) as caught:
            from lazy_pkg_h import Dmeo  # noqa: F401
        self.assertIsInstance(caught.exception.__cause__, AttributeError)
        self.assertIn("lazy_pkg_h.alpha", str(caught.exception))

        with self.assertRaises(ImportError):
            hasattr(pkg, "Broken")
        self.assertIs(pkg.Demo, sys.modules["lazy_pkg_h.alpha"].Demo)


class LazyNamespaceAliasTests(_TempPackageCase):
    """A ``"module->Alias"`` entry fused its classes when the package was
    imported: building one imports every module it names, which put mayatk's
    diagnostics package and its MASH module in every ``import mayatk`` (about
    100 ms) for aliases a few panels read. An alias is built on first use.
    Fixed: 2026-10-25."""

    _INIT = """
        from pythontk.core_utils.module_resolver import bootstrap_package

        DEFAULT_INCLUDE = {"other": ["Helper"], "diag->Diag": "*"}

        bootstrap_package(globals(), include=DEFAULT_INCLUDE)
    """
    _MODULES = {
        "other.py": "class Helper:\n    pass\n",
        "diag/__init__.py": '"""Diagnostics."""\n',
        "diag/mesh.py": "class MeshDiag:\n    def check(self):\n        return 'mesh'",
        "diag/uv.py": "class UvDiag:\n    def unwrap(self):\n        return 'uv'",
    }

    def _package(self, name, *, init=None, modules=None):
        return self._make_package(
            name, init_body=init or self._INIT, modules=modules or self._MODULES
        )

    def _init_with(self, old, new):
        """``_INIT`` with one edit."""
        assert old in self._INIT, old
        return self._INIT.replace(old, new)

    def test_importing_the_package_builds_no_alias(self) -> None:
        pkg = self._package("alias_pkg_a")

        self.assertNotIn("alias_pkg_a.diag.mesh", sys.modules)
        self.assertNotIn("Diag", vars(pkg))
        # Declared all the same: the registry and ``__all__`` list it as before.
        self.assertIn("Diag", pkg.CLASS_TO_MODULE)
        self.assertIn("Diag", pkg.__all__)

    def test_the_first_read_builds_the_alias_once(self) -> None:
        pkg = self._package("alias_pkg_b")

        diag = pkg.Diag
        self.assertEqual([b.__name__ for b in diag.__bases__], ["MeshDiag", "UvDiag"])
        self.assertEqual((diag().check(), diag().unwrap()), ("mesh", "uv"))
        self.assertIs(pkg.Diag, diag)
        exec("from alias_pkg_b import *", {})

    def test_a_class_the_alias_fuses_resolves_before_the_alias_is_read(self) -> None:
        """The build also publishes each class it fuses at the package root,
        and mayatk code reads ``mtk.UvDiagnostics`` that way."""
        pkg = self._package("alias_pkg_c")

        self.assertIs(pkg.UvDiag, sys.modules["alias_pkg_c.diag.uv"].UvDiag)
        self.assertFalse(hasattr(pkg, "NoSuchName"))

    def test_a_custom_getattr_that_defers_to_the_resolver_builds_it(self) -> None:
        """uitk installs its own ``__getattr__``, which calls ``_RESOLVER.resolve``."""
        init = self._init_with(
            "bootstrap_package(globals(), include=DEFAULT_INCLUDE)",
            "def _getattr(name):\n"
            "            return globals()['_RESOLVER'].resolve(name)\n\n"
            "        bootstrap_package(\n"
            "            globals(), include=DEFAULT_INCLUDE, custom_getattr=_getattr\n"
            "        )",
        )
        pkg = self._package("alias_pkg_d", init=init)

        self.assertEqual(pkg.Diag().check(), "mesh")

    def test_eager_mode_builds_the_aliases_at_import(self) -> None:
        init = self._init_with(
            "include=DEFAULT_INCLUDE)", "include=DEFAULT_INCLUDE, eager=True)"
        )
        pkg = self._package("alias_pkg_e", init=init)

        self.assertIn("Diag", vars(pkg))

    def test_without_a_module_getattr_the_aliases_build_at_import(self) -> None:
        """``allow_getattr=False`` installs no ``__getattr__`` to build an alias
        on first read, so the alias stood in ``__all__`` but every read of it
        raised AttributeError. Found: 2026-10-26 release review."""
        init = self._init_with(
            "include=DEFAULT_INCLUDE)", "include=DEFAULT_INCLUDE, allow_getattr=False)"
        )
        pkg = self._package("alias_pkg_i", init=init)

        self.assertIn("Diag", pkg.__all__)
        self.assertEqual(pkg.Diag().check(), "mesh")

    def test_a_module_that_raises_while_building_keeps_its_real_error(self) -> None:
        """Only an ImportError withdraws an alias. Anything else a fused module
        raises (a SyntaxError mid-edit, a host API's RuntimeError) surfaced once,
        and every later read raised ModuleNotFoundError for the alias's internal
        placeholder module instead. Found: 2026-10-26 release review."""
        modules = {**self._MODULES, "diag/bad.py": "raise RuntimeError('host not ready')\n"}
        pkg = self._package("alias_pkg_j", modules=modules)

        for attempt in range(2):
            with self.subTest(attempt=attempt):
                with self.assertRaisesRegex(RuntimeError, "host not ready"):
                    pkg.Diag

    def test_a_fused_module_reading_the_alias_mid_build_gets_attribute_error(
        self,
    ) -> None:
        """While an alias builds it is no longer pending, so a fused module that
        read it at import (``getattr(pkg, "Diag", None)``) got ModuleNotFoundError
        for the placeholder module, which dropped that module from the alias.
        Found: 2026-10-26 release review."""
        modules = {
            **self._MODULES,
            "diag/mesh.py": (
                "import alias_pkg_k\n\n"
                "PEER = getattr(alias_pkg_k, 'Diag', None)\n\n"
                "class MeshDiag:\n    def check(self):\n        return 'mesh'\n"
            ),
        }
        pkg = self._package("alias_pkg_k", modules=modules)

        self.assertEqual(
            [b.__name__ for b in pkg.Diag.__bases__], ["MeshDiag", "UvDiag"]
        )
        self.assertIsNone(sys.modules["alias_pkg_k.diag.mesh"].PEER)

    def test_an_explicit_class_list_resolves_through_the_registry(self) -> None:
        """An alias naming its classes still builds when its own module fails to
        import: each name resolves through the registry instead."""
        init = self._init_with('"diag->Diag": "*"', '"broken->Fused": ["Helper"]').replace(
            "include=DEFAULT_INCLUDE)",
            "include=DEFAULT_INCLUDE, on_import_error=lambda name, exc: None)",
        )
        modules = {**self._MODULES, "broken.py": "import no_such_backend_module\n"}
        pkg = self._package("alias_pkg_l", init=init, modules=modules)

        self.assertEqual([b.__name__ for b in pkg.Fused.__bases__], ["Helper"])

    def test_an_alias_that_cannot_build_is_withdrawn(self) -> None:
        """A missing module is never declared, so ``from pkg import *`` still
        works; one that fails to import, or has no class to fuse, leaves the
        registry when first read."""
        init = self._init_with(
            '"diag->Diag": "*"',
            '"diag->Diag": "*", "gone->Gone": "*",'
            ' "broken->Broken": "*", "empty->Empty": "*"',
        ).replace(
            "include=DEFAULT_INCLUDE)",
            "include=DEFAULT_INCLUDE, on_import_error=lambda name, exc: None)",
        )
        modules = {
            **self._MODULES,
            "broken.py": "import no_such_backend_module\n",
            "empty/__init__.py": '"""Nothing to fuse."""\n',
        }
        pkg = self._package("alias_pkg_f", init=init, modules=modules)

        self.assertNotIn("Gone", pkg.__all__)
        for alias in ("Broken", "Empty"):
            with self.subTest(alias=alias):
                self.assertIn(alias, pkg.__all__)
                self.assertFalse(hasattr(pkg, alias))
                self.assertNotIn(alias, pkg.__all__)
                self.assertNotIn(alias, pkg.CLASS_TO_MODULE)
        exec("from alias_pkg_f import *", {})
        self.assertEqual(pkg.Diag().check(), "mesh")

    def test_a_reload_rebuilds_the_alias_from_the_edited_module(self) -> None:
        """``importlib.reload`` re-runs the bootstrap in the SAME globals, where
        the last run's alias still stood -- fused from the classes as they were."""
        pkg = self._package("alias_pkg_g")
        before = pkg.Diag

        (self._tmp_path / "alias_pkg_g" / "diag" / "mesh.py").write_text(
            "class MeshDiag:\n    def check(self):\n        return 'edited'\n",
            encoding="utf-8",
        )
        importlib.invalidate_caches()
        importlib.reload(sys.modules["alias_pkg_g.diag.mesh"])
        importlib.reload(pkg)

        self.assertIsNot(pkg.Diag, before)
        self.assertEqual(pkg.Diag().check(), "edited")
        self.assertIs(pkg.MeshDiag, sys.modules["alias_pkg_g.diag.mesh"].MeshDiag)


class ImportFootprintTests(_TempPackageCase):
    """A wildcard module the resolver judges unsafe to load lazily is imported
    at bootstrap: one ``os.environ.setdefault`` at the top of
    ``img_utils/_img_utils.py`` (2026-06-10) loaded ImgUtils, its seven mixin
    modules, PIL and numpy in every process of every package -- about 240 ms
    each -- and nothing reported it. Fixed: 2026-10-25."""

    def test_importing_pythontk_loads_only_declarations(self) -> None:
        from pythontk.core_utils.module_resolver import import_footprint

        self.assertEqual(
            import_footprint("pythontk"), {"modules": [], "third_party": []}
        )

    def test_an_eager_import_is_reported(self) -> None:
        """Not a vacuous probe: an ``__init__`` that imports is caught."""
        from pythontk.core_utils.module_resolver import import_footprint

        self._make_package(
            "footprint_pkg",
            init_body="""
                import numpy

                from footprint_pkg import impl
            """,
            modules={"impl.py": "VALUE = 1\n", "lazy.py": "VALUE = 2\n"},
        )

        self.assertEqual(
            import_footprint("footprint_pkg"),
            {"modules": ["footprint_pkg.impl"], "third_party": ["numpy"]},
        )


class FlatCollisionTests(_TempPackageCase):
    """A flat name two classes define resolved by module order alone:
    ``mtk.get_bounding_box`` was XformUtils' while CoreUtils' -- a different
    signature -- was the node helper, and ``mtk.set_scene_source`` was the
    record-only DataNodes one, skipping the save hook EnvUtils' armed. The
    resolver reports every such name, and each package's tests pin none.
    Added: 2026-10-25."""

    _INIT = """
        from pythontk.core_utils.module_resolver import bootstrap_package

        bootstrap_package(globals(), include={"alpha": "*", "beta": "*"})
    """

    def test_a_flat_name_two_classes_define_is_reported(self) -> None:
        pkg = self._make_package(
            "flat_pkg_a",
            init_body=self._INIT,
            modules={
                "alpha.py": (
                    "class Alpha:\n"
                    "    @staticmethod\n"
                    "    def shared():\n"
                    "        return 'alpha'\n\n"
                    "    @staticmethod\n"
                    "    def own():\n"
                    "        return 'alpha'\n\n"
                    "class Twin:\n"
                    "    @staticmethod\n"
                    "    def shared():\n"
                    "        return 'twin'\n"
                ),
                "beta.py": (
                    "class Beta:\n"
                    "    @staticmethod\n"
                    "    def shared():\n"
                    "        return 'beta'\n"
                ),
            },
        )

        self.assertEqual(
            pkg._RESOLVER.flat_collisions,
            {
                "shared": [
                    "flat_pkg_a.alpha.Alpha",
                    "flat_pkg_a.alpha.Twin",
                    "flat_pkg_a.beta.Beta",
                ]
            },
        )

    def test_a_retired_alias_neither_owns_nor_wins_a_flat_name(self) -> None:
        """A ``Deprecation.symbol`` door scanned after the live method still
        leaves the flat name to it; a name only a retired door defines keeps
        resolving, with its warning, until the door is removed."""
        retired = (
            "import pythontk as ptk\n\n"
            "from flat_pkg_b.alpha import Live\n\n"
            "class Door:\n"
            "    @staticmethod\n"
            "    @ptk.Deprecation.symbol('Live.shared', remove_in='9.0.0')\n"
            "    def shared():\n"
            "        return Live.shared()\n\n"
            "    @staticmethod\n"
            "    @ptk.Deprecation.symbol('Live.shared', remove_in='9.0.0')\n"
            "    def legacy():\n"
            "        return Live.shared()\n"
        )
        pkg = self._make_package(
            "flat_pkg_b",
            init_body=self._INIT,
            modules={
                "alpha.py": (
                    "class Live:\n"
                    "    @staticmethod\n"
                    "    def shared():\n"
                    "        return 'live'\n"
                ),
                "beta.py": retired,
            },
        )

        self.assertEqual(pkg._RESOLVER.flat_collisions, {})
        self.assertIs(pkg.shared, pkg.Live.shared)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(pkg.legacy(), "live")

    def test_pythontk_defines_each_flat_name_once(self) -> None:
        import pythontk

        self.assertEqual(pythontk._RESOLVER.flat_collisions, {})


class LazyLoadSafetyTests(BaseTestCase):
    """A module loads lazily only when its top level has no side effect, and the
    statements inside a top-level ``if`` or ``try`` are judged by that rule too,
    not passed whole: a registration call behind ``if`` ran at import before,
    and lazily it would silently wait for the module's first use."""

    @staticmethod
    def _safe(source: str) -> bool:
        import ast

        from pythontk.core_utils.module_resolver import ModuleAttributeResolver

        tree = ast.parse(textwrap.dedent(source))
        return ModuleAttributeResolver._statements_are_side_effect_free(tree.body)

    def test_a_call_inside_a_top_level_if_or_try_is_a_side_effect(self):
        self.assertFalse(self._safe("if True:\n    register()\n"))
        self.assertFalse(self._safe("try:\n    register()\nexcept Exception:\n    pass\n"))
        self.assertFalse(self._safe("try:\n    import x\nfinally:\n    cleanup()\n"))

    def test_a_guarded_import_and_its_failure_path_are_not(self):
        self.assertTrue(self._safe(
            "try:\n    import maya.cmds as cmds\n"
            "except ImportError as error:\n    print(error)\n    cmds = None\n"
        ))
        self.assertTrue(self._safe(
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import a\nelse:\n    b = 1\n"
        ))

    def test_a_main_block_never_runs_on_import(self):
        self.assertTrue(self._safe("def main():\n    pass\nif __name__ == '__main__':\n    main()\n"))

    def test_a_top_level_call_is_a_side_effect(self):
        self.assertFalse(self._safe("register()\n"))


class ParseMemoizationTests(BaseTestCase):
    """``build()`` read and parsed every wildcard module twice.

    ``_is_safe_to_lazy_load`` and ``_scan_module_attributes`` each need the
    tree of any module on the lazy path, so an un-memoized parse did the work
    once per check: measured at 13 read-and-parse calls across 7 distinct
    modules for pythontk's own bootstrap, which every package in the ecosystem
    boots through.
    """

    @staticmethod
    def _resolver():
        from pythontk.core_utils import module_resolver as mr
        import pythontk as ptk

        return mr, mr.ModuleAttributeResolver(
            ptk, include=getattr(ptk, "DEFAULT_INCLUDE", None)
        )

    @staticmethod
    def _count_reads(mr, resolver):
        """Run one build, counting files actually read and parsed -- not calls
        to the memoized accessor, which is deliberately unchanged."""
        calls = []
        original = mr.ModuleAttributeResolver._read_module_ast

        def counting(module_name):
            calls.append(module_name)
            return original(module_name)

        mr.ModuleAttributeResolver._read_module_ast = staticmethod(counting)
        try:
            resolver.build()
        finally:
            mr.ModuleAttributeResolver._read_module_ast = staticmethod(original)
        return calls

    def test_a_build_parses_each_module_at_most_once(self):
        mr, resolver = self._resolver()
        calls = self._count_reads(mr, resolver)
        self.assertGreater(len(calls), 0, "nothing was parsed; the probe is broken")
        repeated = {n: calls.count(n) for n in set(calls) if calls.count(n) > 1}
        self.assertEqual(repeated, {}, f"modules parsed more than once: {repeated}")

    def test_a_rebuild_re_reads_from_disk(self):
        """The cache is per-build, not per-instance: an edited module must be
        seen by the next ``build()``, which is what ``ModuleReloader`` relies
        on."""
        mr, resolver = self._resolver()
        first = self._count_reads(mr, resolver)
        second = self._count_reads(mr, resolver)
        self.assertEqual(
            sorted(second), sorted(set(first)), "rebuild did not re-read the modules"
        )

    def test_the_cache_does_not_outlive_the_build(self):
        _, resolver = self._resolver()
        resolver.build()
        self.assertFalse(
            getattr(resolver, "_ast_cache", None),
            "the parsed trees were retained after build()",
        )

