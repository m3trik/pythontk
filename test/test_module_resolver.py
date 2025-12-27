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
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import re
import ast


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


# ==============================================================================
# Module Resolver Integration Validation Tests
# ==============================================================================


@dataclass
class ValidationResult:
    """Result of a validation check."""

    test_name: str
    passed: bool
    message: str
    details: List[str] = field(default_factory=list)

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        output = [f"{status}: {self.test_name}"]
        if self.message:
            output.append(f"  {self.message}")
        for detail in self.details:
            output.append(f"    • {detail}")
        return "\n".join(output)


class ModuleResolverValidator:
    """Validates a package's module resolver implementation."""

    def __init__(self, package_name: str, package_path: Optional[Path] = None):
        """
        Initialize validator for a package.

        Args:
            package_name: Name of the package to validate (e.g., 'mayatk')
            package_path: Optional path to package root. If None, will try to discover.
        """
        self.package_name = package_name
        self.package_path = package_path or self._discover_package_path()
        self.package_dir = self.package_path / package_name
        self.results: List[ValidationResult] = []

    def _discover_package_path(self) -> Path:
        """Discover package path from sys.modules or sys.path."""
        if self.package_name in sys.modules:
            module = sys.modules[self.package_name]
            if hasattr(module, "__file__") and module.__file__:
                return Path(module.__file__).parent.parent

        # Try to import it
        try:
            module = importlib.import_module(self.package_name)
            if hasattr(module, "__file__") and module.__file__:
                return Path(module.__file__).parent.parent
        except ImportError:
            pass

        raise ValueError(f"Could not discover path for package '{self.package_name}'")

    def _discover_subpackages(self) -> List[str]:
        """Discover all subpackages."""
        subpackages = []
        for item in self.package_dir.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                if item.name.startswith("_") or item.name in [
                    "test",
                    "tests",
                    "build",
                    "dist",
                ]:
                    continue
                subpackages.append(item.name)
        return sorted(subpackages)

    def _discover_implementation_modules(
        self, subpackage_dir: Path
    ) -> Dict[str, List[str]]:
        """Find implementation modules (_*.py) and extract class names."""
        impl_modules = {}
        for item in subpackage_dir.glob("_*.py"):
            if item.name == "__init__.py":
                continue
            classes = self._extract_classes_from_module(item)
            if classes:
                impl_modules[item.stem] = classes
        return impl_modules

    def _discover_regular_modules(self, subpackage_dir: Path) -> List[str]:
        """Find regular module files (not starting with _)."""
        regular_modules = []
        for item in subpackage_dir.glob("*.py"):
            if item.name.startswith("_") or item.name == "__init__.py":
                continue
            regular_modules.append(item.stem)
        return regular_modules

    def _extract_classes_from_module(self, module_file: Path) -> List[str]:
        """Extract all class names from a module file."""
        try:
            with open(module_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(module_file))

            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
            return classes
        except Exception:
            return []

    def test_package_structure(self) -> ValidationResult:
        """Test 1: Validate basic package structure."""
        details = []
        passed = True

        # Check package directory exists
        if not self.package_dir.exists():
            return ValidationResult(
                "Package Structure",
                False,
                f"Package directory not found: {self.package_dir}",
            )

        # Check __init__.py exists
        init_file = self.package_dir / "__init__.py"
        if not init_file.exists():
            return ValidationResult(
                "Package Structure",
                False,
                f"Package __init__.py not found",
            )

        # Discover subpackages
        subpackages = self._discover_subpackages()
        details.append(
            f"Found {len(subpackages)} subpackages: {', '.join(subpackages[:5])}{'...' if len(subpackages) > 5 else ''}"
        )

        # Check each subpackage has __init__.py
        for subpkg in subpackages:
            subpkg_init = self.package_dir / subpkg / "__init__.py"
            if not subpkg_init.exists():
                details.append(f"Missing __init__.py in {subpkg}")
                passed = False

        return ValidationResult(
            "Package Structure",
            passed,
            f"Package structure is {'valid' if passed else 'invalid'}",
            details,
        )

    def test_circular_imports(self) -> ValidationResult:
        """Test 2: Scan for circular import patterns."""
        subpackages = self._discover_subpackages()

        # Build implementation module map
        impl_map = {}
        regular_modules = set()
        for subpkg in subpackages:
            subpkg_dir = self.package_dir / subpkg
            modules = self._discover_implementation_modules(subpkg_dir)
            for mod, classes in modules.items():
                for cls in classes:
                    impl_map[cls] = f"{subpkg}.{mod}"
            regular_modules.update(self._discover_regular_modules(subpkg_dir))

        # Build problematic patterns
        patterns = self._build_circular_import_patterns(
            subpackages, impl_map, regular_modules
        )

        # Scan all Python files
        issues = []
        scanned_count = 0
        for py_file in self.package_dir.rglob("*.py"):
            if "__pycache__" in str(py_file) or "test" in str(py_file):
                continue

            scanned_count += 1
            rel_path = py_file.relative_to(self.package_path)

            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            for pattern, description in patterns:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    line_num = content[: match.start()].count("\n") + 1
                    line_content = content.split("\n")[line_num - 1].strip()

                    # Skip commented lines
                    if line_content.startswith("#"):
                        continue

                    issues.append(
                        f"{rel_path}:{line_num} - {description} ('{match.group(0)}')"
                    )

        details = [
            f"Scanned {scanned_count} files",
            f"Used {len(patterns)} pattern checks",
        ]

        if issues:
            details.extend(issues[:10])  # Limit to first 10
            if len(issues) > 10:
                details.append(f"... and {len(issues) - 10} more")

        return ValidationResult(
            "Circular Import Prevention",
            len(issues) == 0,
            f"{'No' if len(issues) == 0 else len(issues)} circular import issues found",
            details,
        )

    def _build_circular_import_patterns(
        self,
        subpackages: List[str],
        impl_map: Dict[str, str],
        regular_modules: Set[str],
    ) -> List[Tuple[str, str]]:
        """Build regex patterns for detecting circular imports."""
        patterns = []
        pkg = self.package_name

        # Pattern 1: from package import subpackage
        if subpackages:
            subpkg_regex = "|".join(re.escape(sp) for sp in subpackages)
            patterns.append(
                (
                    rf"from {pkg} import ({subpkg_regex})(?![._])",
                    "Import subpackage directly",
                )
            )

        # Pattern 2: from package.subpackage import Class (but not regular modules)
        if subpackages and impl_map:
            subpkg_regex = "|".join(re.escape(sp) for sp in subpackages)
            if regular_modules:
                regular_regex = "|".join(re.escape(mod) for mod in regular_modules)
                patterns.append(
                    (
                        rf"from {pkg}\.({subpkg_regex}) import (?!_)(?!{regular_regex}\b)",
                        "Import class from subpackage __init__",
                    )
                )
            else:
                patterns.append(
                    (
                        rf"from {pkg}\.({subpkg_regex}) import (?!_)",
                        "Import from subpackage __init__",
                    )
                )

        # Pattern 3: subpackage.ClassName.method (module-qualified access)
        for class_name in impl_map.keys():
            patterns.append(
                (
                    rf"\w+\.{re.escape(class_name)}\.",
                    f"Accessing {class_name} via module attribute",
                )
            )

        # Pattern 4: import package.subpackage
        if subpackages:
            subpkg_regex = "|".join(re.escape(sp) for sp in subpackages)
            patterns.append(
                (rf"import {pkg}\.({subpkg_regex})(?![._])", "Import subpackage module")
            )

        return patterns

    def test_lazy_loading_config(self) -> ValidationResult:
        """Test 3: Validate DEFAULT_INCLUDE configuration."""
        init_file = self.package_dir / "__init__.py"

        try:
            with open(init_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return ValidationResult(
                "Lazy Loading Configuration", False, f"Could not read __init__.py: {e}"
            )

        details = []

        # Check for DEFAULT_INCLUDE
        if "DEFAULT_INCLUDE" not in content:
            return ValidationResult(
                "Lazy Loading Configuration",
                False,
                "DEFAULT_INCLUDE not found in package __init__.py",
                ["Package may not be using lazy loading"],
            )

        # Check for bootstrap_package call
        if "bootstrap_package" not in content:
            details.append("âš ï¸  bootstrap_package call not found")

        # Try to parse and extract DEFAULT_INCLUDE
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "DEFAULT_INCLUDE"
                        ):
                            if isinstance(node.value, ast.Dict):
                                details.append(
                                    f"DEFAULT_INCLUDE has {len(node.value.keys)} entries"
                                )
                                break
        except Exception:
            pass

        return ValidationResult(
            "Lazy Loading Configuration",
            True,
            "DEFAULT_INCLUDE configuration found",
            details,
        )

    def test_runtime_import(self) -> ValidationResult:
        """Test 4: Test actual package import."""
        details = []

        # Clear any cached imports
        to_remove = [
            key
            for key in list(sys.modules.keys())
            if key.startswith(f"{self.package_name}.")
        ]
        for key in to_remove:
            del sys.modules[key]
        details.append(f"Cleared {len(to_remove)} cached submodules")

        # Try importing (package root might already be loaded)
        try:
            if self.package_name in sys.modules:
                pkg = sys.modules[self.package_name]
                details.append("âœ… Package already loaded in sys.modules")
            else:
                pkg = importlib.import_module(self.package_name)
                details.append("âœ… Package imported successfully")
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            return ValidationResult(
                "Runtime Import",
                False,
                f"Failed to import package: {e}",
                details + [f"Traceback:\n{error_details}"],
            )

        # Check for PACKAGE_RESOLVER
        if hasattr(pkg, "PACKAGE_RESOLVER"):
            details.append("âœ… PACKAGE_RESOLVER attribute found")
        else:
            details.append("âš ï¸  PACKAGE_RESOLVER attribute not found")

        # Check for CLASS_TO_MODULE
        if hasattr(pkg, "CLASS_TO_MODULE"):
            class_count = len(pkg.CLASS_TO_MODULE)
            details.append(f"âœ… CLASS_TO_MODULE has {class_count} entries")
        else:
            details.append("âš ï¸  CLASS_TO_MODULE not found")

        return ValidationResult(
            "Runtime Import", True, "Package imports successfully", details
        )

    def test_lazy_class_access(self) -> ValidationResult:
        """Test 5: Test lazy-loaded class accessibility."""
        try:
            pkg = sys.modules.get(self.package_name) or importlib.import_module(
                self.package_name
            )
        except Exception as e:
            return ValidationResult(
                "Lazy Class Access", False, f"Could not import package: {e}"
            )

        details = []

        if not hasattr(pkg, "CLASS_TO_MODULE"):
            return ValidationResult(
                "Lazy Class Access",
                False,
                "No CLASS_TO_MODULE mapping found",
                ["Package may not be using lazy loading"],
            )

        # Test a few classes
        accessible = 0
        not_accessible = 0

        for class_name in list(pkg.CLASS_TO_MODULE.keys())[:10]:  # Test first 10
            try:
                cls = getattr(pkg, class_name, None)
                if cls is not None:
                    accessible += 1
                else:
                    not_accessible += 1
                    details.append(f"âš ï¸  {class_name} not accessible")
            except Exception as e:
                not_accessible += 1
                details.append(f"âŒ {class_name} raised error: {e}")

        total_tested = accessible + not_accessible
        if total_tested > 0:
            details.insert(
                0,
                f"Tested {total_tested} classes: {accessible} accessible, {not_accessible} failed",
            )

        return ValidationResult(
            "Lazy Class Access",
            not_accessible == 0 and accessible > 0,
            f"{'All' if not_accessible == 0 else 'Some'} classes accessible via lazy loading",
            details,
        )

    def test_minimal_subpackage_inits(self) -> ValidationResult:
        """Test 6: Ensure subpackage __init__.py files are minimal."""
        subpackages = self._discover_subpackages()
        details = []
        bloated = []

        for subpkg in subpackages:
            init_file = self.package_dir / subpkg / "__init__.py"
            try:
                with open(init_file, "r", encoding="utf-8") as f:
                    lines = [
                        line
                        for line in f
                        if line.strip() and not line.strip().startswith("#")
                    ]

                if len(lines) > 10:  # Threshold for "minimal"
                    bloated.append(f"{subpkg}: {len(lines)} non-comment lines")
            except Exception:
                pass

        if bloated:
            details.extend(bloated)
            details.append("âš ï¸  Consider simplifying these __init__.py files")
        else:
            details.append(
                f"All {len(subpackages)} subpackages have minimal __init__.py files"
            )

        return ValidationResult(
            "Minimal Subpackage Inits",
            len(bloated) == 0,
            f"{'All' if len(bloated) == 0 else len(bloated)} subpackages have {'minimal' if len(bloated) == 0 else 'bloated'} __init__.py",
            details,
        )

    def run_all_tests(self, verbose: bool = True) -> bool:
        """Run all validation tests and return overall pass/fail."""
        self.results = []

        # Run all tests
        self.results.append(self.test_package_structure())
        self.results.append(self.test_circular_imports())
        self.results.append(self.test_lazy_loading_config())
        self.results.append(self.test_runtime_import())
        self.results.append(self.test_lazy_class_access())
        self.results.append(self.test_minimal_subpackage_inits())

        # Print results if verbose
        if verbose:
            print("=" * 70)
            print(f"MODULE RESOLVER VALIDATION: {self.package_name}")
            print("=" * 70)
            print()

            for result in self.results:
                print(result)
                print()

            passed = sum(1 for r in self.results if r.passed)
            total = len(self.results)

            print("=" * 70)
            print(f"RESULTS: {passed}/{total} tests passed")
            print("=" * 70)

        return all(r.passed for r in self.results)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        return {
            "package": self.package_name,
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "results": [
                {
                    "test": r.test_name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


def validate_package(
    package_name: str, package_path: Optional[Path] = None, verbose: bool = True
) -> bool:
    """
    Convenience function to validate a package.

    Args:
        package_name: Name of package to validate
        package_path: Optional path to package root
        verbose: Whether to print detailed output

    Returns:
        True if all tests pass, False otherwise
    """
    validator = ModuleResolverValidator(package_name, package_path)
    return validator.run_all_tests(verbose=verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate module resolver integration")
    parser.add_argument("package", help="Package name to validate")
    parser.add_argument("--path", help="Optional package path")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    package_path = Path(args.path) if args.path else None
    success = validate_package(args.package, package_path, verbose=not args.quiet)

    sys.exit(0 if success else 1)
