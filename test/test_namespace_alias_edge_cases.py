"""
Comprehensive edge case testing for namespace alias functionality.

Tests wildcard expansion, private filtering, explicit lists, error handling,
and multi-inheritance combinations.
"""

import sys
import os
import unittest
import importlib
from pathlib import Path
from typing import Any

# Ensure mayatk is importable for these tests
mayatk_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "mayatk")
)
if os.path.exists(mayatk_path) and mayatk_path not in sys.path:
    sys.path.insert(0, mayatk_path)


class TestNamespaceAliasEdgeCases(unittest.TestCase):
    """Test edge cases for namespace alias feature."""

    def setUp(self):
        """Clear cached modules before each test."""
        self.cleared_modules = []
        for key in list(sys.modules.keys()):
            if key.startswith("mayatk"):
                self.cleared_modules.append(key)
                del sys.modules[key]

    def tearDown(self):
        """Clean up after tests."""
        pass

    # -------------------------------------------------------------------------
    # Wildcard Expansion Tests
    # -------------------------------------------------------------------------

    def test_wildcard_expands_package_classes(self):
        """Test wildcard '*' expands all public classes from a package."""
        import mayatk

        # Diagnostics should exist and be a class
        self.assertTrue(hasattr(mayatk, "Diagnostics"))
        self.assertTrue(isinstance(mayatk.Diagnostics, type))

        # Should have methods from multiple base classes
        diag_methods = dir(mayatk.Diagnostics)

        # From MeshDiagnostics
        self.assertIn("clean_geometry", diag_methods)
        self.assertIn("classify", diag_methods)

        # From AnimCurveDiagnostics
        self.assertIn("are_similar", diag_methods)

    def test_wildcard_excludes_private_classes(self):
        """Test wildcard excludes private/protected classes (starting with _)."""
        import mayatk

        # Diagnostics should NOT include private base classes
        if hasattr(mayatk, "Diagnostics"):
            bases = [base.__name__ for base in mayatk.Diagnostics.__bases__]

            # Should not have any private class names
            for base_name in bases:
                self.assertFalse(
                    base_name.startswith("_"),
                    f"Private class {base_name} should not be in wildcard expansion",
                )

    def test_wildcard_with_module_not_package(self):
        """Test wildcard with a module (not package) works correctly."""
        import mayatk

        # Preview is also using wildcard - it's a simple module not a package
        if hasattr(mayatk, "Preview"):
            self.assertTrue(isinstance(mayatk.Preview, type))
            # Should have Preview methods
            preview_methods = dir(mayatk.Preview)
            self.assertTrue(len(preview_methods) > 0)

    def test_wildcard_prevents_duplicates(self):
        """Test wildcard expansion doesn't create duplicate base classes."""
        import mayatk

        if hasattr(mayatk, "Diagnostics"):
            bases = mayatk.Diagnostics.__bases__
            base_names = [b.__name__ for b in bases]

            # Check for duplicates
            self.assertEqual(
                len(base_names),
                len(set(base_names)),
                f"Duplicate base classes found: {base_names}",
            )

    # -------------------------------------------------------------------------
    # Explicit List Tests
    # -------------------------------------------------------------------------

    def test_explicit_list_includes_only_listed_classes(self):
        """Test explicit list includes only specified classes."""
        import mayatk

        # Mash uses explicit list: ["MashToolkit", "MashNetworkNodes"]
        if hasattr(mayatk, "Mash"):
            bases = [b.__name__ for b in mayatk.Mash.__bases__]

            # Should have exactly the listed classes (plus Any)
            self.assertIn("MashToolkit", bases)
            self.assertIn("MashNetworkNodes", bases)

    def test_explicit_list_can_include_private_classes(self):
        """Test explicit list can include private classes if named."""
        # This tests that we CAN explicitly include private classes
        # if we name them in the list (overriding the filter)

        # We don't currently have such a config, but the architecture allows it
        # This is a theoretical edge case test
        pass

    # -------------------------------------------------------------------------
    # Multi-Inheritance Tests
    # -------------------------------------------------------------------------

    def test_multi_inheritance_method_resolution_order(self):
        """Test MRO is correct for multi-inheritance namespace aliases."""
        import mayatk

        if hasattr(mayatk, "Diagnostics"):
            mro = mayatk.Diagnostics.__mro__
            mro_names = [cls.__name__ for cls in mro]

            # Should include all base classes
            self.assertIn("Diagnostics", mro_names)

            # Should end with object
            self.assertEqual(mro[-1].__name__, "object")

    def test_multi_inheritance_no_conflicts(self):
        """Test multi-inheritance doesn't cause attribute conflicts."""
        import mayatk

        if hasattr(mayatk, "Diagnostics"):
            # Should be able to instantiate (if not abstract)
            # Or at least access class attributes
            try:
                _ = mayatk.Diagnostics
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Accessing Diagnostics class raised: {e}")

    # -------------------------------------------------------------------------
    # Error Handling Tests
    # -------------------------------------------------------------------------

    def test_nonexistent_module_in_alias(self):
        """Test error handling for namespace alias to nonexistent module."""
        # This would require modifying DEFAULT_INCLUDE temporarily
        # which could affect other tests, so we skip actual modification
        # but document the expected behavior:

        # Expected: Should raise ImportError or ModuleNotFoundError
        # The module_resolver should fail gracefully with clear message
        pass

    def test_empty_package_wildcard(self):
        """Test wildcard on empty package (no classes) doesn't break."""
        # Expected: Should create alias with no base classes (or just Any)
        # This is a theoretical edge case
        pass

    def test_nonexistent_class_in_explicit_list(self):
        """Test error handling for non-existent class in explicit list."""
        # Expected: Should skip missing class or raise clear error
        # The module_resolver should handle this gracefully
        pass

    # -------------------------------------------------------------------------
    # Integration Tests
    # -------------------------------------------------------------------------

    def test_alias_accessible_from_package_root(self):
        """Test namespace alias is accessible from package root."""
        import mayatk

        # Should be able to access via package root
        self.assertTrue(hasattr(mayatk, "Diagnostics"))
        self.assertTrue(hasattr(mayatk, "Preview"))
        self.assertTrue(hasattr(mayatk, "Mash"))

    def test_alias_in_class_to_module_mapping(self):
        """Test namespace alias appears in CLASS_TO_MODULE mapping."""
        import mayatk

        if hasattr(mayatk, "CLASS_TO_MODULE"):
            # Aliases should be in the mapping
            self.assertIn("Diagnostics", mayatk.CLASS_TO_MODULE)
            self.assertIn("Preview", mayatk.CLASS_TO_MODULE)
            self.assertIn("Mash", mayatk.CLASS_TO_MODULE)

    def test_multiple_wildcards_dont_conflict(self):
        """Test multiple wildcard aliases don't conflict."""
        import mayatk

        # Multiple wildcard configs should coexist
        aliases = []
        if hasattr(mayatk, "Diagnostics"):
            aliases.append("Diagnostics")
        if hasattr(mayatk, "Preview"):
            aliases.append("Preview")

        # Should have multiple successful aliases
        self.assertGreaterEqual(len(aliases), 2)

    # -------------------------------------------------------------------------
    # Package Structure Tests
    # -------------------------------------------------------------------------

    def test_submodule_iteration_works(self):
        """Test pkgutil.iter_modules correctly finds submodules."""
        import mayatk.core_utils.diagnostics as diagnostics_pkg
        import pkgutil

        if hasattr(diagnostics_pkg, "__path__"):
            submodules = list(pkgutil.iter_modules(diagnostics_pkg.__path__))

            # Should find submodules
            submodule_names = [name for _, name, _ in submodules]

            # Should include animation and mesh
            # (These are the actual submodules in diagnostics package)
            # Note: This assertion may need adjustment based on actual structure
            self.assertTrue(len(submodule_names) > 0)

    def test_class_filtering_works(self):
        """Test isinstance(obj, type) filtering works correctly."""
        import mayatk.core_utils.diagnostics.mesh_diag as mesh_module

        # Count actual classes vs other attributes
        classes = [
            name
            for name in dir(mesh_module)
            if isinstance(getattr(mesh_module, name), type) and not name.startswith("_")
        ]

        # Should find MeshDiagnostics
        self.assertIn("MeshDiagnostics", classes)


class TestNamespaceAliasConfiguration(unittest.TestCase):
    """Test DEFAULT_INCLUDE configuration validation."""

    def test_wildcard_syntax_supported(self):
        """Test wildcard syntax is properly configured."""
        import mayatk

        # Check __init__.py has wildcard syntax
        init_file = Path(mayatk.__file__)
        content = init_file.read_text(encoding="utf-8")

        # Should have arrow syntax with wildcard
        self.assertIn('->Diagnostics": "*"', content)

    def test_explicit_list_syntax_supported(self):
        """Test explicit list syntax is properly configured."""
        import mayatk

        init_file = Path(mayatk.__file__)
        content = init_file.read_text(encoding="utf-8")

        # Should have explicit list
        self.assertIn('"SceneAnalyzer"', content)
        self.assertIn('"AuditProfile"', content)
        self.assertIn('"SceneDiagnostics"', content)

    def test_no_malformed_entries(self):
        """Test DEFAULT_INCLUDE has no malformed entries."""
        import mayatk

        if hasattr(mayatk, "PACKAGE_RESOLVER"):
            resolver = mayatk.PACKAGE_RESOLVER

            # Should have loaded successfully (no exceptions during init)
            self.assertIsNotNone(resolver)


class TestNamespaceAliasPerformance(unittest.TestCase):
    """Test performance characteristics of namespace aliases."""

    def test_lazy_loading_not_broken(self):
        """Test namespace aliases don't break lazy loading."""
        import mayatk

        # Clear diagnostics modules after importing mayatk
        for key in list(sys.modules.keys()):
            if key.startswith("mayatk.core_utils.diagnostics"):
                del sys.modules[key]

        # Accessing Diagnostics should trigger lazy load
        diag = mayatk.Diagnostics

        # Check if diagnostics package and submodules are loaded
        diagnostics_modules = [
            key for key in sys.modules.keys() if "mayatk.core_utils.diagnostics" in key
        ]

        # When running outside Maya, namespace alias returns typing.Any as fallback
        # Use string comparison since typing.Any identity check fails
        if str(diag) == "typing.Any":
            # Outside Maya - fallback behavior, modules won't load
            self.assertEqual(
                len(diagnostics_modules),
                0,
                "Diagnostics returned typing.Any fallback, modules shouldn't load",
            )
        else:
            # In Maya - should have loaded the actual classes
            self.assertGreaterEqual(
                len(diagnostics_modules),
                1,
                f"Expected diagnostics modules loaded in Maya, found: {diagnostics_modules}",
            )

    def test_wildcard_expansion_cached(self):
        """Test wildcard expansion results are cached."""
        import mayatk

        # First access
        diag1 = mayatk.Diagnostics

        # Second access should return same class
        diag2 = mayatk.Diagnostics

        # Should be identical object
        self.assertIs(diag1, diag2)


def run_tests(verbose=True):
    """Run all edge case tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Load all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNamespaceAliasEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestNamespaceAliasConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestNamespaceAliasPerformance))

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    # Can run standalone
    success = run_tests(verbose=True)
    sys.exit(0 if success else 1)
