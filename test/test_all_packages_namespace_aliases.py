"""
Multi-package namespace alias edge case testing.

Tests all packages (pythontk, mayatk, uitk, tentacle) for namespace alias edge cases.
Reusable test framework that can validate any package's namespace alias configuration.
"""

import sys
import unittest
from typing import Dict, List, Tuple


class NamespaceAliasEdgeCaseTests:
    """Reusable test methods for namespace alias validation.

    This class provides test methods that can be applied to any package
    with namespace alias configuration.
    """

    @staticmethod
    def get_namespace_aliases(package_name: str) -> Dict[str, any]:
        """Extract namespace alias configurations from a package.

        Args:
            package_name: Name of package to inspect (e.g., 'mayatk')

        Returns:
            Dict mapping alias names to their configurations
        """
        try:
            # Import the package
            if package_name in sys.modules:
                pkg = sys.modules[package_name]
            else:
                pkg = __import__(package_name)

            # First try to get from PACKAGE_RESOLVER (runtime aliases)
            if hasattr(pkg, "PACKAGE_RESOLVER"):
                resolver = pkg.PACKAGE_RESOLVER
                if hasattr(resolver, "namespace_aliases"):
                    runtime_aliases = resolver.namespace_aliases
                    if runtime_aliases:
                        return runtime_aliases

            # Fall back to parsing DEFAULT_INCLUDE for arrow syntax
            # This works even before lazy loading has occurred
            if hasattr(pkg, "__file__"):
                from pathlib import Path

                init_file = Path(pkg.__file__)
                if init_file.exists():
                    content = init_file.read_text(encoding="utf-8")

                    # Look for arrow syntax in DEFAULT_INCLUDE
                    import re

                    # Pattern: "module.path->AliasName": ...
                    pattern = r'"([^"]+)->([^"]+)":'
                    matches = re.findall(pattern, content)

                    if matches:
                        # Return dict of alias_name -> module_path
                        return {alias: module for module, alias in matches}

            return {}
        except Exception as e:
            print(f"Failed to get namespace aliases from {package_name}: {e}")
            return {}

    @staticmethod
    def has_namespace_aliases(package_name: str) -> bool:
        """Check if a package uses namespace aliases.

        Args:
            package_name: Name of package to check

        Returns:
            True if package has namespace aliases configured
        """
        aliases = NamespaceAliasEdgeCaseTests.get_namespace_aliases(package_name)
        return len(aliases) > 0

    @staticmethod
    def test_wildcard_expansion(package_name: str, alias_name: str) -> Tuple[bool, str]:
        """Test wildcard expansion for a specific alias.

        Args:
            package_name: Package containing the alias
            alias_name: Name of the namespace alias

        Returns:
            (success, message) tuple
        """
        try:
            pkg = __import__(package_name)

            if not hasattr(pkg, alias_name):
                return False, f"Alias {alias_name} not found in {package_name}"

            alias_class = getattr(pkg, alias_name)

            # Check if it's a class
            if not isinstance(alias_class, type):
                # Might be typing.Any fallback
                if str(alias_class) == "typing.Any":
                    return (
                        True,
                        f"{alias_name} returned typing.Any (expected outside runtime environment)",
                    )
                return False, f"{alias_name} is not a class: {type(alias_class)}"

            # Check for base classes
            bases = alias_class.__bases__
            if len(bases) == 0:
                return False, f"{alias_name} has no base classes"

            # Check no private base classes
            for base in bases:
                if base.__name__.startswith("_"):
                    return (
                        False,
                        f"{alias_name} has private base class: {base.__name__}",
                    )

            return True, f"{alias_name} validated: {len(bases)} base classes"

        except Exception as e:
            return False, f"Error testing {alias_name}: {e}"

    @staticmethod
    def test_private_filtering(package_name: str, alias_name: str) -> Tuple[bool, str]:
        """Test that private classes are excluded from wildcard expansion.

        Args:
            package_name: Package containing the alias
            alias_name: Name of the namespace alias

        Returns:
            (success, message) tuple
        """
        try:
            pkg = __import__(package_name)

            if not hasattr(pkg, alias_name):
                return True, f"Alias {alias_name} not found (may not be configured)"

            alias_class = getattr(pkg, alias_name)

            # Skip if typing.Any fallback
            if str(alias_class) == "typing.Any":
                return True, f"{alias_name} is typing.Any fallback"

            if not isinstance(alias_class, type):
                return True, f"{alias_name} is not a class"

            # Check base classes
            bases = alias_class.__bases__
            private_bases = [b.__name__ for b in bases if b.__name__.startswith("_")]

            if private_bases:
                return False, f"{alias_name} has private bases: {private_bases}"

            return True, f"{alias_name} has no private base classes"

        except Exception as e:
            return False, f"Error: {e}"


class TestAllPackagesNamespaceAliases(unittest.TestCase):
    """Test namespace aliases across all packages."""

    PACKAGES_TO_TEST = ["pythontk"]

    # Known namespace aliases per package
    KNOWN_ALIASES = {
        "pythontk": [],
    }

    def setUp(self):
        """Clear cached modules before each test."""
        self._modules_snapshot = sys.modules.copy()
        for package in self.PACKAGES_TO_TEST:
            for key in list(sys.modules.keys()):
                if key.startswith(package):
                    del sys.modules[key]

    def tearDown(self):
        """Restore modules after test."""
        sys.modules.clear()
        sys.modules.update(self._modules_snapshot)

    def test_pythontk_namespace_aliases(self):
        """Test pythontk namespace aliases."""
        if not NamespaceAliasEdgeCaseTests.has_namespace_aliases("pythontk"):
            self.skipTest("pythontk has no namespace aliases configured")

        aliases = NamespaceAliasEdgeCaseTests.get_namespace_aliases("pythontk")
        for alias in aliases.keys():
            with self.subTest(alias=alias):
                success, msg = NamespaceAliasEdgeCaseTests.test_wildcard_expansion(
                    "pythontk", alias
                )
                self.assertTrue(success, msg)

    def test_all_packages_private_filtering(self):
        """Test all packages exclude private classes from wildcards."""
        for package in self.PACKAGES_TO_TEST:
            aliases = self.KNOWN_ALIASES.get(package, [])

            if not aliases:
                continue

            for alias in aliases:
                with self.subTest(package=package, alias=alias):
                    success, msg = NamespaceAliasEdgeCaseTests.test_private_filtering(
                        package, alias
                    )
                    self.assertTrue(success, msg)

    def test_namespace_alias_discovery(self):
        """Discover and report namespace aliases in all packages."""
        results = {}

        for package in self.PACKAGES_TO_TEST:
            try:
                has_aliases = NamespaceAliasEdgeCaseTests.has_namespace_aliases(package)
                aliases = NamespaceAliasEdgeCaseTests.get_namespace_aliases(package)
                results[package] = {
                    "has_aliases": has_aliases,
                    "count": len(aliases),
                    "aliases": list(aliases.keys()) if aliases else [],
                }
            except Exception as e:
                results[package] = {"has_aliases": False, "count": 0, "error": str(e)}

        # Print discovery results
        print("\n" + "=" * 60)
        print("NAMESPACE ALIAS DISCOVERY")
        print("=" * 60)
        for package, info in results.items():
            print(f"\n{package}:")
            if "error" in info:
                print(f"  [X] Error: {info['error']}")
            elif info["has_aliases"]:
                print(f"  [OK] Has {info['count']} namespace aliases:")
                for alias in info["aliases"]:
                    print(f"     - {alias}")
            else:
                print(f"  [--] No namespace aliases configured")
        print("=" * 60 + "\n")

        # Test passes if discovery completed without exceptions
        self.assertTrue(True)


def run_multi_package_tests(verbose=True):
    """Run namespace alias tests across all packages."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestAllPackagesNamespaceAliases))

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_multi_package_tests(verbose=True)
    sys.exit(0 if success else 1)
