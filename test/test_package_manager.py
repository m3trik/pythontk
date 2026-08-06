#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk PackageManager.

Run with:
    python -m pytest test_package_manager.py -v
    python test_package_manager.py
"""
import sys
import tempfile
import os
import unittest

from pythontk.core_utils.package_manager import (
    PackageManager,
    _PkgVersionCheck,
    _PkgVersionUtils,
)

from conftest import BaseTestCase


class PackageManagerTest(BaseTestCase):
    """Tests for PackageManager class."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.pkg_mgr = PackageManager()

    def test_init_default_python_path(self):
        """Test initialization with default python path."""
        self.assertEqual(self.pkg_mgr.python_path, sys.executable)

    def test_init_custom_python_path(self):
        """Test initialization with custom python path."""
        custom_path = "/custom/python"
        pkg_mgr = PackageManager(python_path=custom_path)
        self.assertEqual(pkg_mgr.python_path, custom_path)

    def test_parse_command_string(self):
        """Test parsing command from string."""
        command = "install numpy --upgrade"
        result = self.pkg_mgr._parse_command(command)

        self.assertIsInstance(result, list)
        self.assertIn("install", result)
        self.assertIn("numpy", result)

    def test_parse_command_list(self):
        """Test parsing command from list (passthrough)."""
        command = ["install", "numpy"]
        result = self.pkg_mgr._parse_command(command)

        self.assertEqual(result, command)

    def test_is_informational_message(self):
        """Test detecting informational messages."""
        info_msg = "A new release of pip available: 23.0 -> 24.0"
        error_msg = "ERROR: Package not found"

        self.assertTrue(self.pkg_mgr._is_informational_message(info_msg))
        self.assertFalse(self.pkg_mgr._is_informational_message(error_msg))

    def test_is_informational_message_modern_pip_wordings(self):
        """pip >= 22 says 'A new release of pip IS available' (and prefixes
        [notice]); the self-check can also fail with a WARNING. All are noise
        from successful runs and must not contaminate parseable output."""
        for msg in (
            "[notice] A new release of pip is available: 23.2.1 -> 26.2.1",
            "[notice] To update, run: python.exe -m pip install --upgrade pip",
            "WARNING: There was an error checking the latest version of pip.",
        ):
            self.assertTrue(self.pkg_mgr._is_informational_message(msg), msg)

    def test_list_packages_survives_stderr_noise(self):
        """list_packages must return {name: version} even when pip emits
        stderr noise a filter pattern doesn't anticipate."""
        from unittest.mock import patch

        raw = (
            '[{"name": "uitk", "version": "1.3.54"}, '
            '{"name": "tentacletk", "version": "0.13.42"}]'
            "\nError:\nWARNING: unforeseen proxy chatter"
            "\n[some bracketed noise line]"  # must not extend/derail the parse
        )
        with patch.object(PackageManager, "pip", return_value=raw):
            result = self.pkg_mgr.list_packages()
        self.assertEqual(
            result, {"uitk": "1.3.54", "tentacletk": "0.13.42"}
        )

    def test_latest_versions_isolates_a_failing_lookup(self):
        """One unreachable package must not lose the other answers, and must
        report None rather than a version — a caller comparing installed !=
        latest would otherwise read the failure as "outdated"."""
        from unittest.mock import patch

        def fake(name, timeout=None):
            if name == "boom":
                raise RuntimeError("index unreachable")
            return "1.0"

        with patch.object(PackageManager, "latest_version", side_effect=fake):
            result = self.pkg_mgr.latest_versions(["a", "boom", "c"])

        self.assertEqual(result, {"a": "1.0", "boom": None, "c": "1.0"})

    def test_latest_versions_queries_concurrently(self):
        """Sequential lookups multiply the socket timeout by the package count
        (5 x 10s of a frozen DCC UI); they must overlap."""
        import threading
        import time
        from unittest.mock import patch

        active, peak, lock = 0, [0], threading.Lock()

        def fake(name, timeout=None):
            nonlocal active
            with lock:
                active += 1
                peak[0] = max(peak[0], active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return "1.0"

        with patch.object(PackageManager, "latest_version", side_effect=fake):
            self.pkg_mgr.latest_versions(["a", "b", "c", "d", "e"])

        self.assertGreater(peak[0], 1, "lookups ran one at a time")

    def test_list_packages_distinguishes_empty_env_from_unparseable_output(self):
        """An empty environment legitimately prints ``[]``. Output with no JSON
        array at all is a FAILURE, and returning {} for it would tell every
        caller that nothing is installed — the ecosystem updater would then
        offer to (re)install all five packages."""
        from unittest.mock import patch

        with patch.object(PackageManager, "pip", return_value="[]"):
            self.assertEqual(self.pkg_mgr.list_packages(), {})  # genuinely empty

        with patch.object(PackageManager, "pip", return_value="ERROR: pip exploded"):
            with self.assertRaises(RuntimeError):
                self.pkg_mgr.list_packages()

    def test_latest_version_passes_a_socket_timeout(self):
        """Without a timeout, urlopen blocks on the OS default (forever). This
        runs on the UI thread of a DCC — an unreachable/slow index would hang
        Maya or Blender rather than reporting a failed check, and the caller
        asks once per ecosystem package."""
        from unittest.mock import patch, MagicMock

        response = MagicMock()
        response.read.return_value = b'{"info": {"version": "1.2.3"}}'
        response.__enter__ = lambda s: s
        response.__exit__ = lambda *a: False

        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            self.assertEqual(self.pkg_mgr.latest_version("uitk"), "1.2.3")

        _args, kwargs = urlopen.call_args
        self.assertIn("timeout", kwargs, "urlopen must be given a timeout")
        self.assertGreater(kwargs["timeout"], 0)

    def test_is_list_format(self):
        """Test detecting list format output."""
        list_lines = [
            "Package    Version",
            "---------- -------",
            "numpy      1.24.0",
            "pandas     2.0.0",
        ]
        self.assertTrue(self.pkg_mgr._is_list_format(list_lines))

    def test_parse_list_format(self):
        """Test parsing list format."""
        lines = [
            "Package    Version",
            "---------- -------",
            "numpy      1.24.0",
            "pandas     2.0.0",
        ]
        result = self.pkg_mgr._parse_list_format(lines)

        self.assertEqual(result["numpy"], "1.24.0")
        self.assertEqual(result["pandas"], "2.0.0")

    def test_is_key_value_format(self):
        """Test detecting key-value format."""
        kv_lines = [
            "Name: numpy",
            "Version: 1.24.0",
            "Summary: NumPy is the fundamental package",
        ]
        self.assertTrue(self.pkg_mgr._is_key_value_format(kv_lines))

    def test_parse_key_value_format(self):
        """Test parsing key-value format."""
        lines = [
            "Name: numpy",
            "Version: 1.24.0",
            "Location: /usr/lib/python",
        ]
        result = self.pkg_mgr._parse_key_value_format(lines)

        self.assertEqual(result["name"], "numpy")
        self.assertEqual(result["version"], "1.24.0")

    def test_convert_output_json(self):
        """Test converting JSON output."""
        json_output = '[{"name": "numpy", "version": "1.24.0"}]'
        result = self.pkg_mgr._convert_output(json_output)

        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["name"], "numpy")

    def test_process_output_key_value_with_headers(self):
        """Test processing key-value output with proper headers (like pip show)."""
        # When output has "---", list format is detected first, but parse_list looks for dashes
        # Let's test the key-value format that doesn't match list format
        stdout = "Name:test\nVersion:1.0.0"  # No space after colon
        stderr = ""

        result = self.pkg_mgr._process_output(stdout, stderr, output_as_string=False)
        # This should be parsed as key-value
        self.assertIsInstance(result, dict)

    def test_process_output_as_string(self):
        """Test pip command with string output."""
        stdout = "Package    Version\n---------- -------\nnumpy      1.0.0"
        stderr = ""

        result = self.pkg_mgr._process_output(stdout, stderr, output_as_string=True)
        self.assertIsInstance(result, str)
        self.assertIn("numpy", result)


class PkgVersionCheckTest(BaseTestCase):
    """Tests for _PkgVersionCheck class."""

    def test_init(self):
        """Test initialization."""
        checker = _PkgVersionCheck(package_name="test", python_path="/usr/bin/python")

        self.assertEqual(checker._package_name, "test")
        self.assertEqual(checker._python_path, "/usr/bin/python")

    def test_init_default_python_path(self):
        """Test initialization with default python path."""
        checker = _PkgVersionCheck(package_name="test")

        self.assertEqual(checker._python_path, sys.executable)

    def test_new_version_available_no_versions(self):
        """Test new_version_available with no versions set."""
        checker = _PkgVersionCheck()
        self.assertFalse(checker.new_version_available)

    def test_new_version_available_same_version(self):
        """Test new_version_available when versions are same."""
        checker = _PkgVersionCheck()
        checker._installed_ver = "1.0.0"
        checker._latest_ver = "1.0.0"

        self.assertFalse(checker.new_version_available)

    def test_new_version_available_different_versions(self):
        """Test new_version_available when versions differ."""
        checker = _PkgVersionCheck()
        checker._installed_ver = "1.0.0"
        checker._latest_ver = "1.1.0"

        self.assertTrue(checker.new_version_available)

    def test_start_version_check_no_package_raises(self):
        """Test start_version_check without package name raises."""
        checker = _PkgVersionCheck()

        with self.assertRaises(ValueError):
            checker.start_version_check()


class PkgVersionUtilsTest(BaseTestCase):
    """Tests for _PkgVersionUtils class."""

    def test_update_version_increment_patch(self):
        """Test incrementing patch version."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('__version__ = "1.0.0"\n')
            filepath = f.name

        try:
            result = _PkgVersionUtils.update_version(
                filepath, change="increment", version_part="patch"
            )
            self.assertEqual(result, "1.0.1")
        finally:
            os.unlink(filepath)

    def test_update_version_increment_minor(self):
        """Test incrementing minor version."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('__version__ = "1.0.5"\n')
            filepath = f.name

        try:
            result = _PkgVersionUtils.update_version(
                filepath, change="increment", version_part="minor"
            )
            self.assertEqual(result, "1.1.5")
        finally:
            os.unlink(filepath)

    def test_update_version_increment_major(self):
        """Test incrementing major version."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('__version__ = "1.5.3"\n')
            filepath = f.name

        try:
            result = _PkgVersionUtils.update_version(
                filepath, change="increment", version_part="major"
            )
            self.assertEqual(result, "2.5.3")
        finally:
            os.unlink(filepath)

    def test_update_version_decrement_patch(self):
        """Test decrementing patch version."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('__version__ = "1.0.5"\n')
            filepath = f.name

        try:
            result = _PkgVersionUtils.update_version(
                filepath, change="decrement", version_part="patch"
            )
            self.assertEqual(result, "1.0.4")
        finally:
            os.unlink(filepath)

    def test_update_version_no_version_found(self):
        """Test update_version when no version found."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('print("no version here")\n')
            filepath = f.name

        try:
            result = _PkgVersionUtils.update_version(filepath)
            self.assertEqual(result, "")
        finally:
            os.unlink(filepath)

    def test_update_version_invalid_change_raises(self):
        """Test update_version with invalid change parameter raises."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('__version__ = "1.0.0"\n')
            filepath = f.name

        try:
            with self.assertRaises(ValueError):
                _PkgVersionUtils.update_version(
                    filepath, change="invalid", version_part="patch"
                )
        finally:
            os.unlink(filepath)


class PackageManagerTomllibImportTest(BaseTestCase):
    """Regression: module must import on Python < 3.11 (no stdlib ``tomllib``).

    Previously a module-level ``import tomllib`` (Python 3.11+ only) crashed
    ``import pythontk.core_utils.package_manager`` on Metashape's bundled 3.9/3.10.
    """

    def test_module_imports_without_tomllib(self):
        """Importing the module must succeed when ``tomllib`` is unavailable.

        Simulated in a fresh interpreter by blocking both ``tomllib`` and
        ``tomli`` (the <3.11 fallback) before importing.
        """
        import subprocess

        code = (
            "import sys\n"
            "sys.modules['tomllib'] = None\n"  # simulate Python < 3.11
            "sys.modules['tomli'] = None\n"  # no fallback available either
            "import pythontk.core_utils.package_manager as pm\n"
            "assert hasattr(pm, 'PackageManager')\n"
            "print('IMPORT_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"module import failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        self.assertIn("IMPORT_OK", result.stdout)

    def test_get_local_dependency_order_sorts_dependencies_first(self):
        """The toml parser is resolved locally and deps are ordered before dependents."""
        import textwrap

        with tempfile.TemporaryDirectory() as tmp:
            pkg_a = os.path.join(tmp, "pkg_a")
            pkg_b = os.path.join(tmp, "pkg_b")
            os.makedirs(pkg_a)
            os.makedirs(pkg_b)
            # pkg_a depends on pkg_b -> pkg_b must be ordered first.
            with open(os.path.join(pkg_a, "pyproject.toml"), "w") as f:
                f.write(
                    textwrap.dedent(
                        """
                        [project]
                        name = "pkg_a"
                        dependencies = ["pkg_b>=1.0"]
                        """
                    )
                )
            with open(os.path.join(pkg_b, "pyproject.toml"), "w") as f:
                f.write(
                    textwrap.dedent(
                        """
                        [project]
                        name = "pkg_b"
                        dependencies = []
                        """
                    )
                )

            order = PackageManager.get_local_dependency_order([pkg_a, pkg_b])
            names = [p.name for p in order]
            self.assertIn("pkg_a", names)
            self.assertIn("pkg_b", names)
            self.assertLess(names.index("pkg_b"), names.index("pkg_a"))


if __name__ == "__main__":
    unittest.main(exit=False)
