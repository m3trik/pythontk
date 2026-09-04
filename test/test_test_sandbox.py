# !/usr/bin/python
# coding=utf-8
"""Tests for :class:`pythontk.TestSandbox` -- the process-level test isolation.

The runner activates the sandbox before discovery, so these run INSIDE it and
assert what it does rather than construct a second one; a standalone run of
this module activates it in ``setUp``.
"""

import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
import webbrowser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythontk.core_utils.test_sandbox import TestSandbox
from pythontk.file_utils.temp_artifacts import TempArtifacts

URL = "http://127.0.0.1:1/"


class TestSandboxTestCase(unittest.TestCase):
    def setUp(self):
        TestSandbox.activate()
        self._launches_before = len(TestSandbox.launches)

    def tearDown(self):
        # A refusal this test provoked on purpose must not read as a leak in
        # the runner's end-of-run report.
        del TestSandbox.launches[self._launches_before :]

    def test_a_browser_launch_is_refused_loudly_and_recorded(self):
        for name in ("open", "open_new", "open_new_tab"):
            with self.subTest(launcher=name):
                with self.assertRaises(RuntimeError) as caught:
                    getattr(webbrowser, name)(URL)
                self.assertIn("open_browser=False", str(caught.exception))
        self.assertEqual(TestSandbox.launches[self._launches_before :], [URL] * 3)

    def test_a_test_that_patches_the_launcher_gets_its_mock_then_the_guard_back(self):
        with unittest.mock.patch("webbrowser.open", return_value=True) as opened:
            self.assertTrue(webbrowser.open(URL))
        opened.assert_called_once_with(URL)
        self.assertTrue(TestSandbox.is_active())

    def test_the_preview_server_cannot_open_a_tab_from_here(self):
        """The guard reaches the one production launcher the suites hit."""
        from pythontk.net_utils.preview.server import PreviewServer

        server = PreviewServer(port=0)
        try:
            with self.assertRaises(RuntimeError):
                server.open_in_browser()
        finally:
            server.stop()

    def test_the_temp_dir_is_one_throwaway_root_that_children_inherit(self):
        root = TestSandbox.temp()
        self.assertTrue(os.path.isdir(root))
        self.assertEqual(tempfile.gettempdir(), root)
        for name in ("TMPDIR", "TEMP", "TMP"):
            self.assertEqual(os.environ[name], root)
        # Every default-dir store lands inside it...
        store = TempArtifacts("sandbox_probe", policy="scoped")
        self.assertEqual(os.path.dirname(store.path()), root)
        store.cleanup()
        # ...and so does a child process's.
        child = subprocess.check_output(
            [sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"],
            text=True,
        ).strip()
        self.assertEqual(os.path.normcase(child), os.path.normcase(root))

    def test_activate_is_idempotent(self):
        self.assertEqual(TestSandbox.activate(), TestSandbox.activate())
        self.assertTrue(TestSandbox.is_active())


if __name__ == "__main__":
    unittest.main()
