#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk StatusBadge -- the ecosystem-wide test-badge SSoT.

Run with:
    python -m pytest test_status_badge.py -v
    python test_status_badge.py
"""
import unittest
from pathlib import Path

from pythontk.core_utils.status_badge import StatusBadge

from conftest import BaseTestCase

TEMP_DIR = Path(__file__).parent / "temp_tests"


class StatusBadgeTest(BaseTestCase):
    """StatusBadge test class."""

    def setUp(self):
        TEMP_DIR.mkdir(exist_ok=True)
        self.readme = TEMP_DIR / "status_badge_readme.md"

    def tearDown(self):
        if self.readme.exists():
            self.readme.unlink()

    def _write(self, content: str) -> Path:
        self.readme.write_text(content, encoding="utf-8")
        return self.readme

    # ------------------------------------------------------------- semantics

    def test_all_passing_is_green(self):
        """No failures -> '<n> passed', brightgreen."""
        self.assertEqual(
            StatusBadge.test_status(2194, 0), ("2194 passed", "brightgreen")
        )

    def test_mixed_run_is_orange(self):
        """Some passed, some failed -> both counts, orange."""
        self.assertEqual(
            StatusBadge.test_status(12, 3), ("12 passed, 3 failed", "orange")
        )

    def test_total_failure_is_red(self):
        """Nothing passed -> failures only, red."""
        self.assertEqual(StatusBadge.test_status(0, 7), ("7 failed", "red"))

    def test_zero_zero_run_is_grey_not_green(self):
        """Nothing ran -> unknown. A green badge for no results is the worst lie."""
        self.assertEqual(StatusBadge.test_status(0, 0), ("0 passed", "lightgrey"))

    # ----------------------------------------------------------------- url

    def test_url_escapes_shields_separators(self):
        """Spaces, commas, dashes and underscores are escaped for shields.io."""
        url = StatusBadge.url("12 passed, 3 failed", "orange")
        self.assertEqual(
            url,
            "https://img.shields.io/badge/Tests-12%20passed%2C%203%20failed-orange.svg",
        )

    def test_style_replaces_the_svg_suffix(self):
        """A style query and the .svg suffix are mutually exclusive."""
        url = StatusBadge.url("5 passed", "brightgreen", style="flat-square")
        self.assertTrue(url.endswith("-brightgreen?style=flat-square"))
        self.assertNotIn(".svg", url)

    def test_render_links_when_given_a_target(self):
        """Alt text is the label; the image is wrapped only when linked."""
        self.assertEqual(
            StatusBadge.render("5 passed", "brightgreen", link="../test/"),
            "[![Tests](https://img.shields.io/badge/Tests-5%20passed-brightgreen.svg)]"
            "(../test/)",
        )
        self.assertTrue(
            StatusBadge.render("5 passed", "brightgreen").startswith("![Tests](")
        )

    # -------------------------------------------------------------- update

    def test_replaces_an_existing_badge_in_place(self):
        """An existing Tests badge is rewritten where it stands."""
        self._write(
            "[![License](https://img.shields.io/badge/License-MIT-blue.svg)](x)\n"
            "[![Tests](https://img.shields.io/badge/Tests-1%20passed-brightgreen.svg)]"
            "(../test/)\n"
            "\n# pkg\n"
        )
        StatusBadge.update_test_badge(self.readme, passed=99, failed=0, link="../test/")

        content = self.readme.read_text(encoding="utf-8")
        self.assertIn("Tests-99%20passed-brightgreen", content)
        self.assertNotIn("Tests-1%20passed", content)
        self.assertEqual(content.count("img.shields.io/badge/Tests-"), 1)
        self.assertIn("License-MIT", content)

    def test_migrates_a_legacy_badge_regardless_of_case_or_alt_text(self):
        """Lowercase labels and bare (unlinked) images are migrated, not duplicated."""
        for legacy in (
            "[![Tests](https://img.shields.io/badge/tests-3%20passed-brightgreen.svg)](t)",
            "![Status](https://img.shields.io/badge/Tests-0%2F0_Passing-lightgrey)",
        ):
            with self.subTest(legacy=legacy):
                self._write(f"{legacy}\n\n# pkg\n")
                StatusBadge.update_test_badge(self.readme, 42, 0, link="../test/")

                content = self.readme.read_text(encoding="utf-8")
                self.assertEqual(content.count("img.shields.io/badge/"), 1)
                self.assertIn(
                    "[![Tests](https://img.shields.io/badge/Tests-42%20passed"
                    "-brightgreen.svg)](../test/)",
                    content,
                )

    def test_a_label_needing_escaping_still_replaces_itself(self):
        """Repeated updates converge: the matcher must see the ENCODED label."""
        self._write("# pkg\n")
        for count in (1, 2, 3):
            StatusBadge.update(
                self.readme, f"{count} passed", "brightgreen", label="Unit Tests"
            )

        content = self.readme.read_text(encoding="utf-8")
        self.assertEqual(content.count("img.shields.io/badge/"), 1)
        self.assertIn("Unit%20Tests-3%20passed", content)

    def test_inserts_after_an_existing_badge_block(self):
        """A first-time badge joins the badge row rather than jumping the title."""
        self._write(
            "[![License](https://img.shields.io/badge/License-MIT-blue.svg)](x)\n"
            "[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](y)\n"
            "\n# pkg\n"
        )
        StatusBadge.update_test_badge(self.readme, 7, 0, link="../test/")

        lines = self.readme.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines[2]), len(lines[2].strip()))
        self.assertIn("Tests-7%20passed", lines[2])
        self.assertTrue(lines[0].startswith("[![License]"))

    def test_inserts_at_the_top_when_there_is_no_badge_block(self):
        """A README with no badges gets one above the title."""
        self._write("# blendertk\n\nBlender utilities.\n")
        StatusBadge.update_test_badge(self.readme, 50, 0, link="../test/")

        content = self.readme.read_text(encoding="utf-8")
        self.assertTrue(content.startswith("[![Tests]("))
        self.assertIn("\n# blendertk\n", content)

    def test_link_is_relative_to_the_readme_not_the_cwd(self):
        """docs/README.md -> ../test/, computed from the README's own location."""
        docs = TEMP_DIR / "docs"
        docs.mkdir(exist_ok=True)
        readme = docs / "README.md"
        readme.write_text("# pkg\n", encoding="utf-8")
        try:
            StatusBadge.update_test_badge(readme, 5, 0, test_dir=TEMP_DIR / "test")
            self.assertIn(
                "](../test/)", readme.read_text(encoding="utf-8")
            )
        finally:
            readme.unlink()
            docs.rmdir()

    def test_missing_readme_reports_failure_without_raising(self):
        """A runner on a repo with no README must not crash the test run."""
        self.assertFalse(
            StatusBadge.update_test_badge(TEMP_DIR / "nope.md", 1, 0, link="t")
        )

    def test_unwritable_readme_reports_failure_without_raising(self):
        """A locked / cloud-sync-stalled README must not fail a green test run."""
        self._write("# pkg\n")
        real_write = Path.write_text

        def boom(self, *args, **kwargs):
            raise OSError(22, "Invalid argument")

        Path.write_text = boom
        try:
            self.assertFalse(StatusBadge.update_test_badge(self.readme, 5, 0, link="t"))
        finally:
            Path.write_text = real_write


class RunCompletenessGateTest(BaseTestCase):
    """The gate that decides whether a run may stamp the badge.

    Regression cover for the idiom-dependence defect: unittest reports a
    class/module-level ``SkipTest`` through ``unittest.suite._ErrorHolder``,
    whose ``__module__`` is ``unittest.suite``. Treating that like a
    ``unittest.loader`` stand-in (a module that never imported) meant the badge
    was refused forever for any module whose cases are all setUpClass-gated,
    while the same skip written as ``@unittest.skipUnless`` on the class stayed
    green. Whether a run reads green must not depend on which skip idiom the
    test author reached for -- this module promises that an all-green run with
    environment-gated skips still reads green.
    """

    SETUPCLASS_PROBE = """
import unittest


class TestGated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raise unittest.SkipTest('env gate')

    def test_a(self):
        pass
"""

    SETUPMODULE_PROBE = """
import unittest


def setUpModule():
    raise unittest.SkipTest('env gate')


class TestGated(unittest.TestCase):
    def test_a(self):
        pass
"""

    def _modules_seen(self, name, source):
        """Run *source* as a module through unittest; return credited names."""
        import os
        import sys
        import types

        module = types.ModuleType(name)
        exec(compile(source, name + '.py', 'exec'), module.__dict__)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)

        suite = unittest.TestLoader().loadTestsFromModule(module)
        with open(os.devnull, 'w') as sink:
            result = unittest.TextTestRunner(stream=sink, verbosity=0).run(suite)

        cases = list(result.skipped) + list(result.errors) + list(result.failures)
        return {
            StatusBadge.module_of(case)
            for case, _ in cases
            if not StatusBadge.is_import_standin(case)
        }

    def test_setupclass_skip_credits_its_module(self):
        seen = self._modules_seen('test_gate_probe_setupclass', self.SETUPCLASS_PROBE)
        self.assertEqual(seen, {'test_gate_probe_setupclass'})

    def test_setupmodule_skip_credits_its_module(self):
        seen = self._modules_seen('test_gate_probe_setupmodule', self.SETUPMODULE_PROBE)
        self.assertEqual(seen, {'test_gate_probe_setupmodule'})

    def test_an_environment_gated_module_does_not_block_the_badge(self):
        """The whole point: a green run carrying such skips still stamps."""
        seen = self._modules_seen('test_gate_probe_allowed', self.SETUPCLASS_PROBE)
        allowed, reason = StatusBadge.gate(
            {'test_gate_probe_allowed'}, seen, passed=12, failed=0
        )
        self.assertTrue(allowed, reason)

    def test_a_module_that_never_imported_still_blocks_the_badge(self):
        """The case the gate exists for must keep working."""
        allowed, reason = StatusBadge.gate(
            {'test_a', 'test_b'}, {'test_a'}, passed=9, failed=0
        )
        self.assertFalse(allowed)
        self.assertIn('test_b', reason)

    def test_a_run_with_no_cases_at_all_blocks_the_badge(self):
        allowed, reason = StatusBadge.gate(set(), set(), passed=0, failed=0)
        self.assertFalse(allowed)
        self.assertIn('no test cases ran', reason)

if __name__ == "__main__":
    unittest.main(exit=False)
